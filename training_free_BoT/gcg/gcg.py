import gc
import json
import os
import sys

import torch
from accelerate.utils import find_executable_batch_size
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

sys.path.append("../")
sys.path.append(os.getcwd())

import logging

import matplotlib.pyplot as plt

# Import only the RedTeamingMethod base class to avoid circular imports
from gcg.baseline import SingleBehaviorRedTeamingMethod
from gcg.check_refusal_utils import check_thinking_completions
from gcg.model_utils import get_template
from training_free_BoT.gcg.gcg_utils import get_nonascii_toks, sample_control


# ============================== GCG CLASS DEFINITION ============================== #
class GCG(SingleBehaviorRedTeamingMethod):
    def __init__(
        self,
        target_model,
        targets_path,
        num_steps=50,
        adv_suffix_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        allow_non_ascii=False,
        search_width=512,
        use_prefix_cache=False,
        eval_steps=10,
        eval_with_check_refusal=True,
        check_refusal_min_loss=0.1,
        early_stopping=False,
        early_stopping_min_loss=0.1,
        starting_search_batch_size=512,
        save_dir=None,
        suffix_position="back",
        **model_kwargs,
    ):
        """
        :param target_model: a dictionary specifying the target model (kwargs to load_model_and_tokenizer)
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with
        :param allow_non_ascii: whether to allow non-ascii characters when sampling candidate updates
        :param search_width: the number of candidates to sample at each step
        :param use_prefix_cache: whether to use KV cache for static prefix tokens (WARNING: Not all models support this)
        :param eval_steps: the number of steps between each evaluation
        :param eval_with_check_refusal: whether to generate and check for refusal every eval_steps; used for early stopping
        :param check_refusal_min_loss: a minimum loss before checking for refusal
        :param early_stopping: an alternate form of early stopping, based solely on early_stopping_min_loss
        :param early_stopping_min_loss: the loss at which to stop optimization if early_stopping is True
        :param starting_search_batch_size: the initial search_batch_size; will auto reduce later in case of OOM
        :param suffix_position: where to add the suffix, either "front" or "back" (default: "back")
        """
        super().__init__(
            target_model=target_model, **model_kwargs
        )  # assigns self.model, self.tokenizer, and self.model_name_or_path
        self.num_steps = num_steps
        self.adv_string_init = adv_suffix_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.use_prefix_cache = use_prefix_cache
        self.suffix_position = suffix_position
        if use_prefix_cache and self.model.config.use_cache != use_prefix_cache:
            logging.info(f"WARNING: setting model.config.use_cache={use_prefix_cache}")
            self.model.config.use_cache = use_prefix_cache

        with open(targets_path, "r", encoding="utf-8") as file:
            self.behavior_id_to_target = json.load(file)

        ### Eval Vars ###
        self.eval_steps = eval_steps
        self.eval_with_check_refusal = eval_with_check_refusal
        self.check_refusal_min_loss = check_refusal_min_loss
        self.early_stopping = early_stopping
        self.early_stopping_min_loss = early_stopping_min_loss
        self.starting_search_batch_size = starting_search_batch_size

        ## Instruction template
        template = get_template(**target_model)["prompt"]
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")
        self.save_dir = save_dir

    def generate_test_cases_single_query_single_model(
        self, behavior_dict, verbose=False, **kwargs
    ):
        """
        Generates test cases for a single behavior

        :param behavior: a dictionary specifying the behavior to generate test cases for
        :param verbose: whether to print progress
        :return: a test case and logs
        """

        # starting search_batch_size, will auto reduce batch_size later if go OOM (resets for every new behavior)
        self.search_batch_size = (
            self.starting_search_batch_size
            if self.starting_search_batch_size
            else self.search_width
        )

        # ========== Behavior and Target str ==========
        behavior = behavior_dict["Behavior"]
        context_str = behavior_dict["ContextString"]
        behavior_id = behavior_dict["BehaviorID"]

        target = self.behavior_id_to_target[behavior_id]
        if self.suffix_position == "back":
            behavior += " "
        else:
            behavior = " " + behavior
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        logging.info(f"Behavior: {behavior_id} || Target: {target}")
        ### Targeted Model and Tokenier ###
        model = self.model
        device = model.device
        tokenizer = self.tokenizer

        ### GCG hyperparams ###
        num_steps = self.num_steps
        adv_string_init = self.adv_string_init
        allow_non_ascii = self.allow_non_ascii
        search_width = self.search_width
        before_tc = self.before_tc
        after_tc = self.after_tc

        ### Eval vars ###
        eval_steps = self.eval_steps
        eval_with_check_refusal = self.eval_with_check_refusal
        check_refusal_min_loss = self.check_refusal_min_loss
        early_stopping = self.early_stopping
        early_stopping_min_loss = self.early_stopping_min_loss

        embed_layer = model.get_input_embeddings()
        vocab_size = embed_layer.weight.shape[
            0
        ]  # can be larger than tokenizer.vocab_size for some models
        vocab_embeds = embed_layer(torch.arange(0, vocab_size).long().to(model.device))

        not_allowed_tokens = None if allow_non_ascii else get_nonascii_toks(tokenizer)

        if not_allowed_tokens is None:
            nonascii_toks = []
            if tokenizer.bos_token_id is not None:
                nonascii_toks.append(tokenizer.bos_token_id)
            if tokenizer.eos_token_id is not None:
                nonascii_toks.append(tokenizer.eos_token_id)
            if tokenizer.pad_token_id is not None:
                nonascii_toks.append(tokenizer.pad_token_id)
            if tokenizer.unk_token_id is not None:
                nonascii_toks.append(tokenizer.unk_token_id)
            not_allowed_tokens = torch.tensor(nonascii_toks, device="cpu")
        not_allowed_tokens = torch.unique(not_allowed_tokens)

        # ========== setup optim_ids and cached components ========== #
        optim_ids = tokenizer(
            adv_string_init, return_tensors="pt", add_special_tokens=False
        ).to(device)["input_ids"]
        num_optim_tokens = len(optim_ids[0])

        cache_input_ids = tokenizer([before_tc], padding=False)[
            "input_ids"
        ]  # some tokenizer have <s> for before_tc
        cache_input_ids += tokenizer(
            [behavior, after_tc, target], padding=False, add_special_tokens=False
        )["input_ids"]
        cache_input_ids = [
            torch.tensor(input_ids, device=device).unsqueeze(0)
            for input_ids in cache_input_ids
        ]  # make tensor separately because can't return_tensors='pt' in tokenizer
        before_ids, behavior_ids, after_ids, target_ids = cache_input_ids
        before_embeds, behavior_embeds, after_embeds, target_embeds = [
            embed_layer(input_ids) for input_ids in cache_input_ids
        ]

        if self.use_prefix_cache:
            # precompute KV cache for everything before the optimized tokens
            input_embeds = torch.cat([before_embeds, behavior_embeds], dim=1)
            with torch.no_grad():
                outputs = model(inputs_embeds=input_embeds, use_cache=True)
                self.prefix_cache = outputs.past_key_values

        # ========== run optimization ========== #
        all_losses = []
        all_test_cases = []
        step_logs = []  # New list for storing logs in the requested format

        completions = None
        for i in tqdm(range(num_steps)):
            # ========== compute coordinate token_gradient ========== #
            # create input
            optim_ids_onehot = torch.zeros(
                (1, num_optim_tokens, vocab_size), device=device, dtype=model.dtype
            )
            optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
            optim_embeds = torch.matmul(
                optim_ids_onehot.squeeze(0), vocab_embeds
            ).unsqueeze(0)

            # forward pass
            if self.use_prefix_cache:
                input_embeds = torch.cat(
                    [optim_embeds, after_embeds, target_embeds], dim=1
                )
                outputs = model(
                    inputs_embeds=input_embeds, past_key_values=self.prefix_cache
                )
            else:
                if self.suffix_position == "front":
                    input_embeds = torch.cat(
                        [
                            before_embeds,
                            optim_embeds,
                            behavior_embeds,
                            after_embeds,
                            target_embeds,
                        ],
                        dim=1,
                    )
                else:  # default is "back"
                    input_embeds = torch.cat(
                        [
                            before_embeds,
                            behavior_embeds,
                            optim_embeds,
                            after_embeds,
                            target_embeds,
                        ],
                        dim=1,
                    )
                outputs = model(inputs_embeds=input_embeds)

            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds.shape[1] - target_embeds.shape[1]
            shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
            shift_labels = target_ids
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            token_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[
                0
            ]

            # ========== Sample a batch of new tokens based on the coordinate gradient. ========== #
            sampled_top_indices = sample_control(
                optim_ids.squeeze(0),
                token_grad.squeeze(0),
                search_width,
                topk=256,
                temp=1,
                not_allowed_tokens=not_allowed_tokens,
            )

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_text = tokenizer.batch_decode(sampled_top_indices)
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                # tokenize again
                tmp = tokenizer(
                    sampled_top_indices_text[j],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(device)["input_ids"][0]
                # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                if not torch.equal(tmp, sampled_top_indices[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])

            if len(new_sampled_top_indices) == 0:
                logging.info("All removals; defaulting to keeping all")
                count = 0
            else:
                sampled_top_indices = torch.stack(new_sampled_top_indices)

            if count >= search_width // 2:
                logging.info("\nLots of removals:", count)

            new_search_width = search_width - count

            # ========== Compute loss on these candidates and take the argmin. ========== #
            # create input
            sampled_top_embeds = embed_layer(sampled_top_indices)
            if self.use_prefix_cache:
                input_embeds = torch.cat(
                    [
                        sampled_top_embeds,
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ],
                    dim=1,
                )
            else:
                if self.suffix_position == "front":
                    input_embeds = torch.cat(
                        [
                            before_embeds.repeat(new_search_width, 1, 1),
                            sampled_top_embeds,
                            behavior_embeds.repeat(new_search_width, 1, 1),
                            after_embeds.repeat(new_search_width, 1, 1),
                            target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )
                else:  # default is "back"
                    input_embeds = torch.cat(
                        [
                            before_embeds.repeat(new_search_width, 1, 1),
                            behavior_embeds.repeat(new_search_width, 1, 1),
                            sampled_top_embeds,
                            after_embeds.repeat(new_search_width, 1, 1),
                            target_embeds.repeat(new_search_width, 1, 1),
                        ],
                        dim=1,
                    )

            # Auto Find Batch Size for foward candidates (each time go OOM will decay search_batch_size // 2)
            loss = find_executable_batch_size(
                self.compute_candidates_loss, self.search_batch_size
            )(input_embeds, target_ids)

            # Record all candidate suffixes and their losses
            all_candidates = []
            for j in range(len(sampled_top_indices)):
                candidate_suffix = tokenizer.decode(sampled_top_indices[j])
                candidate_loss = loss[j].item()
                all_candidates.append([candidate_suffix, candidate_loss])

            # ========== Update the optim_ids with the best candidate ========== #
            best_idx = loss.argmin().item()
            optim_ids = sampled_top_indices[best_idx].unsqueeze(0)
            optim_str = tokenizer.decode(optim_ids[0])
            current_loss = loss[best_idx].item()

            # Create step log
            step_log = {
                "step": i,
                "all_candidates": all_candidates,
                "optim_suffix": [optim_str, current_loss],
            }
            step_logs.append(step_log)

            # Construct test case based on suffix_position
            if self.suffix_position == "front":
                test_case_ids = torch.cat([optim_ids, behavior_ids], dim=1)
            else:  # default is "back"
                test_case_ids = torch.cat([behavior_ids, optim_ids], dim=1)

            test_case = tokenizer.decode(test_case_ids[0])
            all_test_cases.append(test_case)
            all_losses.append(current_loss)

            # ========== Eval and Early Stopping ========== #
            if (i % eval_steps == 0) or (i == num_steps - 1):
                p_output = f"\n===>Step {i}\n===>Optim Suffix: {optim_str}\n===>Loss: {current_loss}"
                # Save logs at each eval step in the requested format
                self.save_step_logs(step_logs, i)
                if eval_with_check_refusal and current_loss < check_refusal_min_loss:
                    # input_str = self.template.format(instruction=test_case)
                    input_str = self.template.replace("{instruction}", test_case)
                    is_thinking, completions, _ = check_thinking_completions(
                        model, tokenizer, inputs=[input_str]
                    )
                    p_output += f"\n===>Completion: {completions[0]}"
                    logging.info(p_output)
                    if not is_thinking[0]:
                        break

                if verbose:
                    logging.info(p_output)

            if early_stopping and current_loss < early_stopping_min_loss:
                logging.info(f"Early stopping at step {i} with loss {current_loss}")
                logging.info("")
                break

            del input_embeds, sampled_top_embeds, logits, shift_logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        logs = {
            "final_loss": current_loss,
            "all_losses": all_losses,
            "all_test_cases": all_test_cases,
        }

        return optim_str, test_case, logs, completions

    def compute_candidates_loss(self, search_batch_size, input_embeds, target_ids):
        if self.search_batch_size != search_batch_size:
            logging.info(
                f"INFO: Setting candidates search_batch_size to {search_batch_size})"
            )
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        all_loss = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]

                if self.use_prefix_cache:
                    # Expand prefix cache to match batch size
                    prefix_cache_batch = self.prefix_cache
                    current_batch_size = input_embeds_batch.shape[0]
                    prefix_cache_batch = []
                    for i in range(len(self.prefix_cache)):
                        prefix_cache_batch.append([])
                        for j in range(len(self.prefix_cache[i])):
                            prefix_cache_batch[i].append(
                                self.prefix_cache[i][j].expand(
                                    current_batch_size, -1, -1, -1
                                )
                            )
                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=prefix_cache_batch,
                    )
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)
            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)

            del outputs, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        return torch.cat(all_loss, dim=0)

    def save_step_logs(self, step_logs, current_step):
        """Save step logs in the requested format."""
        if self.save_dir:
            with open(os.path.join(self.save_dir, "logs.json"), "w") as f:
                json.dump(step_logs, f)

            # Also plot the loss curve
            steps = [log["step"] for log in step_logs]
            losses = [log["optim_suffix"][1] for log in step_logs]

            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, label="Loss")
            plt.title(f"Loss vs. Iteration (Current Step {current_step})")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend()

            plot_file = os.path.join(self.save_dir, f"losses.png")
            plt.savefig(plot_file)
            plt.close()

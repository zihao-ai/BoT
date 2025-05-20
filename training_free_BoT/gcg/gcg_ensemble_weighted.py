# TODO: support multi-node; current implementation with Ray only work for 1 node
import gc
import json
import logging
import os
import random
from typing import List

import ray
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append(os.getcwd())
from .baseline import RedTeamingMethod
from utils.utils import is_thinking

# from ..eval_utils import jailbroken
from .gcg_ray_actors import GCGModelWorker, get_nonascii_toks, sample_control


# ============================== EnsembleGCG CLASS DEFINITION ============================== #
class EnsembleGCG(RedTeamingMethod):
    use_ray = True

    def __init__(
        self,
        target_models,
        targets_path=None,
        num_steps=50,
        adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        allow_non_ascii=False,
        search_width=512,
        use_prefix_cache=True,
        eval_steps=10,
        progressive_behaviors=True,
        early_stop=False,
        early_stop_min_loss=0.1,
        eval_with_check_refusal=True,
        check_refusal_min_loss=0.2,
        save_dir=None,
        adaptive_weighting=False,
        adaptive_temp=1.0,
        **kwargs,
    ):
        """
        :param target_models: a list of target model configurations (dictionaries containing num_gpus and kwargs for GCGModelWorker)
        :param targets_path: the path to the targets JSON file
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with
        :param allow_non_ascii: whether to allow non-ascii characters when sampling candidate updates
        :param search_width: the number of candidates to sample at each step
        :param use_prefix_cache: whether to use KV cache for static prefix tokens (WARNING: Not all models support this)
        :param eval_steps: the number of steps between each evaluation
        :param progressive_behaviors: whether to progressively add behaviors to optimize or optimize all at once
        :param adaptive_weighting: whether to adaptively adjust model weights based on previous loss
        :param adaptive_temp: temperature for adaptive weighting (higher = more aggressive weight adjustment)
        """
        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.use_prefix_cache = use_prefix_cache
        self.early_stop = early_stop
        self.early_stop_min_loss = early_stop_min_loss
        self.eval_with_check_refusal = eval_with_check_refusal
        self.check_refusal_min_loss = check_refusal_min_loss
        self.save_dir = save_dir
        self.adaptive_weighting = adaptive_weighting
        self.adaptive_temp = adaptive_temp

        if targets_path is None:
            raise ValueError("targets_path must be specified")
        with open(targets_path, "r", encoding="utf-8") as file:
            self.behavior_id_to_target = json.load(file)
        self.targets_path = targets_path

        self.eval_steps = eval_steps
        self.progressive_behaviors = progressive_behaviors

        ## Init workers accross gpus with ray
        num_gpus_list = [m.get("num_gpus", 1) for m in target_models]
        target_models = [m["target_model"] for m in target_models]
        self.models = target_models

        # Initialize model weights with equal values
        self.model_weights = [1.0 / len(target_models)] * len(target_models)

        self.gcg_models = []
        # Allocate GPUs to workers based on their requirements of gpu
        for model_config, required_gpus in zip(target_models, num_gpus_list):
            # Create a GCGModelWorker instance with allocated GPUs
            worker_actor = GCGModelWorker.options(num_gpus=required_gpus).remote(
                search_width=search_width,
                use_prefix_cache=use_prefix_cache,
                **model_config,
            )
            self.gcg_models.append(worker_actor)

        initialization_models = [w.is_initialized.remote() for w in self.gcg_models]
        logging.info("\n".join(ray.get(initialization_models)))

        self.tokenizers = ray.get(
            [w.get_attribute_.remote("tokenizer") for w in self.gcg_models]
        )
        # Check that all tokenizers are the same type
        assert (
            len(set([type(t) for t in self.tokenizers])) == 1
        ), "All tokenizers must be the same type"
        self.sample_tokenizer = self.tokenizers[
            0
        ]  # Used for initializing optimized tokens and decoding result; assumes all tokenizers are the same
        not_allowed_tokens = (
            None if allow_non_ascii else get_nonascii_toks(self.sample_tokenizer)
        )

        if not_allowed_tokens is None:
            nonascii_toks = []
            if self.sample_tokenizer.bos_token_id is not None:
                nonascii_toks.append(self.sample_tokenizer.bos_token_id)
            if self.sample_tokenizer.eos_token_id is not None:
                nonascii_toks.append(self.sample_tokenizer.eos_token_id)
            if self.sample_tokenizer.pad_token_id is not None:
                nonascii_toks.append(self.sample_tokenizer.pad_token_id)
            if self.sample_tokenizer.unk_token_id is not None:
                nonascii_toks.append(self.sample_tokenizer.unk_token_id)
            not_allowed_tokens = torch.tensor(nonascii_toks, device="cpu")
        self.not_allowed_tokens = torch.unique(not_allowed_tokens)

    def save_and_plot_loss(self, step_logs, current_step):
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

    def update_model_weights(self, model_losses):
        """
        Update model weights based on their losses - models with higher loss get higher weight
        to prioritize their optimization in the next round.

        :param model_losses: Tensor of model losses for the current step
        :return: Updated normalized weights
        """
        # Extract minimum loss value for each model
        losses = torch.tensor(
            [ml.min().item() for ml in model_losses], device=model_losses.device
        )

        # Adjust with temperature to control weight adaptation aggressiveness
        weights = losses**self.adaptive_temp

        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        # Convert to list
        return weights.tolist()

    def generate_test_cases(
        self, behaviors, verbose=False, bot="<think>", eot="</think>"
    ):
        """
        Generates test cases by jointly optimizing over the provided behaviors. A test case for
        each behavior is returned at the end.

        :param behaviors: a list of behavior dictionaries specifying the behaviors to generate test cases for
        :param verbose: whether to logging.info progress
        :return: a dictionary of test cases, where the keys are the behavior IDs and the values are lists of test cases
        """

        # ===================== generate test cases ===================== #
        behavior_list = [b["Behavior"] for b in behaviors]
        context_str_list = [b["ContextString"] for b in behaviors]
        behavior_id_list = [b["BehaviorID"] for b in behaviors]

        targets = [
            self.behavior_id_to_target[behavior_id] for behavior_id in behavior_id_list
        ]
        behavior_list = [
            b + " " for b in behavior_list
        ]  # add a space before the optimized string

        # Adding context string to behavior string if available
        tmp = []
        for behavior, context_str in zip(behavior_list, context_str_list):
            if context_str:
                behavior = f"{context_str}\n\n---\n\n{behavior}"
            tmp.append(behavior)
        behavior_list = tmp

        total_behaviors = len(behavior_list)
        num_behaviors = 1 if self.progressive_behaviors else total_behaviors
        all_behavior_indices = list(range(total_behaviors))

        gcg_models = self.gcg_models
        # Init behaviors, targets, embeds
        ray.get(
            [
                w.init_behaviors_and_targets.remote(behavior_list, targets)
                for w in gcg_models
            ]
        )

        optim_ids = self.sample_tokenizer(
            self.adv_string_init, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        num_optim_tokens = optim_ids.shape[1]

        step_logs = []
        # Initialize steps_since_last_added_behavior
        steps_since_last_added_behavior = 0
        all_model_losses = []
        all_model_weights = [self.model_weights.copy()]  # Track weight history

        # ========== run optimization ========== #
        complations = None
        for step_i in tqdm(range(self.num_steps)):
            # 清空不用的显存
            torch.cuda.empty_cache()
            # select behaviors to use in search step
            current_behavior_indices = all_behavior_indices[:num_behaviors]

            # ========= Get normalized token gradients for all models and average across models ========= #
            futures = []
            for worker in gcg_models:
                futures.append(
                    worker.get_token_gradient.remote(
                        optim_ids, num_optim_tokens, current_behavior_indices
                    )
                )
            grads = ray.get(futures)

            # clipping extra tokens
            min_embedding_value = min(grad.shape[2] for grad in grads)
            grads = [grad[:, :, :min_embedding_value] for grad in grads]

            # Apply current model weights
            token_grad = sum(
                grad * weight for grad, weight in zip(grads, self.model_weights)
            )

            # ========= Select candidate replacements ========= #
            sampled_top_indices = sample_control(
                optim_ids.squeeze(0),
                token_grad.squeeze(0),
                self.search_width,
                topk=256,
                temp=1,
                not_allowed_tokens=self.not_allowed_tokens,
            )  # size: (search_width, num_optim_tokens)

            # ========= Filter sampled_top_indices that aren't the same after detokenize->retokenize ========= #
            sampled_top_indices_text = self.sample_tokenizer.batch_decode(
                sampled_top_indices
            )
            new_sampled_top_indices = []
            count = 0
            for j in range(len(sampled_top_indices_text)):
                # tokenize again
                tmp = self.sample_tokenizer(
                    sampled_top_indices_text[j],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(sampled_top_indices.device)["input_ids"][0]
                # if the tokenized text is different, then set the sampled_top_indices to padding_top_indices
                if not torch.equal(tmp, sampled_top_indices[j]):
                    count += 1
                    continue
                else:
                    new_sampled_top_indices.append(sampled_top_indices[j])

            sampled_top_indices = torch.stack(new_sampled_top_indices)

            if count >= self.search_width // 2:
                logging.info("\nLots of removals:", count)

            # ========= Evaluate candidate optim ========= #
            futures = []
            for worker in gcg_models:
                future = worker.compute_loss_on_candidates_batch.remote(
                    sampled_top_indices, current_behavior_indices
                )
                futures.append(future)

            losses = ray.get(futures)
            losses = torch.stack(
                losses
            )  # size: (num_models, num_behaviors, search_width)

            # mean across all behaviors for each model
            model_losses = losses.mean(1)

            # Convert weights to tensor matching model_losses dimensions
            weights_tensor = torch.tensor(
                self.model_weights, device=model_losses.device
            ).view(-1, 1)
            # Weighted average across models (dimension 0)
            weighted_model_losses = (model_losses * weights_tensor).sum(0)
            # For logging
            avg_model_losses = weighted_model_losses

            # mean across all behaviors
            behavior_losses = losses.mean(0)

            selected_id = avg_model_losses.argmin().item()

            # Record all candidate suffixes and their losses
            all_candidates = []
            for j in range(len(sampled_top_indices)):
                candidate_suffix = self.sample_tokenizer.decode(sampled_top_indices[j])
                candidate_loss = avg_model_losses[j].item()
                all_candidates.append([candidate_suffix, candidate_loss])

            # Update model weights if adaptive weighting is enabled
            if self.adaptive_weighting:
                self.model_weights = self.update_model_weights(model_losses)
                all_model_weights.append(self.model_weights)

            # ========== Update the optim_ids with the best candidate ========== #
            optim_ids = sampled_top_indices[selected_id].unsqueeze(0)
            optim_str = self.sample_tokenizer.decode(optim_ids[0])
            current_loss = avg_model_losses[selected_id].item()

            # Create step log with model weights info for the adaptive version
            step_log = {
                "step": step_i,
                "all_candidates": all_candidates,
                "optim_suffix": [optim_str, current_loss],
            }
            step_logs.append(step_log)

            # Also track model losses for debugging
            all_model_losses.append([ml.min().item() for ml in model_losses])

            # ========== Get test cases ========== #
            test_cases = {
                b_id: [b + optim_str]
                for b, b_id in zip(behavior_list, behavior_id_list)
            }

            if verbose and (
                (step_i % self.eval_steps == 0) or (step_i == self.num_steps - 1)
            ):
                _avg_loss = current_loss

                logging.info("\n" + "=" * 50)
                logging.info(
                    f"\nStep {step_i} | Optimized Suffix: {optim_str} | Avg Loss: {_avg_loss}\n"
                )

                for i in range(len(losses)):
                    model_name = self.models[i]["model_name_or_path"].split("/")[-1]
                    model_loss = model_losses[i].min().item()
                    logging.info(
                        f"==> Model: {model_name} | Loss {model_loss} (weight: {self.model_weights[i]:.3f})"
                    )

                # 打印每个Behavior的loss
                for i in range(num_behaviors):
                    logging.info(
                        f"==> Behavior: {i} | Loss: {behavior_losses[i][selected_id].item()}"
                    )

                self.save_and_plot_loss(step_logs, step_i)

                # 判断是否需要早停 eval_with_check_refusal
                if (
                    self.eval_with_check_refusal
                    and num_behaviors == total_behaviors
                    and _avg_loss < self.check_refusal_min_loss
                ):
                    all_no_thinking = True
                    for i in range(num_behaviors):
                        futures = []
                        for worker in gcg_models:
                            futures.append(worker.generate.remote(optim_str, i))
                        results = ray.get(futures)
                        complations = results
                        if not all(
                            is_thinking(result, bot=bot, eot=eot) for result in results
                        ):
                            all_no_thinking = False
                            break
                    if all_no_thinking:
                        logging.info(
                            f"\nEarly stop at step {step_i}, all behaviors have no thinking."
                        )
                        break

            steps_since_last_added_behavior += 1
            if self.progressive_behaviors and num_behaviors < total_behaviors:
                # if all(behavior_losses[:, selected_id] < 0.2):
                if _avg_loss < self.check_refusal_min_loss:
                    all_no_thinking = True
                    for i in range(num_behaviors):
                        futures = []
                        for worker in gcg_models:
                            futures.append(worker.generate.remote(optim_str, i))
                        results = ray.get(futures)
                        complations = results
                        if not all(
                            is_thinking(result, bot=bot, eot=eot) for result in results
                        ):
                            all_no_thinking = False
                            break
                    if all_no_thinking:
                        num_behaviors += 1
                        logging.info(
                            f"\n😁😁😁😁😁Adding one more behavior to optimize, total: {num_behaviors}/{total_behaviors} (case 1)\n"
                        )
                    steps_since_last_added_behavior = 0
                elif steps_since_last_added_behavior == 512:
                    logging.info(
                        f"\n😱😱😱😱😱Adding one more behavior to optimize, total: {num_behaviors}/{total_behaviors} (case 2)\n"
                    )
                    num_behaviors += 1
                    steps_since_last_added_behavior = 0

            # 判断是否需要早停
            if (
                num_behaviors == total_behaviors
                and self.early_stop
                and weighted_model_losses[selected_id] < self.early_stop_min_loss
            ):
                logging.info(f"\nEarly stop at step {step_i}")
                break

            del grads, losses, model_losses, behavior_losses
            del weighted_model_losses, avg_model_losses
            torch.cuda.empty_cache()
            gc.collect()

        # Return step logs
        logs = step_logs
        return optim_str, test_cases, logs, complations

    # ===== Defining saving methods =====
    # Saves test cases for all behaviors into one test_cases.json file
    @staticmethod
    def get_output_file_path(save_dir, behavior_id, file_type, run_id=None):
        run_id = f"_{run_id}" if run_id else ""
        return os.path.join(save_dir, f"{file_type}{run_id}.json")

    def save_test_cases(
        self, save_dir, test_cases, logs=None, method_config=None, run_id=None
    ):
        test_cases_save_path = self.get_output_file_path(
            save_dir, None, "test_cases", run_id
        )
        logs_save_path = self.get_output_file_path(save_dir, None, "logs", run_id)
        method_config_save_path = self.get_output_file_path(
            save_dir, None, "method_config", run_id
        )

        os.makedirs(os.path.dirname(test_cases_save_path), exist_ok=True)
        if test_cases is not None:
            with open(test_cases_save_path, "w", encoding="utf-8") as f:
                json.dump(test_cases, f, indent=4)

        if logs is not None:
            with open(logs_save_path, "w", encoding="utf-8") as f:
                json.dump(logs, f)

        if method_config is not None:
            # mask token or api_key
            self._replace_tokens(method_config)
            with open(method_config_save_path, "w", encoding="utf-8") as f:
                method_config["dependencies"] = {
                    l.__name__: l.__version__ for l in self.default_dependencies
                }
                json.dump(method_config, f, indent=4)

    @staticmethod
    def merge_test_cases(save_dir):
        """
        Merges {save_dir}/test_cases_{run_id}.json into {save_dir}/test_cases.json
        """
        test_cases = {}
        logs = {}
        assert os.path.exists(save_dir), f"{save_dir} does not exist"

        # Find all subset directories
        run_ids = [
            d.split("_")[-1].split(".")[0]
            for d in os.listdir(save_dir)
            if d.startswith("test_cases_")
        ]

        # Load test cases from each test_cases_{run_id}.json file and merge
        for run_id in run_ids:
            test_cases_path = os.path.join(save_dir, f"test_cases_{run_id}.json")
            with open(test_cases_path, "r") as f:
                test_cases_part = json.load(f)
            for behavior_id in test_cases_part:
                if behavior_id in test_cases:
                    test_cases[behavior_id].extend(test_cases_part[behavior_id])
                else:
                    test_cases[behavior_id] = test_cases_part[behavior_id]

        # Save merged test cases and logs
        test_cases_save_path = os.path.join(save_dir, "test_cases.json")
        with open(test_cases_save_path, "w") as f:
            json.dump(test_cases, f, indent=4)

        logging.info(f"Saved test_cases.json to {test_cases_save_path}")

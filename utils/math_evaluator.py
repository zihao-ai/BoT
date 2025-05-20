from utils.api_models import GPT
from evalscope.benchmarks.aime.aime24_adapter import AIME24Adapter
from evalscope.benchmarks.math_500.math_500_adapter import Math500Adapter
from evalscope.metrics.llm_judge import LLMJudge


class MathEvaluator:

    def __init__(
        self,
        dataset: str,
        llm_judge: bool = False,
        judge_model_name: str = "gpt-4o-mini",
        api_key: str = "sk-xxxxx",
        api_url: str = "https://api.openai.com/v1",
    ):
        self.judge_model_name = judge_model_name
        self.api_key = api_key
        self.api_url = api_url
        self.dataset = dataset
        self.llm_judge = llm_judge

    def load_evalscope_adapter(self):
        if self.dataset == "math500":
            return Math500Adapter(
                name="math500",
                dataset_id="AI-ModelScope/MATH-500",
                model_adapter="gpt-4o-mini",
                subset_list=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
                metric_list=["AveragePass@1"],
            )
        elif self.dataset == "aime2024":
            return AIME24Adapter(
                name="aime24",
                dataset_id="AI-ModelScope/AIME24",
                model_adapter="gpt-4o-mini",
                subset_list=["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"],
                metric_list=["AveragePass@1"],
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

    def evaluate_acc(
        self, question: str, predicted_answer: str, correct_answer: str
    ) -> bool:
        adapter = self.load_evalscope_adapter()
        parsed_gold_answer = adapter.get_gold_answer({"answer": correct_answer})
        parsed_predicted_answer = adapter.parse_pred_result(predicted_answer)
        correct = adapter.match(parsed_gold_answer, parsed_predicted_answer)

        if not correct and self.llm_judge:
            llm_judge = LLMJudge(
                model_id=self.judge_model_name,
                api_key=self.api_key,
                api_url=self.api_url,
            )
            if parsed_predicted_answer != "":
                prompt = llm_judge.build_prompt(
                    pred=parsed_predicted_answer,
                    gold=parsed_gold_answer,
                    question=question,
                )
            else:
                prompt = llm_judge.build_prompt(
                    pred=predicted_answer, gold=parsed_gold_answer, question=question
                )
            llm_response = llm_judge(prompt)
            score = llm_judge.get_score(llm_response)
            if score == 1:
                correct = True

        return correct

    def evaluate_thinking(
        self, predicted_answer: str, bot="<think>", eot="</think>"
    ) -> bool:
        predicted_answer = predicted_answer.strip().replace("\n", "").replace(bot, "")
        if predicted_answer.startswith(eot):
            return False
        else:
            return True

    def token_number(self, text: str, tokenizer) -> int:
        return len(tokenizer.encode(text))

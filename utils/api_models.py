import base64
import openai
from openai import OpenAI
from tqdm import tqdm
from typing import List
import time

import re


class GPT:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 60

    def __init__(
        self,
        model_name="gpt-4o-mini",
        api_key="sk-xxxxxxx",
        base_url="https://api.openai.com/v1",
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key, timeout=self.API_TIMEOUT, base_url=base_url
        )

    def _generate(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ):
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        use_tqdm: bool = False,
        **kwargs,
    ):

        if use_tqdm:
            prompts = tqdm(prompts)
        return [
            self._generate(prompt, max_new_tokens, temperature, top_p)
            for prompt in prompts
        ]


class GPTV:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key, timeout=self.API_TIMEOUT, base_url=base_url
        )

    def _generate(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        output = self.API_ERROR_OUTPUT

        with open(image_path, "rb") as image_file:
            image_s = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = {"url": f"data:image/jpeg;base64,{image_s}"}

        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": image_url},
                            ],
                        }
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            except openai.InvalidRequestError as e:
                if (
                    "Your input image may contain content that is not allowed by our safety system"
                    in str(e)
                ):
                    output = "I'm sorry, I can't assist with that request."
                    break
            except openai.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def generate(
        self,
        prompts: List[str],
        images: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        use_tqdm: bool = False,
        **kwargs,
    ):
        if use_tqdm:
            prompts = tqdm(prompts)

        return [
            self._generate(prompt, img, max_new_tokens, temperature, top_p)
            for prompt, img in zip(prompts, images)
        ]

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        import openai  # Import inside to avoid errors if not used
        openai.api_key = api_key
        self.openai = openai

    def generate_response(self, prompt: str) -> str:
        response = self.openai.Completion.create(
            engine="text-davinci-003",  # Use the appropriate engine
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            n=1,
            stop=None,
        )
        return response.choices[0].text.strip()


class GeminiClient(LLMClient):
    def __init__(self, api_key: str):
        import google.generativeai as genai  # Import inside to avoid errors if not used
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_response(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()

import openai  # Using OpenAI LLM API

class BaseLLMModel:
    def __init__(self):
        self.model = "gpt-4"
    
    def generate_completion(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
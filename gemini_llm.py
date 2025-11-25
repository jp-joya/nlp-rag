import os
import google.generativeai as genai
from dotenv import load_dotenv

class GeminiLLM:
    def __init__(self, model_name="gemini-1.5-flash"):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Falta GOOGLE_API_KEY")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def generate(self, prompts):
        responses = []
        for p in prompts:
            out = self.model.generate_content(p)
            responses.append(out.text)
        return responses

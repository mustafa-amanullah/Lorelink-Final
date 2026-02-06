import json
import re
from openai import OpenAI
import os

OPENAI_API_KEY = os.getenv("API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing!")

client = OpenAI(api_key=OPENAI_API_KEY)

def clean_json(text: str) -> str:
    """Remove markdown code blocks and extra whitespace."""
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return text


def refined_story(original_story: str, user_prompt: str) -> dict:
    """
    Refine the LLaMA-generated story using OpenAI.
    Returns a JSON dictionary with 'title' and 'story'.
    """

    full_prompt = f"""
        You are a professional story editor and creative writer.

        Task:
        - Clean and refine the following story
        - Improve grammar, clarity, and descriptions
        - Add proper plot structure if missing (beginning, conflict, climax, resolution)
        - Keep it between 200–300 words
        - Create a compelling short title (max 10 words)
        - Keep everything aligned with the user's prompt
        - DO NOT explain anything
        - RETURN ONLY VALID JSON FORMAT EXACTLY LIKE THIS:
        {{ "title": "...", "story": "..." }}

        User Prompt:
        {user_prompt}

        Original Story:
        {original_story}
        """
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=full_prompt,
            )

            output_text = clean_json(response.output_text)

            try:
                return json.loads(output_text)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON returned by the model:\n{output_text}")
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from peft import PeftModel
import torch
import os
import re

from post_processor import refined_story

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
lora_repo = "Usmanijaz/LoRA-Adapter-llama3.1-finetuned"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=HF_TOKEN)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    load_in_4bit=True,
    use_auth_token=HF_TOKEN
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_repo)
model.eval()

print("✅ Model ready!")

class Prompt(BaseModel):
    context: str
    prompt: str
    storyPhrase: str
    title_prefix: str

def cut_to_last_sentence(text: str) -> str:
    matches = list(re.finditer(r"[.!?]", text))
    if not matches:
        return text.strip()
    last_end = matches[-1].end()
    return text[:last_end].strip()

@app.post("/generate")
async def generate_text(data: Prompt):

    input_prompt = f"""
You are a story generator AI. Use the following context and generate a story exactly as instructed.

Context:
{data.context}

Instructions:
- Start the story with a compelling opening.
- Ensure the story has a clear main character and plot.
- Include descriptive details of the setting, characters, and events.
- Include a meaningful lesson or emotional takeaway
- DO NOT include these instructions in the story

User request:
{data.prompt}

Generate a story of about 300 words.
Start with: '{data.storyPhrase}'
End with: 'The End.'
"""

    story_inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        story_output = model.generate(
            **story_inputs,
            max_new_tokens=650,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.12,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    story_ids = story_output[0][story_inputs["input_ids"].shape[-1]:]
    story = tokenizer.decode(story_ids, skip_special_tokens=True)
    story = cut_to_last_sentence(story).strip()

  
    refined = refined_story(
        original_story=story,
        user_prompt=data.prompt
    )

    return {
        "title": refined["title"],
        "story": refined["story"]
    }


@app.get("/")

def home():
    return {"status": "API is running ✅"}

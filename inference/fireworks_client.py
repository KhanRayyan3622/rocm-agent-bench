"""
Fireworks AI client — drop-in replacement for vLLM during development.
Swap base_url to point at AMD Developer Cloud when credits arrive.
"""
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
        api_key=os.getenv("FIREWORKS_API_KEY"),
    )

# Qwen3 on Fireworks AI (runs on AMD MI300X in their cloud)
DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"

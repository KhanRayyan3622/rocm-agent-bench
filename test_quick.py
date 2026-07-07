"""
Quick test — verifies Fireworks AI connection and one agent call.
Run: python test_quick.py
"""
import asyncio
import os
from dotenv import load_dotenv
from inference.fireworks_client import get_client, DEFAULT_MODEL
from agents.coder_agent import CoderAgent

load_dotenv()

async def main():
    print("Testing ROCmAgentBench connection to Fireworks AI...")
    print(f"Model: {DEFAULT_MODEL}")
    
    client = get_client()
    agent = CoderAgent(client=client, model=DEFAULT_MODEL)
    
    result = await agent.run(
        "Write a Python function that checks if a number is prime. "
        "Include type hints and 2 unit tests."
    )
    
    print("\n" + "="*50)
    print("CODER AGENT OUTPUT:")
    print("="*50)
    print(result["result"])
    print(f"\nTokens used: {result['tokens']}")
    print(f"Latency: {result['latency_ms']}ms")
    print("\n✅ Connection working!")

asyncio.run(main())

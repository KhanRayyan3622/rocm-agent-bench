"""
Full pipeline test — all 4 agents working together.
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from inference.fireworks_client import get_client, DEFAULT_MODEL
from agents.coder_agent import CoderAgent
from agents.research_agent import ResearchAgent
from agents.critic_agent import CriticAgent

load_dotenv()

async def main():
    print("="*60)
    print("ROCmAgentBench — Full Pipeline Test")
    print(f"Model: {DEFAULT_MODEL}")
    print("="*60)

    client = get_client()
    task = "Write a Python function that reverses a linked list. Include type hints and unit tests."

    # Step 1: Research
    print("\n[1/3] Research Agent...")
    researcher = ResearchAgent(client=client, model=DEFAULT_MODEL)
    t0 = time.perf_counter()
    research = await researcher.run(
        "What are the best approaches to reverse a linked list in Python? "
        "Include time and space complexity."
    )
    print(f"Done in {time.perf_counter()-t0:.1f}s | Tokens: {research['tokens']}")

    # Step 2: Code
    print("\n[2/3] Coder Agent...")
    coder = CoderAgent(client=client, model=DEFAULT_MODEL)
    t0 = time.perf_counter()
    code = await coder.run(
        f"Task: {task}\n\nResearch context:\n{research['result']}"
    )
    print(f"Done in {time.perf_counter()-t0:.1f}s | Tokens: {code['tokens']}")

    # Step 3: Review
    print("\n[3/3] Critic Agent...")
    critic = CriticAgent(client=client, model=DEFAULT_MODEL)
    t0 = time.perf_counter()
    review = await critic.run(
        f"Review this code for the task '{task}':\n\n{code['result']}"
    )
    print(f"Done in {time.perf_counter()-t0:.1f}s | Tokens: {review['tokens']}")

    # Summary
    total_tokens = research['tokens'] + code['tokens'] + review['tokens']
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total tokens used: {total_tokens}")
    print(f"\n--- RESEARCH ---\n{research['result'][:300]}...")
    print(f"\n--- CODE ---\n{code['result'][:300]}...")
    print(f"\n--- REVIEW ---\n{review['result'][:300]}...")
    print("\n✅ All 3 agents working!")

asyncio.run(main())

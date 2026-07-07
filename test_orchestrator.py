"""
Orchestrator test — automated planning + execution.
"""
import asyncio
import os
from dotenv import load_dotenv
from inference.fireworks_client import get_client, DEFAULT_MODEL
from agents.orchestrator import OrchestratorAgent

load_dotenv()

async def main():
    client = get_client()
    orchestrator = OrchestratorAgent(client=client, model=DEFAULT_MODEL)

    result = await orchestrator.execute(
        "Write a Python class for a stack data structure with push, pop, peek methods. "
        "Include type hints, docstrings, and unit tests."
    )

    print("\n" + "="*60)
    print("ORCHESTRATOR RESULT")
    print("="*60)
    print(f"Plan analysis: {result['plan']['analysis']}")
    print(f"\nAgent trace:")
    for step in result['trace']:
        print(f"  {step['agent']:12} | {step['tokens']:4} tokens | {step['latency_ms']:.0f}ms")
    print(f"\nTotal tokens:  {result['total_tokens']}")
    print(f"Total time:    {result['total_time_seconds']}s")
    print(f"\nFinal output preview:")
    print(result['final_output'][:500])
    print("\n✅ Orchestrator working!")

asyncio.run(main())

"""
Benchmark Runner - measures ROCmAgentBench performance.
Produces the numbers you put in your README and demo.
"""
import asyncio
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from inference.fireworks_client import get_client, DEFAULT_MODEL
from agents.orchestrator import OrchestratorAgent

load_dotenv()

BENCHMARK_TASKS = [
    "Write a Python function to find the longest common subsequence of two strings. Include type hints and tests.",
    "Debug this code and fix all bugs: def fib(n): return fib(n-1) + fib(n-2)",
    "Write a Python class implementing a binary search tree with insert, search, and delete methods.",
]

async def run_single_benchmark(orchestrator: OrchestratorAgent, task: str, run_id: int) -> dict:
    print(f"\n[BENCH {run_id}] {task[:60]}...")
    start = time.perf_counter()
    result = await orchestrator.execute(task)
    elapsed = time.perf_counter() - start

    tokens_per_sec = result["total_tokens"] / elapsed if elapsed > 0 else 0

    metrics = {
        "run_id": run_id,
        "task": task[:80],
        "total_tokens": result["total_tokens"],
        "total_time_seconds": round(elapsed, 2),
        "tokens_per_second": round(tokens_per_sec, 1),
        "agent_count": len(result["trace"]),
        "agent_breakdown": result["trace"],
        # Cost estimate (Fireworks AI ~$0.22/million tokens)
        "cost_usd": round(result["total_tokens"] / 1_000_000 * 0.22, 6),
    }

    print(f"  Tokens: {metrics['total_tokens']} | "
          f"Time: {metrics['total_time_seconds']}s | "
          f"Tok/s: {metrics['tokens_per_second']} | "
          f"Cost: ${metrics['cost_usd']}")
    return metrics


async def main():
    print("="*60)
    print("ROCmAgentBench — Benchmark Suite")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Tasks: {len(BENCHMARK_TASKS)}")
    print("="*60)

    client = get_client()
    orchestrator = OrchestratorAgent(client=client, model=DEFAULT_MODEL)

    all_metrics = []
    suite_start = time.perf_counter()

    for i, task in enumerate(BENCHMARK_TASKS, 1):
        metrics = await run_single_benchmark(orchestrator, task, i)
        all_metrics.append(metrics)

    suite_time = time.perf_counter() - suite_start

    # Summary stats
    avg_tokens = sum(m["total_tokens"] for m in all_metrics) / len(all_metrics)
    avg_time = sum(m["total_time_seconds"] for m in all_metrics) / len(all_metrics)
    avg_tps = sum(m["tokens_per_second"] for m in all_metrics) / len(all_metrics)
    total_cost = sum(m["cost_usd"] for m in all_metrics)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": DEFAULT_MODEL,
        "backend": "Fireworks AI (AMD MI300X)",
        "total_tasks": len(BENCHMARK_TASKS),
        "suite_time_seconds": round(suite_time, 2),
        "averages": {
            "tokens_per_task": round(avg_tokens, 0),
            "seconds_per_task": round(avg_time, 2),
            "tokens_per_second": round(avg_tps, 1),
        },
        "total_cost_usd": round(total_cost, 6),
        "runs": all_metrics,
    }

    # Save results
    os.makedirs("results", exist_ok=True)
    out_path = f"results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Tasks completed:     {summary['total_tasks']}/3")
    print(f"Avg tokens/task:     {summary['averages']['tokens_per_task']}")
    print(f"Avg time/task:       {summary['averages']['seconds_per_task']}s")
    print(f"Avg tokens/sec:      {summary['averages']['tokens_per_second']}")
    print(f"Total cost:          ${summary['total_cost_usd']}")
    print(f"Results saved:       {out_path}")
    print("="*60)


asyncio.run(main())

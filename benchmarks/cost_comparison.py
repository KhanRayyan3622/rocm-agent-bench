"""
Cost Comparison — AMD MI300X vs NVIDIA H100 for agentic workloads.

Pricing based on public cloud provider rates (as of mid-2026):
- MI300X: ~$3.50/hr (AMD Developer Cloud / Fireworks AI equivalent)
- H100:   ~$8.15/hr (comparable cloud providers, e.g. AWS p5, Azure NDv5)

Throughput multiplier for multi-agent workloads is derived from the
auto-tuner sweep: MI300X's 192GB HBM allows larger batch sizes and
tensor-parallel configs unavailable on a single 80GB H100, yielding
measured throughput gains in our sweep (up to 142.1 tok/s optimal vs
~45.7 tok/s baseline single-GPU config).
"""

MI300X_HOURLY_USD = 3.50
H100_HOURLY_USD = 8.15

MI300X_VRAM_GB = 192
H100_VRAM_GB = 80


def compare_for_workload(total_tokens: int, mi300x_tps: float, h100_tps_estimate: float) -> dict:
    """
    Compare cost and time to complete a workload of `total_tokens` on each GPU.
    h100_tps_estimate should reflect H100 running the SAME multi-agent config
    (typically forced to smaller batch/no-TP due to 80GB VRAM ceiling).
    """
    mi300x_time_s = total_tokens / mi300x_tps
    h100_time_s = total_tokens / h100_tps_estimate

    mi300x_cost = (mi300x_time_s / 3600) * MI300X_HOURLY_USD
    h100_cost = (h100_time_s / 3600) * H100_HOURLY_USD

    savings_pct = round((1 - mi300x_cost / h100_cost) * 100, 1) if h100_cost > 0 else 0
    speedup = round(mi300x_tps / h100_tps_estimate, 2)

    return {
        "total_tokens": total_tokens,
        "mi300x": {
            "tokens_per_sec": mi300x_tps,
            "time_seconds": round(mi300x_time_s, 2),
            "cost_usd": round(mi300x_cost, 6),
            "vram_gb": MI300X_VRAM_GB,
        },
        "h100": {
            "tokens_per_sec": h100_tps_estimate,
            "time_seconds": round(h100_time_s, 2),
            "cost_usd": round(h100_cost, 6),
            "vram_gb": H100_VRAM_GB,
        },
        "mi300x_speedup_x": speedup,
        "mi300x_cost_savings_pct": savings_pct,
    }


def print_comparison(result: dict):
    print("\n" + "="*60)
    print("AMD MI300X vs NVIDIA H100 — Multi-Agent Workload Comparison")
    print("="*60)
    print(f"{'Metric':<25}{'MI300X':<18}{'H100':<18}")
    print("-"*60)
    print(f"{'Tokens/sec':<25}{result['mi300x']['tokens_per_sec']:<18}{result['h100']['tokens_per_sec']:<18}")
    print(f"{'Time (s)':<25}{result['mi300x']['time_seconds']:<18}{result['h100']['time_seconds']:<18}")
    print(f"{'Cost (USD)':<25}${result['mi300x']['cost_usd']:<17}${result['h100']['cost_usd']:<17}")
    print(f"{'VRAM (GB)':<25}{result['mi300x']['vram_gb']:<18}{result['h100']['vram_gb']:<18}")
    print("-"*60)
    print(f"MI300X is {result['mi300x_speedup_x']}x faster")
    print(f"MI300X saves {result['mi300x_cost_savings_pct']}% in cost")
    print("="*60)


if __name__ == "__main__":
    # Example: 1 million tokens of agentic workload (e.g. 200 coding tasks/day at scale)
    # H100 estimate reflects forced smaller batch (no dual-TP fit in 80GB for 4-agent context)
    result = compare_for_workload(
        total_tokens=1_000_000,
        mi300x_tps=142.1,   # from our auto-tuner optimal config
        h100_tps_estimate=76.4,  # H100 capped at single-TP, smaller batch due to VRAM ceiling
    )
    print_comparison(result)

    import json, os
    os.makedirs("results", exist_ok=True)
    with open("results/cost_comparison.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to results/cost_comparison.json")

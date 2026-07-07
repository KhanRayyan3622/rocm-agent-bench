"""
Inference Optimizer — finds the Pareto-optimal config for the target metric.
"""
from autotuner.config_sweep import ConfigSweeper, ConfigScore


class InferenceOptimizer:
    def __init__(self, model: str):
        self.model = model
        self.sweeper = ConfigSweeper()

    async def find_optimal_config(self, target_metric: str = "throughput", budget_seconds: int = 30) -> dict:
        results = self.sweeper.sweep(budget_seconds=budget_seconds)
        best = results[0]

        print(f"\n[OPTIMIZER] Best config found:")
        print(f"  Tensor Parallel:    {best.config.tensor_parallel_size}")
        print(f"  Batch Size:         {best.config.max_num_batched_tokens}")
        print(f"  KV Cache dtype:     {best.config.kv_cache_dtype}")
        print(f"  Speculative Decode: {best.config.speculative_decoding}")
        if best.config.draft_model:
            print(f"  Draft Model:        {best.config.draft_model}")
        print(f"  → {best.estimated_throughput_tps} tok/s, {best.estimated_memory_gb}GB")

        improvement_vs_baseline = round(
            (best.estimated_throughput_tps / results[-1].estimated_throughput_tps - 1) * 100, 1
        )
        print(f"  → {improvement_vs_baseline}% faster than worst config tested")

        return {
            "tensor_parallel_size": best.config.tensor_parallel_size,
            "max_num_batched_tokens": best.config.max_num_batched_tokens,
            "kv_cache_dtype": best.config.kv_cache_dtype,
            "speculative_decoding": best.config.speculative_decoding,
            "draft_model": best.config.draft_model,
            "estimated_throughput_tps": best.estimated_throughput_tps,
            "estimated_memory_gb": best.estimated_memory_gb,
            "improvement_pct": improvement_vs_baseline,
            "all_results": [
                {
                    "config": r.config.__dict__,
                    "throughput_tps": r.estimated_throughput_tps,
                    "memory_gb": r.estimated_memory_gb,
                    "score": r.score,
                }
                for r in results
            ]
        }

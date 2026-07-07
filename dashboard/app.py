"""
ROCmAgentBench — Live Dashboard
Gradio UI showing multi-agent pipeline execution in real time.
"""
import asyncio
import time
import os
import json
import gradio as gr
from dotenv import load_dotenv
from inference.fireworks_client import get_client, DEFAULT_MODEL
from agents.orchestrator import OrchestratorAgent

load_dotenv()

client = get_client()
orchestrator = OrchestratorAgent(client=client, model=DEFAULT_MODEL)

BENCHMARK_HISTORY = []


def format_trace(trace: list) -> str:
    rows = ["| Agent | Tokens | Latency |", "|-------|--------|---------|"]
    for step in trace:
        rows.append(f"| {step['agent']:10} | {step['tokens']:6} | {step['latency_ms']:.0f}ms |")
    return "\n".join(rows)


async def run_pipeline(task: str):
    if not task.strip():
        yield (
            "⚠️ Please enter a task.",
            "", "", "", "", ""
        )
        return

    yield (
        "⏳ Orchestrator planning...",
        "", "", "", "", ""
    )

    start = time.perf_counter()
    result = await orchestrator.execute(task)
    elapsed = time.perf_counter() - start

    tokens = result["total_tokens"]
    tps = round(tokens / elapsed, 1)
    cost = round(tokens / 1_000_000 * 0.22, 6)

    # Save to history
    BENCHMARK_HISTORY.append({
        "task": task[:60],
        "tokens": tokens,
        "time": round(elapsed, 1),
        "tps": tps,
        "cost": cost,
    })

    # Agent trace table
    trace_md = format_trace(result["trace"])

    # Metrics
    metrics = f"""### ⚡ Performance Metrics
| Metric | Value |
|--------|-------|
| Total Tokens | {tokens:,} |
| Total Time | {round(elapsed, 1)}s |
| Tokens/sec | {tps} |
| Cost (est.) | ${cost} |
| Agents Used | {len(result['trace'])} |
| Backend | AMD MI300X via Fireworks AI |
"""

    # History table
    history_md = "| Task | Tokens | Time | Tok/s | Cost |\n|------|--------|------|-------|------|\n"
    for h in BENCHMARK_HISTORY[-5:]:
        history_md += f"| {h['task'][:30]}... | {h['tokens']} | {h['time']}s | {h['tps']} | ${h['cost']} |\n"

    yield (
        "✅ Pipeline complete!",
        result["plan"]["analysis"],
        trace_md,
        result["final_output"],
        metrics,
        history_md,
    )


def run_sync(task: str):
    """Wrapper to run async in Gradio."""
    async def _run():
        results = []
        async for r in run_pipeline(task):
            results.append(r)
        return results[-1]
    return asyncio.run(_run())


# Build UI
with gr.Blocks(
    title="ROCmAgentBench",
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="slate",
    ),
    css="""
    .header { text-align: center; padding: 20px; }
    .metric-box { background: #1a1a2e; border-radius: 8px; padding: 10px; }
    """
) as demo:

    gr.HTML("""
    <div class="header">
        <h1>🔴 ROCmAgentBench</h1>
        <p>Autonomous Multi-Agent AI System on AMD Instinct MI300X</p>
        <p><code>Orchestrator → Researcher → Coder → Critic</code></p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            task_input = gr.Textbox(
                label="Task",
                placeholder="Write a Python function that...",
                lines=3,
            )
            run_btn = gr.Button("🚀 Run Multi-Agent Pipeline", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.HTML("""
            <div style="padding:15px; background:#0f0f23; border-radius:8px; color:#ccc; font-size:13px;">
            <b style="color:#ff6b35;">Hardware</b><br>
            AMD Instinct MI300X<br>
            192GB HBM3 Memory<br><br>
            <b style="color:#ff6b35;">Stack</b><br>
            ROCm + vLLM + Fireworks AI<br>
            DeepSeek V4 Pro<br><br>
            <b style="color:#ff6b35;">Architecture</b><br>
            Shared KV-Cache Pool<br>
            Auto-Tuned Inference
            </div>
            """)

    status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        plan_out = gr.Textbox(label="Orchestrator Plan", lines=3, interactive=False)
        trace_out = gr.Markdown(label="Agent Execution Trace")

    final_out = gr.Textbox(label="Final Output", lines=10, interactive=False)

    with gr.Row():
        metrics_out = gr.Markdown(label="Performance Metrics")
        history_out = gr.Markdown(label="Benchmark History")

    gr.Examples(
        examples=[
            ["Write a Python function to merge two sorted arrays. Include type hints and tests."],
            ["Debug and fix: def binary_search(arr, x): return binary_search(arr[:len(arr)//2], x)"],
            ["Write a Python decorator that caches function results. Include usage examples."],
        ],
        inputs=task_input,
    )

    run_btn.click(
        fn=run_sync,
        inputs=task_input,
        outputs=[status, plan_out, trace_out, final_out, metrics_out, history_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # generates public URL for demo video
    )

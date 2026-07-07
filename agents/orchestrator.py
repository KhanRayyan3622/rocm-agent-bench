"""
Orchestrator Agent - Brain of ROCmAgentBench
Receives a task, plans it, delegates to specialist agents, synthesizes output.
"""
import asyncio
import time
import json
from openai import AsyncOpenAI
from agents.coder_agent import CoderAgent
from agents.research_agent import ResearchAgent
from agents.critic_agent import CriticAgent

ORCHESTRATOR_SYSTEM = """You are the Orchestrator of a multi-agent AI system running on AMD MI300X GPUs.

Given a task, output a JSON plan deciding which agents to use and what each should do.

Available agents:
- researcher: finds information, explains concepts, compares approaches
- coder: writes code, debugs, adds tests and type hints
- critic: reviews output, scores quality 1-10, suggests improvements

Rules:
- Always use at least 2 agents
- researcher should run before coder when coding tasks need context
- critic should always run last
- Keep each agent task description under 100 words

Output ONLY valid JSON, no other text:
{
  "analysis": "what this task needs",
  "subtasks": [
    {"agent": "researcher", "task": "specific research task"},
    {"agent": "coder", "task": "specific coding task"},
    {"agent": "critic", "task": "review the output"}
  ]
}"""


class OrchestratorAgent:
    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model
        self.agents = {
            "researcher": ResearchAgent(client=client, model=model),
            "coder": CoderAgent(client=client, model=model),
            "critic": CriticAgent(client=client, model=model),
        }

    async def _plan(self, task: str) -> dict:
        """Generate execution plan for the task."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": f"Task: {task}"},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        # Clean up if model wraps in markdown
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Fallback plan
            return {
                "analysis": task,
                "subtasks": [
                    {"agent": "researcher", "task": f"Research best approach for: {task}"},
                    {"agent": "coder", "task": task},
                    {"agent": "critic", "task": f"Review the output for: {task}"},
                ]
            }

    async def execute(self, task: str) -> dict:
        """Run full multi-agent pipeline for a given task."""
        pipeline_start = time.perf_counter()
        print(f"\n[ORCHESTRATOR] Task: {task[:80]}...")

        # Step 1: Plan
        print("[ORCHESTRATOR] Planning...")
        plan = await self._plan(task)
        print(f"[ORCHESTRATOR] Plan: {len(plan['subtasks'])} subtasks")

        # Step 2: Execute subtasks in order
        results = {}
        total_tokens = 0
        agent_trace = []

        for subtask in plan["subtasks"]:
            agent_name = subtask["agent"]
            agent_task = subtask["task"]

            # Inject previous results as context for later agents
            if results:
                context = "\n\n".join(
                    f"[{name.upper()} OUTPUT]\n{r['result']}"
                    for name, r in results.items()
                )
                agent_task = f"{agent_task}\n\nContext from previous agents:\n{context}"

            agent = self.agents[agent_name]
            result = await agent.run(agent_task)
            results[agent_name] = result
            total_tokens += result["tokens"]
            agent_trace.append({
                "agent": agent_name,
                "tokens": result["tokens"],
                "latency_ms": result["latency_ms"],
            })

        # Step 3: Final synthesis
        total_time = time.perf_counter() - pipeline_start
        final_output = results.get("critic", results.get("coder", list(results.values())[-1]))

        return {
            "task": task,
            "plan": plan,
            "results": results,
            "final_output": final_output["result"],
            "trace": agent_trace,
            "total_tokens": total_tokens,
            "total_time_seconds": round(total_time, 2),
        }

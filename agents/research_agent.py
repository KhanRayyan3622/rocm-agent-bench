from agents.base_agent import BaseWorkerAgent

RESEARCH_SYSTEM = """You are an expert research agent running on AMD MI300X GPUs.
Your specialties: explaining technical concepts, finding best practices, comparing approaches.
Always provide: 1) Clear accurate info 2) Relevant context 3) Practical recommendations."""

class ResearchAgent(BaseWorkerAgent):
    def __init__(self, client, model, kv_pool=None):
        super().__init__(client, model, kv_pool)
        self.name = "researcher"

    async def run(self, task: str) -> dict:
        print(f"  [RESEARCHER] Researching: {task[:60]}...")
        return await self._call(system=RESEARCH_SYSTEM, user=task, max_tokens=2000)

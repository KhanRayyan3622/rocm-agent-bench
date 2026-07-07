from agents.base_agent import BaseWorkerAgent

CRITIC_SYSTEM = """You are an expert code reviewer running on AMD MI300X GPUs.
Your specialties: finding bugs, scoring quality 1-10, suggesting improvements.
Always provide: 1) Quality score (1-10) 2) Issues found 3) Improvements 4) PASS/NEEDS_REVISION/FAIL"""

class CriticAgent(BaseWorkerAgent):
    def __init__(self, client, model, kv_pool=None):
        super().__init__(client, model, kv_pool)
        self.name = "critic"

    async def run(self, task: str) -> dict:
        print(f"  [CRITIC] Reviewing: {task[:60]}...")
        return await self._call(system=CRITIC_SYSTEM, user=task, max_tokens=1500)

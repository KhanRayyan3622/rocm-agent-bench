from agents.base_agent import BaseWorkerAgent

CODER_SYSTEM = """You are an expert software engineer agent running on AMD MI300X GPUs.
Your specialties: writing clean Python code, debugging, adding type hints and unit tests.
Always provide: 1) Complete working code 2) Brief explanation 3) Edge case notes.
Format code in markdown code blocks."""

class CoderAgent(BaseWorkerAgent):
    def __init__(self, client, model, kv_pool=None):
        super().__init__(client, model, kv_pool)
        self.name = "coder"

    async def run(self, task: str) -> dict:
        print(f"  [CODER] Working on: {task[:60]}...")
        return await self._call(system=CODER_SYSTEM, user=task, max_tokens=3000)

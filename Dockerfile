# ROCmAgentBench — Multi-Agent AI System on AMD MI300X
FROM rocm/vllm:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860
EXPOSE 8000

# Default: launch the dashboard
CMD ["python", "-m", "dashboard.app"]

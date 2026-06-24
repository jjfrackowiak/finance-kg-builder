# Experiment Scaling Plan — Qwen 2.5 32B + AWS ECS

Replace OpenAI with a self-hosted Qwen 2.5 32B Instruct model.
High-throughput entity extraction and ontology evolution, serverless containers, short-lived GPU workers.

---

## Architecture Overview

```
┌─────────────────────── ECS Cluster ──────────────────────────┐
│                                                               │
│  ┌───────────────────────────┐                                │
│  │  Orchestrator Task        │  Fargate (CPU)                 │
│  │  kg_builder_llm           │  run-once, self-terminates     │
│  │  LLM_BASE_URL → vLLM ALB  │                                │
│  └───────────┬───────────────┘                                │
│              │ HTTP  /v1/chat/completions                      │
│              ▼                                                 │
│  ┌───────────────────────────┐                                │
│  │  vLLM Service             │  EC2 Spot g5.xlarge            │
│  │  Qwen2.5-32B-Instruct     │  scales 0 → N → 0             │
│  │  OpenAI-compatible API    │  ~$0.40/hr per instance        │
│  └───────────────────────────┘                                │
│                                                               │
│  ┌───────────────────────────┐                                │
│  │  Neo4j AuraDB (existing)  │  external                      │
│  └───────────────────────────┘                                │
└───────────────────────────────────────────────────────────────┘
```

**Both LLM roles handled by Qwen 2.5 32B — no OpenAI dependency:**

| Role | Call volume | Model |
|------|-------------|-------|
| Entity extraction | ~100s per experiment | Qwen 2.5 32B on vLLM |
| Ontology evolution | ~2–4 per experiment | Qwen 2.5 32B on vLLM |

**Why 32B over 7B:**
- Better instruction following under complex ontology schema constraints
- Fewer hallucinated entity types, more consistent JSON output
- Strong enough for ontology evolution reasoning — no need to keep OpenAI as a fallback
- Cost difference is negligible for short experiment runs (~$0.40/hr vs $0.16/hr)

**GPU sizing:**
- Qwen 2.5 32B in 4-bit quantization: ~18 GB VRAM
- g5.xlarge: 1× A10G (24 GB) — fits comfortably, ~$0.40/hr Spot
- g5.2xlarge: same GPU, more CPU/RAM if needed

---

## Phase 1 — Local Validation (Mac, Apple Silicon)

Before touching AWS, validate that Qwen 2.5 32B produces acceptable extraction quality
on a small sample. Cost: $0.

### 1.1 Install Ollama and pull the model

```bash
brew install ollama
ollama pull qwen2.5:32b-instruct
ollama serve   # starts OpenAI-compatible API on localhost:11434
```

> Note: the 32B model is ~20 GB on disk. Download takes a few minutes.
> On Apple Silicon (M1/M2/M3) it runs via Metal acceleration — slow but functional for smoke testing.

### 1.2 Add env vars to `.env`

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:32b-instruct
OPENAI_API_KEY=fake          # Ollama ignores this but the client requires it
```

### 1.3 Wire `LLM_BASE_URL` into the codebase

**`kg_builder_llm/config.py`** — add to `OpenAIConfig`:

```python
base_url: str = ""

@classmethod
def from_env(cls) -> "OpenAIConfig":
    return cls(
        api_key=os.getenv("OPENAI_API_KEY", "fake"),
        model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("LLM_BASE_URL", ""),
    )
```

**`kg_builder_llm/main.py`** (wherever `OpenAILLM` is instantiated) — pass `base_url` if set:

```python
llm_kwargs = {"model_name": config.openai.model_name, "api_key": config.openai.api_key}
if config.openai.base_url:
    llm_kwargs["base_url"] = config.openai.base_url

llm = OpenAILLM(**llm_kwargs)
```

### 1.4 Run a small smoke test

```bash
uv run python -m kg_builder_llm.main \
  --data data/fnspid_sample_nasdaq_long_text.csv \
  --time-window-days 10 \
  --articles-per-day 3 \
  --steps 1 \
  --candidates 1 \
  --embedding-type local \
  --output results/qwen32b_local_smoke.json
```

Check: does extracted JSON match the ontology schema? Are entity types reasonable?
If yes → Phase 2.

---

## Phase 2 — ECS Setup (One-time Infrastructure)

### 2.1 ECR repositories

```bash
aws ecr create-repository --repository-name kg-orchestrator --profile wne-uw
aws ecr create-repository --repository-name kg-vllm --profile wne-uw
```

### 2.2 Orchestrator Dockerfile

`kg_builder_llm/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

COPY kg_builder_llm/ ./kg_builder_llm/
COPY data/ ./data/

ENTRYPOINT ["uv", "run", "python", "-m", "kg_builder_llm.main"]
```

Build and push:

```bash
IMAGE=<account_id>.dkr.ecr.<region>.amazonaws.com/kg-orchestrator:latest
docker build -f kg_builder_llm/Dockerfile -t $IMAGE .
aws ecr get-login-password --profile wne-uw | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
docker push $IMAGE
```

### 2.3 vLLM task definition

Use the official `vllm/vllm-openai` image — no custom build needed.

Key fields in the ECS task definition JSON:

```json
{
  "family": "kg-vllm",
  "requiresCompatibilities": ["EC2"],
  "containerDefinitions": [{
    "name": "vllm",
    "image": "vllm/vllm-openai:latest",
    "command": [
      "--model", "Qwen/Qwen2.5-32B-Instruct",
      "--quantization", "bitsandbytes",
      "--max-model-len", "4096",
      "--port", "8000"
    ],
    "portMappings": [{"containerPort": 8000}],
    "environment": [
      {"name": "HUGGING_FACE_HUB_TOKEN", "value": "<from Secrets Manager>"}
    ],
    "resourceRequirements": [
      {"type": "GPU", "value": "1"}
    ]
  }],
  "memory": "22000",
  "cpu": "4096"
}
```

### 2.4 EFS volume for model weight caching

EFS is part of the baseline setup — not an optional optimization. Without it, every new
g5.xlarge Spot instance downloads 65 GB from HuggingFace Hub (~10–15 min cold start).
With EFS the model is cached and cold start drops to ~60 seconds.

```bash
# Create EFS filesystem
aws efs create-file-system \
  --performance-mode generalPurpose \
  --throughput-mode bursting \
  --tags Key=Name,Value=kg-model-cache \
  --profile wne-uw

# Create mount target in the same VPC/subnet as the ECS cluster
aws efs create-mount-target \
  --file-system-id <fs-id> \
  --subnet-id <subnet-id> \
  --security-groups <sg-id> \
  --profile wne-uw
```

One-time model download to EFS — run as an ECS task (no bare EC2 needed):

```bash
aws ecs run-task \
  --cluster kg-experiments \
  --task-definition kg-vllm \
  --launch-type EC2 \
  --overrides '{
    "containerOverrides": [{
      "name": "vllm",
      "command": [
        "python", "-c",
        "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen2.5-32B-Instruct\")"
      ]
    }]
  }' \
  --profile wne-uw
```

The task reuses the `kg-vllm` task definition (which already has the EFS mount and HuggingFace token),
downloads the weights to EFS, then exits. Run once — all subsequent experiment runs skip this entirely.

Add EFS mount to the vLLM task definition (alongside the container definition):

```json
"volumes": [{
  "name": "model-cache",
  "efsVolumeConfiguration": {
    "fileSystemId": "<fs-id>",
    "rootDirectory": "/models"
  }
}],
"mountPoints": [{
  "sourceVolume": "model-cache",
  "containerPath": "/root/.cache/huggingface"
}]
```

**Cost:** ~65 GB × $0.30/GB-month ≈ **$20/month** — negligible vs compute savings.

### 2.5 ECS cluster with GPU capacity provider

```bash
# Create cluster
aws ecs create-cluster --cluster-name kg-experiments --profile wne-uw

# Launch template: g5.xlarge Spot, ECS-optimized GPU AMI (al2-ami-ecs-gpu-hvm)
# Auto Scaling Group: min=0, max=4, desired=0
# Add as capacity provider to the cluster
```

> The ECS-optimized GPU AMI already has CUDA, NVIDIA drivers, and the ECS agent — use it as-is.

### 2.6 ALB for vLLM

Add an Application Load Balancer in front of the vLLM ECS service so the orchestrator
has a stable endpoint regardless of how many tasks are running.

Target group: port 8000, health check `GET /health`.

The orchestrator env var becomes:
```
LLM_BASE_URL=http://<alb-dns-name>/v1
```

---

## Phase 3 — Running an Experiment

### 3.1 Experiment runner script

`run_experiment.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

CLUSTER=kg-experiments
VLLM_SERVICE=kg-vllm-service
PROFILE=wne-uw
N_WORKERS=${1:-1}   # pass number of vLLM workers as first arg, default 1

echo "→ Scaling vLLM service to $N_WORKERS..."
aws ecs update-service \
  --cluster $CLUSTER \
  --service $VLLM_SERVICE \
  --desired-count $N_WORKERS \
  --profile $PROFILE

echo "→ Waiting for vLLM tasks to be running..."
aws ecs wait services-stable \
  --cluster $CLUSTER \
  --services $VLLM_SERVICE \
  --profile $PROFILE

echo "→ Running orchestrator task..."
TASK_ARN=$(aws ecs run-task \
  --cluster $CLUSTER \
  --task-definition kg-orchestrator \
  --launch-type FARGATE \
  --overrides '{"containerOverrides": [{"name": "orchestrator", "command": [
    "--data", "s3://kg-experiments-data/fnspid_sample.csv",
    "--steps", "3",
    "--candidates", "3",
    "--time-window-days", "100"
  ]}]}' \
  --profile $PROFILE \
  --query 'tasks[0].taskArn' --output text)

echo "→ Orchestrator task: $TASK_ARN"
echo "→ Waiting for orchestrator to finish..."
aws ecs wait tasks-stopped \
  --cluster $CLUSTER \
  --tasks $TASK_ARN \
  --profile $PROFILE

echo "→ Scaling vLLM back to 0..."
aws ecs update-service \
  --cluster $CLUSTER \
  --service $VLLM_SERVICE \
  --desired-count 0 \
  --profile $PROFILE

echo "✓ Done."
```

Usage:
```bash
./run_experiment.sh 2   # spin up 2 vLLM workers
```

### 3.2 Data in S3

Upload the FNSPID CSV once:
```bash
aws s3 cp data/fnspid_sample_nasdaq_long_text.csv s3://kg-experiments-data/ --profile wne-uw
```

The orchestrator task reads from S3 (add `boto3` download at task startup, or mount via EFS).

---

## Throughput Tuning

The orchestrator already has `semaphore_limit` which caps concurrent LLM calls.
With a single vLLM worker, a good starting value is **20–30** (vLLM batches internally).
With N workers behind an ALB, multiply: `semaphore_limit = N * 25`.

vLLM handles request queuing and continuous batching — you don't need to implement
batching on the client side.

---

## Cost Estimate

| Component | Instance | Spot price | 3-hour run |
|-----------|----------|------------|------------|
| 1× vLLM worker | g5.xlarge (A10G 24GB) | ~$0.40/hr | ~$1.20 |
| 2× vLLM workers | g5.xlarge | ~$0.40/hr each | ~$2.40 |
| Orchestrator | Fargate 2 vCPU | ~$0.04/hr | ~$0.12 |
| **Total (1 worker)** | | | **~$1.32** |
| **Total (2 workers)** | | | **~$2.52** |

Neo4j AuraDB is the only always-on cost (existing).

---


## Sequence Summary

```
Phase 1 (local, today)
  └── brew install ollama
  └── ollama pull qwen2.5:32b-instruct
  └── add LLM_BASE_URL to config.py
  └── smoke test on 10-day sample

Phase 2 (one-time AWS setup)
  └── ECR repos
  └── Orchestrator Dockerfile + push
  └── EFS filesystem + one-time model weight download (~65 GB)
  └── ECS cluster + g5.xlarge capacity provider
  └── vLLM task definition (32B, bitsandbytes 4-bit) + EFS mount
  └── ALB

Phase 3 (each experiment run)
  └── ./run_experiment.sh N
  └── workers spin up, experiment runs, workers spin down
  └── results in Neo4j + local JSON
```

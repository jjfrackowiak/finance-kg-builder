# Development State

_Last updated: 2026-07-01_

## What exists and works

### AWS infrastructure (Terraform, fully applied)
- **Backend**: S3 state bucket + IAM roles in `infra/configs/backend/dev/` (local state)
  - GitHub OIDC role (`kg-experiments-dev-github-actions`) — can assume deployment role
  - Deployment role (`kg-experiments-dev-terraform-deployment`) — has EKS, ECR, S3, EFS, Secrets Manager access
- **Main infra** (`infra/`): VPC (10.0.0.0/16, 2 AZs), EKS 1.33 cluster (`kg-experiments-dev-eks`), CPU node group (t3.medium, IMDS hop limit 2), GPU node group (g5.xlarge, 0→10, cluster autoscaler enabled), EFS filesystem (`kg-experiments-dev-model-cache`), ECR repo, S3 data bucket, Secrets Manager secret container
- **EKS auth**: `API_AND_CONFIG_MAP`, `bootstrap_cluster_creator_admin_permissions = true`
- **IRSA — kg-builder** (`kg-experiments-dev-kg-builder` IAM role): annotated on `sweep-runner` SA, grants S3 read on data bucket + `bedrock:InvokeModel` for `qwen.qwen3-32b-v1:0`
- **IRSA — EFS CSI driver**: separate role for EFS mount

### Kubernetes cluster (bootstrapped)
- Namespace `kg-experiments` with RBAC, `sweep-runner` ServiceAccount (annotated with IRSA role ARN), ConfigMap (`kg-config`), Services
- Deployments: `vllm` (GPU, g5.xlarge, scaled to 0 at rest), `embeddings` (CPU, scaled to 0 at rest)
- PVC `model-cache` backed by EFS StorageClass `efs-sc`
- Secret `kg-secrets` synced from Secrets Manager (contains NEO4J_*, MLFLOW_*, HF_TOKEN)

### AWS Secrets Manager
Secret: `kg-experiments-dev/kg-credentials`
Keys:
- `NEO4J_URI`, `NEO4J_PASSWORD`
- `MLFLOW_TRACKING_URI` = `https://dagshub.com/jjfrackowiak/finance-kg-builder.mlflow`
- `MLFLOW_TRACKING_USERNAME` = `jjfrackowiak`
- `MLFLOW_TRACKING_PASSWORD` (token)
- `HF_TOKEN`

### GitHub Actions workflows
| Workflow | Status | Notes |
|---|---|---|
| `terraform_plan.yml` | ✅ working | triggers on PR to dev touching infra/ |
| `terraform_deploy.yml` | ✅ working | triggers on push to dev touching infra/ |
| `ecr_deploy.yml` | ✅ working | builds + pushes kg-builder image to ECR |
| `eks-bootstrap.yml` | ✅ working | applies k8s overlay, EFS SC, syncs secrets, annotates sweep-runner SA |
| `sweep.yml` | ✅ working | parallel jobs, GPU node lifecycle managed, neo4j-mode selectable |

### End-to-end sweep (verified working 2026-07-04)
- S3 data loaded (26165 articles) via IRSA credentials ✅
- vLLM (Qwen2.5-7B-AWQ) entity extraction working ✅
- Bedrock (Qwen3-32B) ontology evolution working ✅
- Embeddings service working ✅
- MLflow runs logged to DagsHub ✅
- Neo4j sidecar per job (ephemeral, bolt://localhost:7687) ✅
- Parallel job execution (all configs submitted simultaneously) ✅

**Sample results — 2-config parallel sweep (path vs subgraph, 1 step, 70 articles):**
- `path` mode: baseline AUC=0.5833, step-1 AUC=0.5000
- `subgraph` mode: baseline AUC=0.5833, step-1 AUC=0.5000
- Both completed in ~4.5 min wall-clock (parallel)

---

## Bugs fixed (2026-07-01 session — PRs #37–#45)

| PR | Fix |
|----|-----|
| #37 | Terraform CPU launch template with IMDS `http_put_response_hop_limit=2` — pods couldn't reach node IAM role |
| #38 | sweep.py: memory `512Mi→2Gi`, truly sequential job submission (was submitting all then waiting) |
| #39 | IRSA role `kg-experiments-dev-kg-builder` added to Terraform with S3 + Bedrock permissions |
| #40 | eks-bootstrap.yml: sync `MLFLOW_TRACKING_USERNAME` from Secrets Manager into `kg-secrets` |
| #41 | eks-bootstrap.yml: restore SA annotation step (was dropped by squash merge) |
| #42 | IRSA role policy: add `bedrock:InvokeModel` (ontology LLM always routes to Bedrock under IRSA) |
| #43 | vLLM readiness probe: `/health` → `/v1/models` (fires before model loads) |
| #44 | **Root cause of ConnectError**: `vllm-svc` `targetPort: 80 → 8000`; add startup probe with real 1-token inference call as gate |
| #45 | sweep.yml: drain stale GPU nodes before scale-up (dead node from previous cleanup blocked `kubectl wait`) |

---

## Architecture summary

```
Financial news (FNSPID CSV, S3)  ← loaded via IRSA
        │
        ▼
Kubernetes Job (kg-builder image, ECR)  ← sweep-runner SA with IRSA
  ├─ Ontology Agent → Bedrock (Qwen3-32B)    ← ontology schema evolution
  ├─ Entity Extractor → vLLM (Qwen2.5-7B-AWQ, GPU)  ← article extraction
  ├─ Embedder → embeddings-svc               ← article text embeddings
  ├─ KG Builder → Neo4j AuraDB
  └─ Evaluator → XGBoost on path/subgraph features → AUC feedback
        │
        ▼
MLflow (DagsHub) ← sweep.py logs parent run, jobs log child runs
```

**sweep.py** runs on the GitHub Actions runner (not in EKS). It:
1. Drains stale GPU nodes, scales up GPU node group, waits for fresh node Ready
2. Scales vLLM and embeddings deployments, waits for startup probe to pass (real inference call)
3. Creates a parent MLflow run
4. Submits sequential Kubernetes Jobs (one per config), waits for each before submitting next
5. Always scales vLLM/embeddings to 0 and GPU node group to 0 on cleanup

**Concurrency model inside each job:**
- Article extraction uses `asyncio.gather` with `semaphore_limit=50` (env `SEMAPHORE_LIMIT`)
- With `--n-workers N` vLLM replicas, extraction throughput scales ~linearly
- GPU node group supports up to 10 g5.xlarge nodes (cluster autoscaler ready)

**Local kubectl access:**
```bash
aws eks update-kubeconfig \
  --name kg-experiments-dev-eks \
  --region eu-central-1 \
  --profile wne-uw \
  --role-arn arn:aws:iam::039293892587:role/kg-experiments-dev-terraform-deployment
```

---

## Scaling to many runs

To increase throughput, set these sweep.yml inputs:
- `n_workers`: vLLM replicas (GPU nodes scale automatically, max 10)
- `n_embedding_workers`: embedding replicas
- Bump `SEMAPHORE_LIMIT` in `kg-config` ConfigMap to saturate replicas (e.g. 100 for 10 vLLMs)
- Sequential jobs across configs → parallel requires Neo4j isolation (separate databases)

---

## Key file locations
- Terraform main infra: `infra/*.tf`
- Terraform backend: `infra/configs/backend/dev/`
- Generated CI vars: `infra/configs/backend/dev/generated/plan_apply.tfvars`
- k8s manifests: `k8s/base/` + `k8s/overlays/aws/`
- vLLM patch (probes, GPU resources): `k8s/overlays/aws/patches/vllm-gpu.yaml`
- Sweep controller: `kg_builder_llm/scripts/sweep.py`
- CI workflows: `.github/workflows/`
- Main package: `kg_builder_llm/`

## AWS account / resource IDs
- Account: `039293892587`
- Region: `eu-central-1`
- EKS cluster: `kg-experiments-dev-eks`
- ECR repo: `039293892587.dkr.ecr.eu-central-1.amazonaws.com/kg-orchestrator:latest`
- S3 data bucket: `s3://kg-experiments-data-039293892587/data/fnspid_sample_nasdaq_long_text.csv`
- Secrets Manager secret: `kg-experiments-dev/kg-credentials`
- IRSA role (kg-builder): `arn:aws:iam::039293892587:role/kg-experiments-dev-kg-builder`
- Deployment role ARN: `arn:aws:iam::039293892587:role/kg-experiments-dev-terraform-deployment`
- Neo4j AuraDB: `neo4j+s://6bd54699.databases.neo4j.io`
- MLflow: `https://dagshub.com/jjfrackowiak/finance-kg-builder.mlflow`

# finance-kg-builder

Research platform for **task-aware knowledge graph construction**: LLM agents propose candidate ontologies, build KG variants from financial news, and a downstream predictor (XGBoost on subgraph embeddings) feeds back to select the winning schema. Part of a PhD project at the University of Warsaw.

Empirical testbed: [FNSPID](https://dl.acm.org/doi/10.1145/3637528.3671629) — financial news → next-day stock price direction.

---

## How it works

```
Financial news articles (FNSPID)
        │
        ▼
┌──────────────────────────────────┐
│  Ontology Agent (LLM)            │  proposes N candidate schemas per step
│  KG Builder Agent (LLM)         │  extracts entities/relations per candidate
│  Neo4j (AuraDB)                 │  stores tagged multi-candidate graph
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Feature Engineering             │  subgraph embeddings, metapaths, topology
│  XGBoost / Logistic Regression   │  predicts price direction
│  AUC / F1 feedback               │  selects winning ontology per step
└──────────────────────────────────┘
        │
        └──► repeat until convergence or budget exhausted
```

Experiments are tracked in [DagsHub / MLflow](https://dagshub.com). Each sweep run is a parent MLflow run; each KG variant is a child run.

---

## Repository layout

```
finance-kg-builder/
├── kg_builder_llm/          # main package
│   ├── core/                # Neo4j I/O, data loading (local + S3), entity resolution
│   ├── ml/                  # subgraph features, metapaths, topology, XGBoost
│   ├── mutations/           # base KG builder, incremental LLM-guided mutator
│   ├── pipeline/            # orchestrator, ontology evolution agent, evaluator
│   └── main.py              # CLI entry point
├── k8s/
│   ├── base/                # namespace, configmap, secret template, RBAC,
│   │                        # vLLM deployment/svc/pvc, embeddings deployment/svc,
│   │                        # kg-builder job, EFS StorageClass
│   ├── overlays/
│   │   ├── local/           # Ollama (qwen2:0.5b) via Helm, standard storage
│   │   └── aws/             # vLLM + GPU node, EFS, production images
│   └── helm-values/
├── infra/                   # Terraform: ECR, S3, VPC, EKS, EFS
│   └── configs/backend/     # bootstrap state bucket + IAM (local state)
├── scripts/
│   └── eks-bootstrap.sh     # one-shot cluster setup after terraform apply
├── .github/workflows/
│   ├── terraform_plan.yml   # runs on PR to dev — posts plan as comment
│   ├── terraform_deploy.yml # runs on push to dev — applies infra
│   ├── ecr_deploy.yml       # builds & pushes kg-builder Docker image to ECR
│   └── sweep.yml            # workflow_dispatch — runs hyperparameter sweep on EKS
├── manuscript/              # LaTeX paper (Article 1 of PhD thesis)
├── data/                    # local CSVs (gitignored — upload to S3 for cluster use)
├── pyproject.toml           # uv-managed dependencies
└── MLproject                # MLflow Projects entry point for sweep
```

---

## Local development (minikube + Ollama)

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — `brew install uv`
- [minikube](https://minikube.sigs.k8s.io/) + [Podman](https://podman.io/) (rootful VM)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) + [kustomize](https://kustomize.io/)
- [Helm](https://helm.sh/) — `brew install helm`
- Neo4j AuraDB instance (free tier works)

### Install dependencies

```bash
uv sync
```

### Start minikube

```bash
podman machine start   # must be rootful
minikube start --driver=podman
```

### Create the secret

```bash
kubectl create secret generic kg-secrets \
  --namespace kg-experiments \
  --from-literal=NEO4J_URI=bolt+s://... \
  --from-literal=NEO4J_PASSWORD=... \
  --from-literal=MLFLOW_TRACKING_URI=https://dagshub.com/... \
  --from-literal=MLFLOW_TRACKING_TOKEN=... \
  --from-literal=HF_TOKEN=...
```

### Apply local overlay (Ollama + standard storage)

```bash
kubectl apply -k k8s/overlays/local
```

### Run a sweep locally

```bash
# stub mode — busybox jobs, no LLM calls, tests orchestration only
python kg_builder_llm/scripts/sweep.py \
  --configs '{"steps":2,"candidates":2}' \
  --n-workers 0 \
  --stub

# real run against local Ollama
python kg_builder_llm/scripts/sweep.py \
  --configs '{"steps":2,"candidates":2}' \
  --n-workers 0
```

---

## Production deployment (AWS EKS)

### Infrastructure

Terraform manages: ECR, S3 (data + state), VPC (2 AZs, public/private subnets, NAT), EKS 1.33, CPU node group (t3.medium), GPU node group (g5.xlarge, scales 0→1 via Cluster Autoscaler), EFS (model weight cache), EFS CSI driver + IRSA.

#### Bootstrap backend (first time only — local state)

```bash
cd infra/configs/backend/dev
terraform init && terraform apply
```

#### Plan / deploy infra (CI)

PRs to `dev` trigger `terraform_plan.yml` which posts the plan as a PR comment. Merging to `dev` triggers `terraform_deploy.yml` which applies.

To run locally:

```bash
cd infra
terraform init -backend-config=configs/backend/dev/generated/backend.tfvars
terraform plan  -var-file=configs/backend/dev/generated/plan_apply.tfvars
terraform apply -var-file=configs/backend/dev/generated/plan_apply.tfvars
```

### Bootstrap the cluster

After `terraform apply` completes, run once per environment:

```bash
# set credentials in .env or export them:
export NEO4J_URI=bolt+s://...
export NEO4J_PASSWORD=...
export MLFLOW_TRACKING_URI=https://dagshub.com/...
export MLFLOW_TRACKING_TOKEN=...
export HF_TOKEN=...

AWS_PROFILE=wne-uw ./scripts/eks-bootstrap.sh dev
```

This script:
1. Runs `aws eks update-kubeconfig`
2. Applies `k8s/overlays/aws`
3. Applies the EFS StorageClass with the real filesystem ID (from `terraform output`)
4. Creates / updates `kg-secrets` in the cluster (idempotent)
5. Uploads the data CSV to S3 (skipped if not present locally)

### Push a Docker image

```bash
# manually (CI does this automatically on push to dev):
aws ecr get-login-password --region eu-central-1 --profile wne-uw \
  | docker login --username AWS --password-stdin \
    039293892587.dkr.ecr.eu-central-1.amazonaws.com

docker build -t kg-builder kg_builder_llm/
docker tag kg-builder:latest \
  039293892587.dkr.ecr.eu-central-1.amazonaws.com/kg-experiments-dev-kg-builder:latest
docker push \
  039293892587.dkr.ecr.eu-central-1.amazonaws.com/kg-experiments-dev-kg-builder:latest
```

### Run a sweep on EKS

Trigger via GitHub Actions UI → **Run Sweep** → set `configs` and `n_workers`:

```json
{"steps": 3, "candidates": 2}
```

Or with the GitHub CLI:

```bash
gh workflow run sweep.yml \
  -f configs='{"steps":3,"candidates":2}' \
  -f n_workers=1
```

Results appear in the MLflow/DagsHub UI under the project tracking URI.

---

## Architecture notes

**LLM serving**

| Environment | LLM | Embeddings |
|---|---|---|
| Local (minikube) | Ollama `qwen2:0.5b` (Helm chart) | HuggingFace TEI (CPU) |
| AWS (EKS) | vLLM `Qwen2.5-32B` (GPU, g5.xlarge) | HuggingFace TEI (GPU) |

**Sequential sweep** — AuraDB free tier allows one database, so experiments run one at a time. `sweep.py` submits Kubernetes Jobs sequentially; vLLM/TEI handle concurrency within each Job via async calls.

**Data access** — `DATA_URI` in the ConfigMap points to an S3 path. `kg_builder_llm/core/data.py` detects `s3://` prefixes and reads via boto3. For local runs, pass a local CSV path.

**Secrets** — `kg-secrets` is never committed. Create it with `eks-bootstrap.sh` or `kubectl create secret`. The `k8s/base/secret.yaml` is a template with placeholders only.

---

## CI/CD overview

| Workflow | Trigger | What it does |
|---|---|---|
| `terraform_plan.yml` | PR → `dev` touching `infra/` | `terraform plan`, posts output as PR comment |
| `terraform_deploy.yml` | Push → `dev` touching `infra/` | `terraform apply` |
| `ecr_deploy.yml` | Push → `dev` touching `kg_builder_llm/` or `Dockerfile` | Build + push image to ECR |
| `sweep.yml` | `workflow_dispatch` | OIDC auth → EKS → run sweep → log to MLflow |

All workflows authenticate to AWS via GitHub OIDC (no stored credentials). The GitHub Actions role chains to the Terraform deployment role via `sts:AssumeRole` + `sts:TagSession`.

---

## Environment variables

| Variable | Where set | Description |
|---|---|---|
| `NEO4J_URI` | `kg-secrets` | AuraDB bolt+s URI |
| `NEO4J_PASSWORD` | `kg-secrets` | AuraDB password |
| `MLFLOW_TRACKING_URI` | `kg-secrets` | DagsHub MLflow URI |
| `MLFLOW_TRACKING_TOKEN` | `kg-secrets` | DagsHub access token |
| `HF_TOKEN` | `kg-secrets` | HuggingFace token (for TEI model pull) |
| `LLM_BASE_URL` | ConfigMap | vLLM or Ollama base URL |
| `LLM_MODEL` | ConfigMap | model name |
| `EMBEDDING_BASE_URL` | ConfigMap | TEI service URL |
| `EMBEDDING_MODEL` | ConfigMap | embedding model name |
| `DATA_URI` | ConfigMap | S3 or local path to articles CSV |

# Experiment Scaling Plan — Kubernetes + Qwen 2.5 32B

Cloud-agnostic deployment: identical manifests run on local minikube and AWS EKS.
Experiments are triggered from DagsHub via MLflow Projects Kubernetes backend.

---

## Architecture

```
DagsHub UI  ──►  mlflow run (Kubernetes backend)
                       │
                       ▼
              Job: sweep  (kg-experiments namespace)
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Job: kg-builder  Job: kg-builder  ...   (one per config, parallel)
          │
          └──► MLflow child run (DagsHub tracking)

Cluster services (always present during experiment):
  Deployment: vllm          ← scales 0 → N before Jobs, N → 0 after
  Service:    vllm-svc      ← ClusterIP, stable endpoint for kg-builder Jobs
  PVC:        model-cache   ← model weights cached across runs
```

**LLM roles — both handled by Qwen 2.5 32B on vLLM:**

| Role | Volume | Notes |
|------|--------|-------|
| Entity extraction | ~100s per experiment | high concurrency via semaphore |
| Ontology evolution | 2–4 per step | sequential, low volume |

**GPU sizing (cloud only):**
- Qwen 2.5 32B 4-bit: ~18 GB VRAM
- g5.xlarge: 1× A10G (24 GB) — fits, ~$0.40/hr Spot

---

## Repository layout

```
k8s/
├── base/                        # environment-agnostic manifests
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── configmap.yaml           # LLM_BASE_URL, model name, experiment defaults
│   ├── rbac/
│   │   ├── serviceaccount.yaml  # sweep-runner ServiceAccount
│   │   └── role.yaml            # allows sweep Job to CRUD Jobs in namespace
│   ├── vllm/
│   │   ├── deployment.yaml      # vLLM server, replicas=0 at rest
│   │   ├── service.yaml         # ClusterIP on port 8000
│   │   └── pvc.yaml             # model weight cache (ReadWriteMany)
│   └── kg-builder/
│       └── job.yaml             # Job template — params injected as env vars
├── overlays/
│   ├── local/                   # minikube: CPU only, hostPath storage, stub vLLM
│   │   ├── kustomization.yaml
│   │   └── patches/
│   │       ├── vllm-cpu.yaml    # removes GPU request, uses tiny stub image
│   │       └── storage-hostpath.yaml
│   └── aws/                     # EKS: GPU node selector, EFS storage class
│       ├── kustomization.yaml
│       └── patches/
│           ├── vllm-gpu.yaml    # nvidia.com/gpu: 1, node selector
│           └── storage-efs.yaml
└── mlproject/
    ├── kubernetes_config.json   # MLflow backend config (context, namespace)
    └── sweep_values.yaml        # default param matrix for DagsHub UI
```

---

## Phase 1 — Local validation (Mac, no cost)

Validate extraction quality with Ollama before touching any cluster.

```bash
brew install ollama
ollama pull qwen2.5:32b-instruct
ollama serve   # OpenAI-compatible API on localhost:11434
```

```dotenv
# .env
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:32b-instruct
OPENAI_API_KEY=fake
```

```bash
uv run python -m kg_builder_llm.main \
  --data data/fnspid_sample_nasdaq_long_text.csv \
  --time-window-days 10 --articles-per-day 3 \
  --steps 1 --candidates 1 \
  --embedding-type local
```

If JSON output matches ontology schema → Phase 2.

---

## Phase 2 — Local Kubernetes (minikube)

Goal: full experiment flow running locally, no GPU, no real vLLM.

### 2.1 Prerequisites

```bash
brew install minikube kubectl kustomize
minikube start --cpus 4 --memory 8g --driver docker
```

### 2.2 Manifests — base layer

`k8s/base/namespace.yaml` — isolates all resources:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: kg-experiments
```

`k8s/base/configmap.yaml` — non-secret config:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kg-config
  namespace: kg-experiments
data:
  LLM_BASE_URL: "http://vllm-svc:8000/v1"
  LLM_MODEL: "Qwen/Qwen2.5-32B-Instruct"
  EMBEDDING_TYPE: "local"
  LOCAL_MODEL_NAME: "all-MiniLM-L6-v2"
```

`k8s/base/rbac/` — sweep Job needs to manage other Jobs:
```yaml
# serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sweep-runner
  namespace: kg-experiments
---
# role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: job-manager
  namespace: kg-experiments
rules:
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["create", "get", "list", "watch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "deployments/scale"]
  verbs: ["get", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sweep-runner-job-manager
  namespace: kg-experiments
subjects:
- kind: ServiceAccount
  name: sweep-runner
roleRef:
  kind: Role
  name: job-manager
  apiGroup: rbac.authorization.k8s.io
```

`k8s/base/vllm/deployment.yaml` — starts at 0 replicas:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
  namespace: kg-experiments
spec:
  replicas: 0          # sweep scales this before submitting kg-builder Jobs
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - --model
        - $(LLM_MODEL)
        - --quantization
        - bitsandbytes
        - --max-model-len
        - "4096"
        - --port
        - "8000"
        envFrom:
        - configMapRef:
            name: kg-config
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache
```

`k8s/base/kg-builder/job.yaml` — Job template (params overridden per-run):
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: kg-builder        # sweep appends -<hash> per config
  namespace: kg-experiments
spec:
  ttlSecondsAfterFinished: 3600
  backoffLimit: 0         # fail fast — no retries for experiments
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: kg-builder
        image: <account>.dkr.ecr.eu-central-1.amazonaws.com/kg-orchestrator:latest
        envFrom:
        - configMapRef:
            name: kg-config
        - secretRef:
            name: kg-secrets   # NEO4J_URI, NEO4J_PASSWORD, MLFLOW_TRACKING_URI
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
```

### 2.3 Local overlay — CPU stub for vLLM

`k8s/overlays/local/patches/vllm-cpu.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
  namespace: kg-experiments
spec:
  template:
    spec:
      containers:
      - name: vllm
        image: kennethreitz/httpbin   # stub: returns 200 to health checks
        args: []
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
```

### 2.4 Apply locally

```bash
kubectl apply -k k8s/overlays/local
kubectl get all -n kg-experiments
```

---

## Phase 3 — Sweep launcher + MLflow Projects

### 3.1 MLproject file (repo root)

```yaml
name: kg-experiments

docker_env:
  image: <account>.dkr.ecr.eu-central-1.amazonaws.com/kg-orchestrator:latest

entry_points:
  main:
    parameters:
      steps:        {type: int,   default: 3}
      candidates:   {type: int,   default: 2}
      feature_mode: {type: string, default: path}
      time_window_days: {type: int, default: 150}
    command: >
      python -m kg_builder_llm.main
        --steps {steps}
        --candidates {candidates}
        --feature-mode {feature_mode}
        --time-window-days {time_window_days}

  sweep:
    parameters:
      configs: {type: string, default: "[]"}   # JSON array of param dicts
    command: "python kg_builder_llm/scripts/sweep.py --configs '{configs}'"
```

### 3.2 sweep.py — manages vLLM lifecycle and parallel Jobs

`kg_builder_llm/scripts/sweep.py`:
- Parses `--configs` JSON array (list of param dicts)
- Creates parent MLflow run
- Patches vLLM Deployment replicas → N (computed from len(configs))
- Waits for vLLM pod Ready
- Submits N Kubernetes Jobs (one per config), each with child MLflow run ID in env
- Watches Jobs via `kubernetes` Python client until all Complete or Failed
- Patches vLLM replicas → 0
- Logs summary to parent run

### 3.3 Trigger from DagsHub

```bash
# local test
mlflow run . -e sweep \
  --backend kubernetes \
  --backend-config k8s/mlproject/kubernetes_config.json \
  -P configs='[{"steps":3,"candidates":2,"feature_mode":"path"},{"steps":3,"candidates":2,"feature_mode":"subgraph"}]'
```

DagsHub UI: set `configs` param → Run → sweep Job appears in cluster, child runs appear in MLflow.

---

## Phase 4 — EKS on AWS (Terraform)

Replaces Phase 2 cluster. Same manifests, different overlay.

### 4.1 Terraform resources (infra/)

- `eks.tf` — EKS cluster, two node groups:
  - `cpu-nodes`: `m5.xlarge` On-Demand (sweep + kg-builder Jobs)
  - `gpu-nodes`: `g5.xlarge` Spot, min=0 max=4 (vLLM only, scales to 0 at rest)
- `efs.tf` — EFS filesystem, EFS CSI driver add-on, StorageClass
- `irsa.tf` — IAM Roles for Service Accounts (IRSA): sweep-runner assumes role with EKS + ECR permissions

### 4.2 AWS overlay

`k8s/overlays/aws/patches/vllm-gpu.yaml` — adds GPU resource and node selector:
```yaml
spec:
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: g5.xlarge
      containers:
      - name: vllm
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: "22Gi"
          requests:
            nvidia.com/gpu: "1"
            memory: "22Gi"
```

One-time model download to EFS (run once, weights persist across cluster restarts):
```bash
kubectl run model-download \
  --image=vllm/vllm-openai:latest \
  --restart=Never \
  --overrides='{"spec":{"nodeSelector":{"node.kubernetes.io/instance-type":"g5.xlarge"}}}' \
  -n kg-experiments \
  -- python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-32B-Instruct')"
```

### 4.3 kubeconfig for DagsHub

```bash
aws eks update-kubeconfig --name kg-experiments --region eu-central-1 --profile wne-uw
```

Store `~/.kube/config` (EKS context only) as DagsHub secret `KUBECONFIG_B64` (base64-encoded).
The sweep container decodes it at runtime → `mlflow run` submits Jobs to EKS.

---

## Cost estimate (AWS, per experiment batch)

| Component | Instance | Price | 3-hr batch |
|-----------|----------|-------|------------|
| 1× vLLM worker | g5.xlarge Spot | ~$0.40/hr | ~$1.20 |
| N× kg-builder Jobs | m5.xlarge On-Demand | ~$0.19/hr | ~$0.57 per Job |
| EFS model cache | 65 GB | $0.30/GB-month | ~$0.03/day |
| EKS control plane | — | $0.10/hr | ~$0.30 |
| **2 kg-builder configs** | | | **~$3.30 total** |

CPU nodes scale to 0 between batches (Cluster Autoscaler).
GPU node scales to 0 after vLLM Deployment replicas → 0.

---

## Sequence summary

```
Phase 1 (local, Ollama)
  └── validate extraction quality on 10-day sample

Phase 2 (minikube)
  └── apply k8s/overlays/local
  └── test sweep.py with 2 configs, CPU stub vLLM
  └── verify child MLflow runs in DagsHub

Phase 3 (MLflow Projects)
  └── MLproject file + sweep entry point
  └── mlflow run --backend kubernetes
  └── trigger from DagsHub UI

Phase 4 (EKS)
  └── terraform apply (EKS + EFS + IRSA)
  └── kubectl apply -k k8s/overlays/aws
  └── one-time model download to EFS
  └── run full experiment batch from DagsHub
```

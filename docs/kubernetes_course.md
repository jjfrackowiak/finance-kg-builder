# Kubernetes for ML Experiment Platforms — Practical Course

Everything here is grounded in the `kg-experiments` platform you are building.
No generic examples — every concept maps directly to a real resource in the project.

---

## Mental model before anything else

Kubernetes is a **desired-state system**. You write YAML that says "I want 3 replicas of vLLM running" and Kubernetes figures out how to make that true and keep it true. You never SSH into machines. You declare what you want, and the control plane reconciles reality toward it.

Three things to hold in your head:

| Term | What it is |
|------|-----------|
| **Control plane** | The brain — API server, scheduler, controller manager. Runs on master nodes. You never touch it directly. |
| **Node** | A machine (VM or bare metal) that runs your workloads. |
| **Pod** | The smallest deployable unit — one or more containers that share a network and storage. |

Everything else is a layer on top of Pods.

---

## Module 1 — Namespaces

A namespace is a virtual cluster inside a cluster. It isolates resources so that `kg-experiments` resources don't collide with anything else running on the same cluster.

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: kg-experiments
```

```bash
kubectl apply -f k8s/base/namespace.yaml
kubectl get namespaces
kubectl get pods -n kg-experiments        # -n flag scopes to namespace
kubectl config set-context --current --namespace=kg-experiments  # set default
```

**Why it matters here:** All your resources (vLLM, kg-builder Jobs, ConfigMaps, Secrets, RBAC) live in `kg-experiments`. On a shared company cluster this prevents your GPU workloads from interfering with other teams.

---

## Module 2 — Pods

A Pod wraps one or more containers. Containers in the same Pod share:
- The same IP address
- The same localhost network
- The same mounted volumes

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: debug
  namespace: kg-experiments
spec:
  containers:
  - name: shell
    image: busybox
    command: ["sleep", "3600"]
```

```bash
kubectl apply -f pod.yaml
kubectl get pods -n kg-experiments
kubectl describe pod debug -n kg-experiments   # full detail, events, errors
kubectl logs debug -n kg-experiments           # stdout of the container
kubectl exec -it debug -n kg-experiments -- sh # shell into it
kubectl delete pod debug -n kg-experiments
```

**You almost never write raw Pods.** Pods are ephemeral — if the node dies, the Pod is gone. You use higher-level resources (Deployment, Job) that manage Pods for you.

---

## Module 3 — ConfigMap

A ConfigMap stores non-secret configuration as key-value pairs. Containers consume it as environment variables or mounted files.

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

Referencing in a container:
```yaml
envFrom:
- configMapRef:
    name: kg-config
```

This injects every key in the ConfigMap as an environment variable — exactly what `kg_builder_llm/config.py` reads via `os.getenv()`.

```bash
kubectl apply -f k8s/base/configmap.yaml
kubectl get configmap kg-config -n kg-experiments -o yaml
kubectl describe configmap kg-config -n kg-experiments
```

---

## Module 4 — Secrets

Secrets hold sensitive values (Neo4j password, HuggingFace token, MLflow URI with credentials). They are base64-encoded at rest (not encrypted by default — encryption at rest is a cluster configuration option).

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: kg-secrets
  namespace: kg-experiments
type: Opaque
stringData:                        # stringData auto-encodes to base64
  NEO4J_URI: "neo4j+s://..."
  NEO4J_PASSWORD: "..."
  MLFLOW_TRACKING_URI: "https://dagshub.com/..."
  HF_TOKEN: "hf_..."
```

```bash
kubectl create secret generic kg-secrets \
  --from-env-file=.env \
  -n kg-experiments

kubectl get secret kg-secrets -n kg-experiments -o jsonpath='{.data.NEO4J_PASSWORD}' | base64 -d
```

**Never commit Secret YAML with real values.** In cloud deployments, use **External Secrets Operator** (ESO) instead — it pulls from AWS Secrets Manager / Azure Key Vault and creates the Kubernetes Secret automatically:

```yaml
# ExternalSecret CRD (ESO) — cloud-agnostic interface
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: kg-secrets
  namespace: kg-experiments
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager      # swap for azure-keyvault, ibm-secrets-manager
    kind: ClusterSecretStore
  target:
    name: kg-secrets
  data:
  - secretKey: NEO4J_PASSWORD
    remoteRef:
      key: kg-experiments/neo4j
      property: password
```

Locally you just `kubectl apply` the Secret directly from `.env`. In cloud, ESO handles it — same manifest structure for your Pods either way.

---

## Module 5 — Deployment

A Deployment manages a set of identical, long-running Pods. It handles:
- Desired replica count
- Rolling updates (new image → gradually replace old Pods)
- Self-healing (Pod crashes → Deployment recreates it)

**vLLM is a Deployment** because it is a persistent service that needs to be reachable for the duration of an experiment batch.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
  namespace: kg-experiments
spec:
  replicas: 0                      # 0 at rest — sweep scales this up
  selector:
    matchLabels:
      app: vllm
  template:                        # Pod template — this is what actually runs
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
        - secretRef:
            name: kg-secrets
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "20Gi"
          limits:
            memory: "22Gi"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache
```

```bash
kubectl apply -f k8s/base/vllm/deployment.yaml
kubectl get deployment vllm -n kg-experiments
kubectl get pods -n kg-experiments -l app=vllm    # -l filters by label
kubectl rollout status deployment/vllm -n kg-experiments

# scale manually (sweep.py does this via Python client)
kubectl scale deployment vllm --replicas=2 -n kg-experiments
kubectl scale deployment vllm --replicas=0 -n kg-experiments
```

**Key insight:** `replicas: 0` means the Deployment exists (configuration is stored) but no Pods run. The sweep script patches this to N before submitting kg-builder Jobs, then back to 0 after — no GPU cost at rest.

---

## Module 6 — Service

A Service gives a stable DNS name and IP to a set of Pods selected by label. Without it, Pod IPs change every restart.

**vLLM needs a Service** so kg-builder Jobs can reach it at `http://vllm-svc:8000/v1` regardless of how many replicas are running or which node they landed on.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-svc
  namespace: kg-experiments
spec:
  type: ClusterIP             # internal only — not exposed outside the cluster
  selector:
    app: vllm                 # routes to all Pods with this label
  ports:
  - port: 8000
    targetPort: 8000
```

```bash
kubectl apply -f k8s/base/vllm/service.yaml
kubectl get service vllm-svc -n kg-experiments

# test DNS resolution from inside the cluster
kubectl run curl-test --image=curlimages/curl -it --rm \
  -n kg-experiments -- curl http://vllm-svc:8000/health
```

**DNS works like this inside the cluster:**
- Same namespace: `http://vllm-svc:8000`
- Cross-namespace: `http://vllm-svc.kg-experiments.svc.cluster.local:8000`

The ConfigMap sets `LLM_BASE_URL: "http://vllm-svc:8000/v1"` — this is how kg-builder Jobs find vLLM without hardcoding IPs.

---

## Module 7 — Job

A Job runs Pods to **completion** rather than keeping them running forever. It is the right primitive for experiments: run, produce output, exit. Kubernetes marks the Job `Complete` or `Failed`.

**kg-builder is a Job** — it processes articles, builds the KG, logs to MLflow, then exits.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: kg-builder-abc123       # sweep appends a unique suffix per config
  namespace: kg-experiments
spec:
  ttlSecondsAfterFinished: 3600 # auto-delete 1h after completion (keeps cluster clean)
  backoffLimit: 0               # no retries — failed experiment = failed Job, investigate logs
  template:
    spec:
      restartPolicy: Never      # required for Jobs (Never or OnFailure)
      containers:
      - name: kg-builder
        image: ghcr.io/jjfrackowiak/kg-orchestrator:latest
        args:
        - --steps
        - "3"
        - --candidates
        - "2"
        - --feature-mode
        - "path"
        - --time-window-days
        - "150"
        envFrom:
        - configMapRef:
            name: kg-config
        - secretRef:
            name: kg-secrets
        env:
        - name: MLFLOW_PARENT_RUN_ID    # injected per-Job by sweep.py
          value: "abc123..."
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
```

```bash
kubectl apply -f job.yaml
kubectl get jobs -n kg-experiments
kubectl get pods -n kg-experiments                       # Job creates a Pod
kubectl logs -n kg-experiments -l job-name=kg-builder-abc123 --follow
kubectl describe job kg-builder-abc123 -n kg-experiments # see completion status
```

**Why `backoffLimit: 0`?** Retrying a failed ML experiment silently hides bugs and wastes GPU time. Fail fast, read the logs, fix the issue.

---

## Module 8 — Storage (PV, PVC, StorageClass)

Three layers:

| Resource | What it is |
|----------|-----------|
| **PersistentVolume (PV)** | Actual storage — a disk, NFS share, EFS filesystem |
| **PersistentVolumeClaim (PVC)** | A request for storage — "I need 100Gi ReadWriteMany" |
| **StorageClass** | A recipe for dynamically creating PVs — different per cloud |

Your Pods reference a PVC. The PVC binds to a PV. The StorageClass determines what kind of PV is created.

**Model weight cache PVC** — Qwen 2.5 32B is 65GB. Without caching, every new vLLM Pod downloads it from HuggingFace (~15 min). With a PVC mounted at `/root/.cache/huggingface`, it downloads once and persists.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: kg-experiments
spec:
  accessModes:
  - ReadWriteMany             # multiple vLLM Pods can mount it simultaneously
  storageClassName: efs-sc    # patched per overlay (hostPath locally, efs-sc on AWS)
  resources:
    requests:
      storage: 100Gi
```

**Access modes:**
- `ReadWriteOnce` — one node at a time (standard block storage, EBS)
- `ReadWriteMany` — multiple nodes simultaneously (NFS, EFS, Azure Files) ← you need this for vLLM

**Per-cloud StorageClass names (set in overlays):**

| Environment | StorageClass | Backend |
|-------------|-------------|---------|
| local (minikube) | `standard` | hostPath |
| AWS EKS | `efs-sc` | Amazon EFS |
| Azure AKS | `azurefile` | Azure Files |
| IBM IKS | `ibmc-file-gold` | IBM Cloud File Storage |
| On-prem | `nfs-client` | NFS provisioner |

```bash
kubectl get pvc -n kg-experiments
kubectl describe pvc model-cache -n kg-experiments   # check if Bound or Pending
kubectl get storageclass                              # see available StorageClasses
```

---

## Module 9 — RBAC

RBAC controls what identities can do inside the cluster.

Three resources:

| Resource | What it is |
|----------|-----------|
| **ServiceAccount** | An identity for a Pod (not a human) |
| **Role** | A set of permissions scoped to a namespace |
| **RoleBinding** | Binds a Role to a ServiceAccount |

**Why sweep.py needs RBAC:** The sweep Job runs inside the cluster and uses the Kubernetes Python client to create other Jobs and scale the vLLM Deployment. Without explicit permissions it gets `403 Forbidden`.

```yaml
# ServiceAccount — identity for the sweep Pod
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sweep-runner
  namespace: kg-experiments
---
# Role — exactly the permissions sweep needs, nothing more
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
# RoleBinding — connect them
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sweep-runner-job-manager
  namespace: kg-experiments
subjects:
- kind: ServiceAccount
  name: sweep-runner
  namespace: kg-experiments
roleRef:
  kind: Role
  name: job-manager
  apiGroup: rbac.authorization.k8s.io
```

The sweep Job spec references the ServiceAccount:
```yaml
spec:
  template:
    spec:
      serviceAccountName: sweep-runner
```

The Kubernetes Python client inside the Pod then automatically uses this ServiceAccount token — no credentials to manage.

```bash
kubectl auth can-i create jobs \
  --as=system:serviceaccount:kg-experiments:sweep-runner \
  -n kg-experiments              # should return "yes"
```

---

## Module 10 — Kustomize

Kustomize is built into `kubectl` (`kubectl apply -k`). It lets you define a **base** set of manifests and **overlay** patches per environment — no templating, just strategic merges.

**Directory structure:**
```
k8s/
├── base/
│   ├── kustomization.yaml       # lists all base resources
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── rbac/...
│   ├── vllm/...
│   └── kg-builder/...
└── overlays/
    ├── local/
    │   ├── kustomization.yaml   # references base + patches
    │   └── patches/
    │       ├── vllm-cpu.yaml    # replace GPU container with CPU stub
    │       └── storage-hostpath.yaml  # use standard StorageClass
    └── aws/
        ├── kustomization.yaml
        └── patches/
            ├── vllm-gpu.yaml
            ├── storage-efs.yaml
            └── image-ecr.yaml   # patch image to ECR URL
```

**base/kustomization.yaml:**
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- namespace.yaml
- configmap.yaml
- rbac/serviceaccount.yaml
- rbac/role.yaml
- vllm/deployment.yaml
- vllm/service.yaml
- vllm/pvc.yaml
- kg-builder/job.yaml
```

**overlays/local/kustomization.yaml:**
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- ../../base
patches:
- path: patches/vllm-cpu.yaml
  target:
    kind: Deployment
    name: vllm
- path: patches/storage-hostpath.yaml
  target:
    kind: PersistentVolumeClaim
    name: model-cache
images:
- name: ghcr.io/jjfrackowiak/kg-orchestrator
  newTag: latest
```

**overlays/aws/kustomization.yaml:**
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
- ../../base
patches:
- path: patches/vllm-gpu.yaml
  target:
    kind: Deployment
    name: vllm
- path: patches/storage-efs.yaml
  target:
    kind: PersistentVolumeClaim
    name: model-cache
images:
- name: ghcr.io/jjfrackowiak/kg-orchestrator
  newName: <account>.dkr.ecr.eu-central-1.amazonaws.com/kg-orchestrator
  newTag: latest
```

```bash
# preview what will be applied (no apply yet)
kubectl kustomize k8s/overlays/local

# apply
kubectl apply -k k8s/overlays/local
kubectl apply -k k8s/overlays/aws
```

**Strategic merge patch** — you only write what changes. A patch for the PVC storage class:
```yaml
# patches/storage-efs.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  namespace: kg-experiments
spec:
  storageClassName: efs-sc
```
Kustomize merges this into the base PVC, replacing only `storageClassName`. Everything else stays the same.

---

## Module 11 — Helm

Helm is a **package manager** for Kubernetes. A Helm chart bundles all the manifests for an application into a versioned, configurable package.

**You use Helm for vLLM** because there is an official chart — battle-tested, maintained, handles edge cases. You write a values file to configure it.

```bash
helm repo add vllm https://vllm-project.github.io/vllm
helm repo update
helm show values vllm/vllm-chart > k8s/helm-values/vllm-defaults.yaml  # inspect all options
```

**k8s/helm-values/vllm-local.yaml** (CPU stub for minikube):
```yaml
replicaCount: 0
image:
  repository: kennethreitz/httpbin  # stub — real vLLM needs GPU
  tag: latest
resources:
  requests:
    cpu: 100m
    memory: 128Mi
service:
  type: ClusterIP
  port: 8000
```

**k8s/helm-values/vllm-aws.yaml** (real GPU on EKS):
```yaml
replicaCount: 0
model: Qwen/Qwen2.5-32B-Instruct
extraArgs:
- --quantization
- bitsandbytes
- --max-model-len
- "4096"
resources:
  limits:
    nvidia.com/gpu: "1"
    memory: 22Gi
  requests:
    nvidia.com/gpu: "1"
    memory: 22Gi
nodeSelector:
  node-role: gpu
persistence:
  enabled: true
  existingClaim: model-cache
env:
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: kg-secrets
      key: HF_TOKEN
```

```bash
# install/upgrade
helm upgrade --install vllm vllm/vllm-chart \
  -f k8s/helm-values/vllm-aws.yaml \
  -n kg-experiments \
  --atomic                  # rollback automatically if rollout fails

helm list -n kg-experiments
helm history vllm -n kg-experiments
helm rollback vllm 1 -n kg-experiments   # roll back to revision 1
```

**Helm vs Kustomize — when to use which:**

| Helm | Kustomize |
|------|-----------|
| Third-party charts (vLLM, ESO, cert-manager) | Your own resources |
| Complex templating (loops, conditionals) | Simple environment overlays |
| Release management (upgrade, rollback) | Strategic merge patches |

---

## Module 12 — GPU Scheduling

Kubernetes schedules GPU workloads via **resource requests**. The NVIDIA device plugin runs as a DaemonSet on GPU nodes, advertising `nvidia.com/gpu` as a schedulable resource.

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"    # requests must equal limits for GPU
```

**Node labels for cloud-agnostic GPU selection** — instead of `node.kubernetes.io/instance-type: g5.xlarge` (AWS-specific), label GPU nodes with a generic label at provisioning time:

```bash
kubectl label node <gpu-node-name> node-role=gpu
```

Then in the vLLM Deployment:
```yaml
nodeSelector:
  node-role: gpu
```

This works identically on AWS (`g5.xlarge`), Azure (`Standard_NC6s_v3`), IBM (`cx2-8x16` with GPU), or bare metal with an NVIDIA card.

**Cluster Autoscaler** — on cloud clusters, GPU nodes scale to 0 when the vLLM Deployment has `replicas: 0`. The Cluster Autoscaler drains and terminates idle GPU nodes. When replicas go back to N, Autoscaler provisions a new GPU node (~3-5 min cold start without EFS cache, ~60s with).

---

## Module 13 — sweep.py and the Kubernetes Python client

The sweep Job manages the experiment lifecycle using `kubernetes` Python client — the same API as `kubectl` but from Python.

```python
from kubernetes import client, config

# load kubeconfig (in-cluster when running as a Job)
config.load_incluster_config()

apps_v1 = client.AppsV1Api()
batch_v1 = client.BatchV1Api()

# scale vLLM up
apps_v1.patch_namespaced_deployment_scale(
    name="vllm",
    namespace="kg-experiments",
    body={"spec": {"replicas": n_workers}}
)

# submit a kg-builder Job
batch_v1.create_namespaced_job(
    namespace="kg-experiments",
    body=build_job_manifest(config_dict, parent_run_id)
)

# wait for all Jobs
while True:
    jobs = batch_v1.list_namespaced_job(namespace="kg-experiments", label_selector="sweep-id=abc")
    if all(j.status.completion_time for j in jobs.items):
        break
    time.sleep(10)

# scale vLLM back to 0
apps_v1.patch_namespaced_deployment_scale(
    name="vllm", namespace="kg-experiments",
    body={"spec": {"replicas": 0}}
)
```

`config.load_incluster_config()` automatically uses the ServiceAccount token mounted at `/var/run/secrets/kubernetes.io/serviceaccount/token` — the RBAC Role we defined grants exactly the permissions this code needs.

---

## Module 14 — MLflow Projects Kubernetes backend

MLflow Projects lets DagsHub trigger a run that submits a Kubernetes Job.

**MLproject (repo root):**
```yaml
name: kg-experiments
docker_env:
  image: ghcr.io/jjfrackowiak/kg-orchestrator:latest
entry_points:
  sweep:
    parameters:
      configs: {type: string, default: "[]"}
    command: "python kg_builder_llm/scripts/sweep.py --configs '{configs}'"
```

**k8s/mlproject/kubernetes_config.json:**
```json
{
  "kube-context": "kg-experiments-eks",
  "kube-namespace": "kg-experiments",
  "resources.requests.memory": "512Mi",
  "resources.requests.cpu": "0.5"
}
```

```bash
mlflow run . -e sweep \
  --backend kubernetes \
  --backend-config k8s/mlproject/kubernetes_config.json \
  -P configs='[{"steps":3,"feature_mode":"path"},{"steps":3,"feature_mode":"subgraph"}]'
```

MLflow creates a Kubernetes Job for the sweep entry point. The sweep Job then creates child kg-builder Jobs. All results flow back to DagsHub via the MLflow tracking URI in the Secret.

---

## Module 15 — Essential kubectl reference

```bash
# context / cluster
kubectl config get-contexts
kubectl config use-context minikube
kubectl config use-context kg-experiments-eks

# apply
kubectl apply -f file.yaml
kubectl apply -k k8s/overlays/local     # Kustomize
kubectl delete -k k8s/overlays/local    # tear down

# inspect
kubectl get all -n kg-experiments
kubectl get pods -n kg-experiments -w   # -w watches for changes
kubectl describe pod <name> -n kg-experiments
kubectl logs <pod> -n kg-experiments --follow
kubectl events -n kg-experiments        # recent cluster events

# debug
kubectl exec -it <pod> -n kg-experiments -- bash
kubectl port-forward svc/vllm-svc 8000:8000 -n kg-experiments  # access service locally
kubectl run debug --image=busybox -it --rm -n kg-experiments -- sh

# Jobs
kubectl get jobs -n kg-experiments
kubectl delete job <name> -n kg-experiments

# scale
kubectl scale deployment vllm --replicas=1 -n kg-experiments

# Helm
helm upgrade --install <release> <chart> -f values.yaml -n kg-experiments --atomic
helm uninstall <release> -n kg-experiments
```

---

## Build order for this project

```
1. minikube start
2. kubectl apply -f k8s/base/namespace.yaml
3. kubectl apply -f k8s/base/configmap.yaml
4. kubectl create secret generic kg-secrets --from-env-file=.env -n kg-experiments
5. kubectl apply -f k8s/base/rbac/
6. helm upgrade --install vllm vllm/vllm-chart -f k8s/helm-values/vllm-local.yaml -n kg-experiments
7. kubectl apply -f k8s/base/vllm/pvc.yaml
8. kubectl apply -k k8s/overlays/local          # ties it all together
9. kubectl apply -f test-job.yaml               # single kg-builder Job, verify MLflow run appears
10. python kg_builder_llm/scripts/sweep.py --configs '[{...},{...}]'  # local sweep test
11. mlflow run . -e sweep --backend kubernetes ...  # DagsHub trigger test
```

Each step is independently verifiable before moving to the next.

#!/usr/bin/env bash
# Bootstrap a freshly-created EKS cluster:
#   1. Update kubeconfig
#   2. Apply k8s/overlays/aws (namespace, configmap, deployments, RBAC)
#   3. Apply EFS StorageClass with real filesystem ID from terraform output
#   4. Create/update kg-secrets in the cluster (idempotent)
#   5. Upload data CSV to S3 (skipped if file absent — CI path)
#
# Usage (local):
#   export NEO4J_URI=... NEO4J_PASSWORD=... MLFLOW_TRACKING_URI=... \
#          MLFLOW_TRACKING_TOKEN=... HF_TOKEN=...
#   ./scripts/eks-bootstrap.sh [dev|prod]
#
# Credentials can also be placed in a .env file at the repo root:
#   NEO4J_URI=bolt+s://...
#   NEO4J_PASSWORD=...
#   MLFLOW_TRACKING_URI=...
#   MLFLOW_TRACKING_TOKEN=...
#   HF_TOKEN=...
#
# AWS credentials: set AWS_PROFILE for local runs; in CI credentials come
# from OIDC so no profile is needed.

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

ENV="${1:-dev}"
REGION="${AWS_REGION:-eu-central-1}"
CLUSTER_NAME="kg-experiments-${ENV}-eks"
NAMESPACE="kg-experiments"
S3_BUCKET="kg-experiments-data-039293892587"
DATA_FILE="data/fnspid_sample_nasdaq_long_text.csv"
INFRA_DIR="infra"
OVERLAY_DIR="k8s/overlays/aws"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Load .env if present ──────────────────────────────────────────────────────

if [[ -f ".env" ]]; then
  echo "→ Loading credentials from .env"
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

# ── Validate required env vars ────────────────────────────────────────────────

required=(NEO4J_URI NEO4J_PASSWORD MLFLOW_TRACKING_URI MLFLOW_TRACKING_TOKEN HF_TOKEN)
missing=()
for var in "${required[@]}"; do
  [[ -z "${!var:-}" ]] && missing+=("$var")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "ERROR: missing required env vars: ${missing[*]}"
  echo "       Set them in the environment or in a .env file at the repo root."
  exit 1
fi

# ── AWS profile (optional — skipped in CI) ────────────────────────────────────

AWS_ARGS=()
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS_ARGS=(--profile "$AWS_PROFILE")
fi

# ── 1. Update kubeconfig ──────────────────────────────────────────────────────
# Authenticate as the Terraform deployment role so kubectl has cluster-admin
# access regardless of which IAM identity is currently active.

echo "→ Configuring kubectl for cluster: $CLUSTER_NAME"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text "${AWS_ARGS[@]}")
DEPLOY_ROLE_ARN="${DEPLOY_ROLE_ARN:-arn:aws:iam::${ACCOUNT_ID}:role/kg-experiments-${ENV}-terraform-deployment}"

aws eks update-kubeconfig \
  --name "$CLUSTER_NAME" \
  --region "$REGION" \
  --role-arn "$DEPLOY_ROLE_ARN" \
  "${AWS_ARGS[@]}"

# ── 2. Apply k8s overlay ──────────────────────────────────────────────────────

echo "→ Applying k8s overlay: $OVERLAY_DIR"
kubectl apply -k "$OVERLAY_DIR"

# ── 3. Apply EFS StorageClass with real filesystem ID ─────────────────────────

echo "→ Fetching EFS filesystem ID from terraform"
EFS_ID=$(terraform -chdir="$INFRA_DIR" output -raw efs_id)
echo "   EFS ID: $EFS_ID"

kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: ${EFS_ID}
  directoryPerms: "700"
reclaimPolicy: Delete
volumeBindingMode: Immediate
EOF

# ── 4. Create / update kg-secrets (idempotent via dry-run + apply) ────────────

echo "→ Syncing kg-secrets to namespace: $NAMESPACE"
kubectl create secret generic kg-secrets \
  --namespace "$NAMESPACE" \
  --from-literal=NEO4J_URI="$NEO4J_URI" \
  --from-literal=NEO4J_PASSWORD="$NEO4J_PASSWORD" \
  --from-literal=MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
  --from-literal=MLFLOW_TRACKING_TOKEN="$MLFLOW_TRACKING_TOKEN" \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

# ── 5. Upload data CSV to S3 (skipped if not present locally) ─────────────────

if [[ -f "$DATA_FILE" ]]; then
  echo "→ Uploading $DATA_FILE to s3://$S3_BUCKET/"
  aws s3 cp "$DATA_FILE" "s3://${S3_BUCKET}/" "${AWS_ARGS[@]}"
else
  echo "→ $DATA_FILE not found locally — skipping S3 upload (run manually or in CI)"
fi

echo ""
echo "✓ Bootstrap complete for cluster: $CLUSTER_NAME"
echo "  kubectl get nodes"

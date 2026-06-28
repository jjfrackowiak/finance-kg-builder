#!/usr/bin/env bash
# Upload local experiment data to S3.
# This is the only bootstrap step that cannot run in CI (file is local only).
#
# Usage:
#   AWS_PROFILE=wne-uw ./scripts/upload-data.sh [file] [env]
#
#   file  path to CSV  (default: data/fnspid_sample_nasdaq_long_text.csv)
#   env   dev | prod   (default: dev)

set -euo pipefail

DATA_FILE="${1:-data/fnspid_sample_nasdaq_long_text.csv}"
ENV="${2:-dev}"
AWS_PROFILE="${AWS_PROFILE:-wne-uw}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: $DATA_FILE not found"
  exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity \
  --query Account --output text --profile "$AWS_PROFILE")
BUCKET="kg-experiments-data-${ACCOUNT_ID}"

echo "→ Uploading $DATA_FILE to s3://$BUCKET/"
aws s3 cp "$DATA_FILE" "s3://$BUCKET/" --profile "$AWS_PROFILE"
echo "✓ Done — s3://$BUCKET/$(basename "$DATA_FILE")"

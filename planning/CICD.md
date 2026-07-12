# CI/CD — Terraform + GitHub OIDC

GitHub Actions runs `terraform plan` on every PR and `terraform apply` on merge to `main`.
No static AWS credentials anywhere — authentication via GitHub OIDC.

---

## Directory Structure

```
infra/
  bootstrap/          ← run locally once; owns its own local state
    main.tf           ← S3 bucket, DynamoDB lock table, OIDC provider, IAM role
    outputs.tf
  backend.tf          ← S3 backend config (points at bootstrap-created bucket)
  main.tf             ← actual infrastructure (ECS, EFS, ALB, ECR…)
  variables.tf
  outputs.tf
.github/
  workflows/
    terraform.yml
```

---

## Phase 0 — Bootstrap (local, one-time)

Creates the S3 backend and the GitHub OIDC trust — everything CI/CD depends on.
Uses local Terraform state (bootstrap manages itself). Run once with your IAM user.

### `infra/bootstrap/main.tf`

```hcl
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region  = "eu-central-1"
  profile = "wne-uw"
}

# ── S3 bucket for Terraform state ──────────────────────────────────────────

resource "aws_s3_bucket" "tfstate" {
  bucket = "kg-experiments-tfstate-039293892587"
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── DynamoDB table for state locking ────────────────────────────────────────

resource "aws_dynamodb_table" "tflock" {
  name         = "kg-experiments-tflock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ── GitHub OIDC provider ─────────────────────────────────────────────────────

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# ── IAM role assumed by GitHub Actions ──────────────────────────────────────

resource "aws_iam_role" "github_actions" {
  name = "github-actions-kg-experiments"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          # Scope to this repo only; restrict further to main branch for apply if desired
          "token.actions.githubusercontent.com:sub" = "repo:jjfrackowiak/finance-kg-builder:*"
        }
      }
    }]
  })
}

# Permissions: broad enough for ECS/EFS/ALB/ECR/Secrets infra
# Tighten to specific actions after the setup is stable
resource "aws_iam_role_policy_attachment" "github_actions_admin" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}
```

### `infra/bootstrap/outputs.tf`

```hcl
output "tfstate_bucket" {
  value = aws_s3_bucket.tfstate.bucket
}

output "tflock_table" {
  value = aws_dynamodb_table.tflock.name
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions.arn
}

output "oidc_provider_arn" {
  value = aws_iam_openid_connect_provider.github.arn
}
```

### Run bootstrap locally

```bash
cd infra/bootstrap
terraform init
terraform apply
```

Note the outputs — you'll need `github_actions_role_arn` for the GitHub Actions workflow.

---

## Phase 1 — Configure Main Infra Backend

After bootstrap, point the main infra at the S3 backend.

### `infra/backend.tf`

```hcl
terraform {
  backend "s3" {
    bucket         = "kg-experiments-tfstate-039293892587"
    key            = "infra/terraform.tfstate"
    region         = "eu-central-1"
    dynamodb_table = "kg-experiments-tflock"
    encrypt        = true
    profile        = "wne-uw"   # only used for local runs; CI uses OIDC
  }
}
```

Initialise the backend (migrates any existing local state to S3):

```bash
cd infra
terraform init
```

---

## Phase 2 — GitHub Actions Workflow

### `.github/workflows/terraform.yml`

```yaml
name: Terraform

on:
  pull_request:
    paths: ["infra/**"]
  push:
    branches: [main]
    paths: ["infra/**"]

permissions:
  id-token: write       # required for OIDC token
  contents: read
  pull-requests: write  # required to post plan as PR comment

env:
  TF_WORKING_DIR: infra/
  AWS_REGION: eu-central-1

jobs:
  terraform:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::039293892587:role/github-actions-kg-experiments
          aws-region: ${{ env.AWS_REGION }}

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "~1.9"

      - name: Terraform Init
        run: terraform init
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Terraform Format Check
        run: terraform fmt -check -recursive
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Terraform Validate
        run: terraform validate
        working-directory: ${{ env.TF_WORKING_DIR }}

      - name: Terraform Plan
        id: plan
        run: terraform plan -no-color -out=tfplan
        working-directory: ${{ env.TF_WORKING_DIR }}
        continue-on-error: true   # post comment even if plan fails

      - name: Post Plan as PR Comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const output = `#### Terraform Plan \`${{ steps.plan.outcome }}\`
            <details><summary>Show Plan</summary>

            \`\`\`
            ${{ steps.plan.outputs.stdout }}
            \`\`\`
            </details>`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: output
            });

      - name: Fail if Plan Failed
        if: steps.plan.outcome == 'failure'
        run: exit 1

      - name: Terraform Apply
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: terraform apply -auto-approve tfplan
        working-directory: ${{ env.TF_WORKING_DIR }}
```

---

## GitHub Repo Setup

No secrets needed — OIDC is credential-free. One setting to enable:

```
GitHub repo → Settings → Actions → General
  → "Allow GitHub Actions to create and approve pull requests" ✓
  → Workflow permissions → "Read and write permissions" ✓
```

The role ARN is hardcoded in the workflow (`039293892587` / `github-actions-kg-experiments`).
If you want it as a variable instead, add a repo variable (not secret):
```
Settings → Secrets and variables → Actions → Variables
  TF_ROLE_ARN = arn:aws:iam::039293892587:role/github-actions-kg-experiments
```

---

## Sequence Summary

```
Phase 0 (local, once)
  └── cd infra/bootstrap && terraform init && terraform apply
  └── S3 bucket + DynamoDB lock table created
  └── GitHub OIDC provider registered in AWS
  └── IAM role github-actions-kg-experiments created

Phase 1 (local, once)
  └── cd infra && terraform init   ← backend now points to S3

Phase 2 (every PR / merge)
  └── PR opened → GitHub Actions → terraform plan → posted as PR comment
  └── PR merged to main → terraform apply → infra updated
  └── No AWS credentials stored anywhere in GitHub
```

---

## Notes

- **Bootstrap state** lives locally (`infra/bootstrap/terraform.tfstate`). Commit it or keep it safe — losing it means re-importing the S3 bucket and OIDC provider manually.
- **AdministratorAccess** on the GitHub Actions role is intentional for now. Scope it down to specific IAM actions once the full infra is stable.
- The `profile = "wne-uw"` in `backend.tf` is ignored by CI (OIDC takes over); it only applies to local `terraform` runs.

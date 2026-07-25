terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

locals {
  prefix = "${var.usecase}-${var.environment}"
  tags = merge(var.tags, {
    Environment = var.environment
    UseCase     = var.usecase
    ManagedBy   = "terraform"
  })
}

# ── S3 access logging bucket ─────────────────────────────────────────────────

resource "aws_s3_bucket" "logs" {
  bucket        = "${local.prefix}-logs-${var.account_id}"
  force_destroy = var.environment == "dev"
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "logs" {
  bucket = aws_s3_bucket.logs.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket                  = aws_s3_bucket.logs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "logs" {
  bucket = aws_s3_bucket.logs.id
  rule { object_ownership = "BucketOwnerPreferred" }
}

# ── S3 Terraform state bucket ────────────────────────────────────────────────

resource "aws_s3_bucket" "tfstate" {
  bucket        = "${local.prefix}-tfstate-${var.account_id}"
  force_destroy = false
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_logging" "tfstate" {
  bucket        = aws_s3_bucket.tfstate.id
  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "tfstate-access/"
}

# ── GitHub OIDC provider ─────────────────────────────────────────────────────

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
  tags            = local.tags
}

# ── GitHub Actions IAM role (assumes via OIDC) ───────────────────────────────

resource "aws_iam_role" "github" {
  name = "${local.prefix}-github-actions"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*"
        }
      }
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "github_assume_deployment" {
  name = "assume-terraform-deployment-role"
  role = aws_iam_role.github.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["sts:AssumeRole", "sts:TagSession"]
      Resource = aws_iam_role.terraform_deployment.arn
    }]
  })
}

# ── Terraform deployment IAM role ────────────────────────────────────────────
# Assumed by both GitHub Actions (via github role) and local runs (via IAM user)

resource "aws_iam_role" "terraform_deployment" {
  name                 = "${local.prefix}-terraform-deployment"
  max_session_duration = 14400 # 4h — sweeps assuming this role can run past the 1h default

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { AWS = aws_iam_role.github.arn }
        Action    = "sts:AssumeRole"
      },
      {
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${var.account_id}:root" }
        Action    = "sts:AssumeRole"
        Condition = {
          StringEquals = { "aws:PrincipalArn" = var.local_operator_arn }
        }
      },
      {
        # Self-trust: lets an already-assumed session re-assume this same role.
        # Chained sessions (OIDC -> github -> this role) are hard-capped at 1h by
        # AWS regardless of max_session_duration above, so long GHA sweeps
        # proactively re-assume every 45min (see sweep.py CredentialRefresher) —
        # without this statement that re-assumption is denied and the sweep dies.
        # ARN is constructed (not self-referenced) because Terraform disallows a
        # resource referring to its own attribute within its own configuration.
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${var.account_id}:role/${local.prefix}-terraform-deployment" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "terraform_deployment" {
  role       = aws_iam_role.terraform_deployment.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

terraform {
  # Local state intentionally — this config creates the S3 bucket
  # that all other modules use as their backend. Cannot bootstrap
  # from S3 before the bucket exists.
  required_providers {
    aws   = { source = "hashicorp/aws", version = "~> 5.0" }
    local = { source = "hashicorp/local", version = "~> 2.0" }
  }
}

provider "aws" {
  region  = "eu-central-1"
  profile = "wne-uw"
}

module "backend" {
  source = "../modules/backend"

  usecase            = "kg-experiments"
  environment        = "dev"
  aws_region         = "eu-central-1"
  account_id         = "039293892587"
  github_repo        = "jjfrackowiak/finance-kg-builder"
  local_operator_arn = "arn:aws:iam::039293892587:user/jj.frackowiak2@uw.edu.pl"
  tags               = {}
}

# ── Generated: backend config for all other Terraform roots ─────────────────

resource "local_file" "backend_tfvars" {
  filename        = "${path.module}/generated/backend.tfvars"
  file_permission = "0600"

  content = <<-EOT
    bucket       = "${module.backend.bucket}"
    use_lockfile = true
    assume_role  = {
      role_arn     = "${module.backend.terraform_deployment_role_arn}"
      session_name = "terraform"
    }
    key     = "terraform.tfstate"
    region  = "${module.backend.region}"
    encrypt = true
  EOT
}

# ── Generated: CI/CD vars for GitHub Actions workflow ───────────────────────

resource "local_file" "plan_apply_tfvars" {
  filename        = "${path.module}/generated/plan_apply.tfvars"
  file_permission = "0600"

  content = <<-EOT
    github_role_arn = "${module.backend.github_role_arn}"
    role_arn        = "${module.backend.terraform_deployment_role_arn}"
    environment     = "${module.backend.environment}"
    usecase         = "${module.backend.usecase}"
    region          = "${module.backend.region}"
    prefix          = "${module.backend.prefix}"
    tags            = ${jsonencode(module.backend.tags)}
  EOT
}

terraform {
  # Backend config supplied at init time via:
  # terraform init -backend-config=../../backend/dev/generated/backend.tfvars
  backend "s3" {}

  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = "eu-central-1"
  assume_role {
    role_arn     = var.role_arn
    session_name = "terraform"
  }
}

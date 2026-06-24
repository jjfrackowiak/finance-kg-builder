output "bucket" {
  value       = aws_s3_bucket.tfstate.bucket
  description = "Terraform state S3 bucket name"
}

output "region" {
  value       = var.aws_region
  description = "AWS region"
}

output "terraform_deployment_role_arn" {
  value       = aws_iam_role.terraform_deployment.arn
  description = "IAM role ARN Terraform assumes for all deployments"
}

output "github_role_arn" {
  value       = aws_iam_role.github.arn
  description = "IAM role ARN assumed by GitHub Actions via OIDC"
}

output "environment" {
  value = var.environment
}

output "usecase" {
  value = var.usecase
}

output "prefix" {
  value       = local.prefix
  description = "Resource name prefix: <usecase>-<environment>"
}

output "tags" {
  value = local.tags
}

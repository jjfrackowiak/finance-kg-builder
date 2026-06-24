output "terraform_deployment_role_arn" {
  value       = module.backend.terraform_deployment_role_arn
  description = "IAM role ARN Terraform assumes for all deployments"
}

output "github_role_arn" {
  value       = module.backend.github_role_arn
  description = "IAM role ARN assumed by GitHub Actions via OIDC"
}

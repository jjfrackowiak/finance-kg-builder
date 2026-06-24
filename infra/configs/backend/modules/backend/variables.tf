variable "usecase" {
  type        = string
  description = "Short project slug — used in resource names and prefix"
}

variable "environment" {
  type        = string
  description = "Deployment environment"
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be dev or prod"
  }
}

variable "aws_region" {
  type    = string
  default = "eu-central-1"
}

variable "account_id" {
  type        = string
  description = "AWS account ID — appended to bucket names for global uniqueness"
}

variable "github_repo" {
  type        = string
  description = "GitHub repo in org/repo format — scopes the OIDC trust"
}

variable "local_operator_arn" {
  type        = string
  description = "IAM user ARN allowed to assume the terraform deployment role locally"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Additional tags merged onto all resources"
}

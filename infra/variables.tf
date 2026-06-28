variable "role_arn" {
  type        = string
  description = "IAM role Terraform assumes — injected from plan_apply.tfvars"
}

variable "prefix" {
  type        = string
  description = "Resource name prefix"
  default     = "kg-experiments-dev"
}

variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "tags" {
  type    = map(string)
  default = {}
}

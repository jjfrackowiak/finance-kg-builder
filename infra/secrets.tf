# ── Secrets Manager — kg credentials ─────────────────────────────────────────
# Creates the secret container only. Set the value manually after apply:
#
#   aws secretsmanager put-secret-value \
#     --secret-id kg-experiments-dev/kg-credentials \
#     --secret-string '{
#       "NEO4J_URI":             "bolt+s://...",
#       "NEO4J_PASSWORD":        "...",
#       "MLFLOW_TRACKING_URI":   "https://dagshub.com/...",
#       "MLFLOW_TRACKING_TOKEN": "...",
#       "HF_TOKEN":              "hf_..."
#     }'

resource "aws_secretsmanager_secret" "kg_credentials" {
  name                    = "${var.prefix}/kg-credentials"
  description             = "KG Builder runtime credentials (Neo4j, MLflow, HuggingFace)"
  recovery_window_in_days = 0
  tags                    = var.tags
}

# ── IAM — allow deployment role to read the secret (for CI/CD bootstrap) ──────

resource "aws_iam_role_policy" "deployment_read_secret" {
  name = "${var.prefix}-read-kg-credentials"
  role = split("/", var.role_arn)[1]
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]
      Resource = aws_secretsmanager_secret.kg_credentials.arn
    }]
  })
}

output "kg_credentials_secret_name" {
  value = aws_secretsmanager_secret.kg_credentials.name
}

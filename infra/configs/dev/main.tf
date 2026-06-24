resource "aws_s3_bucket" "test" {
  bucket = "kg-experiments-dev-cicd-test-039293892587"
  tags = {
    Environment = "dev"
    ManagedBy   = "terraform"
    Purpose     = "cicd-smoke-test"
  }
}

resource "aws_s3_bucket_public_access_block" "test" {
  bucket                  = aws_s3_bucket.test.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

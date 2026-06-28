resource "aws_s3_bucket" "data" {
  bucket = "kg-experiments-data-039293892587"
  tags = {
    Environment = "dev"
    ManagedBy   = "terraform"
    Purpose     = "experiment-data"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

output "data_bucket" {
  value = aws_s3_bucket.data.bucket
}

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

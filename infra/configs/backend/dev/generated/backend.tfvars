bucket       = "kg-experiments-dev-tfstate-039293892587"
use_lockfile = true
assume_role  = {
  role_arn     = "arn:aws:iam::039293892587:role/kg-experiments-dev-terraform-deployment"
  session_name = "terraform"
}
key     = "terraform.tfstate"
region  = "eu-central-1"
encrypt = true

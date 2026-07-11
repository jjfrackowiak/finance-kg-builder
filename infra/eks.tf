# ── IAM — cluster ────────────────────────────────────────────────────────────

resource "aws_iam_role" "eks_cluster" {
  name = "${var.prefix}-eks-cluster"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# ── IAM — node group ─────────────────────────────────────────────────────────

resource "aws_iam_role" "eks_nodes" {
  name = "${var.prefix}-eks-nodes"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "eks_worker_node" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "eks_cni" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "eks_ecr" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy" "eks_nodes_s3_efs_bedrock" {
  name = "${var.prefix}-eks-nodes-s3-efs-bedrock"
  role = aws_iam_role.eks_nodes.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = [
          "elasticfilesystem:ClientMount",
          "elasticfilesystem:ClientWrite",
          "elasticfilesystem:DescribeMountTargets"
        ]
        Resource = aws_efs_file_system.model_cache.arn
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resource = "arn:aws:bedrock:${var.region}::foundation-model/qwen.qwen3-32b-v1:0"
      }
    ]
  })
}

# ── Security group — cluster ──────────────────────────────────────────────────

resource "aws_security_group" "eks_cluster" {
  name   = "${var.prefix}-eks-cluster"
  vpc_id = aws_vpc.main.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(var.tags, { Name = "${var.prefix}-eks-cluster" })
}

# ── EKS cluster ───────────────────────────────────────────────────────────────

resource "aws_eks_cluster" "main" {
  name     = "${var.prefix}-eks"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.33"

  vpc_config {
    subnet_ids              = concat(aws_subnet.private[*].id, aws_subnet.public[*].id)
    security_group_ids      = [aws_security_group.eks_cluster.id]
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  access_config {
    authentication_mode                         = "API_AND_CONFIG_MAP"
    bootstrap_cluster_creator_admin_permissions = true
  }

  depends_on = [aws_iam_role_policy_attachment.eks_cluster]
  tags       = var.tags
}

# ── OIDC provider (for IRSA) ─────────────────────────────────────────────────

data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer
  tags            = var.tags
}

# ── Node group — CPU (kg-builder, sweep, embeddings) ─────────────────────────

# AL2023 nodes default to IMDS hop limit 1, which blocks containers from reaching
# the node IAM role. Hop limit 2 allows pods to use node credentials via IMDS.
resource "aws_launch_template" "cpu" {
  name_prefix = "${var.prefix}-cpu-"

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
}

resource "aws_eks_node_group" "cpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.prefix}-cpu"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = ["t3.medium"]

  launch_template {
    id      = aws_launch_template.cpu.id
    version = aws_launch_template.cpu.latest_version
  }

  scaling_config {
    desired_size = 1
    min_size     = 1
    max_size     = 3
  }

  labels = { node-role = "cpu" }
  tags   = var.tags

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node,
    aws_iam_role_policy_attachment.eks_cni,
    aws_iam_role_policy_attachment.eks_ecr,
  ]
}

# ── Node group — GPU (vLLM) ───────────────────────────────────────────────────

resource "aws_launch_template" "gpu" {
  name_prefix = "${var.prefix}-gpu-"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }
}

resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.prefix}-gpu"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id
  instance_types  = ["g5.xlarge"]
  ami_type        = "AL2023_x86_64_NVIDIA"

  launch_template {
    id      = aws_launch_template.gpu.id
    version = aws_launch_template.gpu.latest_version
  }

  scaling_config {
    desired_size = 0
    min_size     = 0
    max_size     = 10
  }

  labels = { node-role = "gpu" }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = merge(var.tags, {
    "k8s.io/cluster-autoscaler/enabled"           = "true"
    "k8s.io/cluster-autoscaler/${var.prefix}-eks" = "owned"
  })

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node,
    aws_iam_role_policy_attachment.eks_cni,
    aws_iam_role_policy_attachment.eks_ecr,
    aws_eks_addon.vpc_cni,
  ]
}

# Keep 1 stopped g5.xlarge in a warm pool so scale-up takes ~2 min instead of ~12 min.
# Stopped instances incur only EBS storage cost (~$1-3/month).
resource "aws_autoscaling_warm_pool" "gpu" {
  autoscaling_group_name = aws_eks_node_group.gpu.resources[0].autoscaling_groups[0].name

  min_size = 1

  pool_state = "Stopped"

  instance_reuse_policy {
    reuse_on_scale_in = true
  }
}

# ── EKS add-ons ───────────────────────────────────────────────────────────────

resource "aws_eks_addon" "vpc_cni" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "vpc-cni"
}

resource "aws_eks_addon" "coredns" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "coredns"
  depends_on   = [aws_eks_node_group.cpu]
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name = aws_eks_cluster.main.name
  addon_name   = "kube-proxy"
}

resource "aws_eks_addon" "efs_csi" {
  cluster_name             = aws_eks_cluster.main.name
  addon_name               = "aws-efs-csi-driver"
  service_account_role_arn = aws_iam_role.efs_csi.arn
  depends_on = [
    aws_eks_node_group.cpu,
    aws_iam_role_policy_attachment.efs_csi,
  ]
}

# ── IRSA — EFS CSI driver ─────────────────────────────────────────────────────

resource "aws_iam_role" "efs_csi" {
  name = "${var.prefix}-efs-csi"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub" = "system:serviceaccount:kube-system:efs-csi-controller-sa"
          "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "efs_csi" {
  role       = aws_iam_role.efs_csi.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy"
}

# ── IRSA — kg-builder (sweep-runner SA) ──────────────────────────────────────

resource "aws_iam_role" "kg_builder" {
  name = "${var.prefix}-kg-builder"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub" = "system:serviceaccount:kg-experiments:sweep-runner"
          "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
  tags = var.tags
}

resource "aws_iam_role_policy" "kg_builder_s3" {
  name = "${var.prefix}-kg-builder-s3"
  role = aws_iam_role.kg_builder.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*",
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resource = "arn:aws:bedrock:${var.region}::foundation-model/qwen.qwen3-32b-v1:0"
      },
    ]
  })
}

# ── outputs ───────────────────────────────────────────────────────────────────

output "eks_cluster_name" {
  value = aws_eks_cluster.main.name
}

output "eks_cluster_endpoint" {
  value = aws_eks_cluster.main.endpoint
}

output "kubeconfig_command" {
  value = "aws eks update-kubeconfig --name ${aws_eks_cluster.main.name} --region ${var.region}"
}

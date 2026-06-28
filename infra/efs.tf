resource "aws_security_group" "efs" {
  name   = "${var.prefix}-efs"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = local.private_subnets
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.prefix}-efs" })
}

resource "aws_efs_file_system" "model_cache" {
  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"
  tags             = merge(var.tags, { Name = "${var.prefix}-model-cache" })
}

resource "aws_efs_mount_target" "model_cache" {
  count           = length(aws_subnet.private)
  file_system_id  = aws_efs_file_system.model_cache.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
  depends_on      = [aws_eks_cluster.main]
}

output "efs_id" {
  value = aws_efs_file_system.model_cache.id
}

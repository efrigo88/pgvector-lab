#!/bin/bash

# Exit on error
set -e

# Source environment variables
source .env

# Export AWS credentials from .env
export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION

# Initialize and apply Terraform
cd infra
terraform init
terraform apply -auto-approve

# Configuration
AWS_REGION="eu-west-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="data-pipeline"
IMAGE_TAG="latest"

echo "🚀 Starting deployment process..."
echo "Using AWS Region: ${AWS_REGION}"
echo "Using AWS Account ID: ${AWS_ACCOUNT_ID}"

# Get the bucket name and EC2 IP from Terraform outputs
BUCKET_NAME=$(terraform output -raw bucket_name 2>/dev/null || echo "BUCKET_NAME_NOT_AVAILABLE")
EC2_IP=$(terraform output -raw ec2_public_ip 2>/dev/null || echo "EC2_IP_NOT_AVAILABLE")
cd ..

# Download and set up SSH key
echo "🔑 Downloading SSH key..."
aws s3 cp s3://${BUCKET_NAME}/ssh/docker-pipeline-ml-ec2-lab-key.pem ./key.pem
chmod 400 ./key.pem

echo "To connect to the EC2 instance:"
echo "   ssh -i key.pem ubuntu@${EC2_IP}"
echo ""
echo "Note: The EC2 instance may take a few minutes to fully initialize."

# Build Docker image
echo "📦 Building Docker image..."
docker buildx build --platform linux/amd64 -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

# Login to ECR
echo "🔑 Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Tag and push Docker image
echo "🏷️  Tagging and pushing Docker image..."
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}

echo "✅ Deployment completed successfully!"
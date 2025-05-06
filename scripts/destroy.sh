#!/bin/bash

# Exit on error
set -e

# Source environment variables
if [ -f .env ]; then
    echo "📝 Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "❌ .env file not found!"
    exit 1
fi

# Configuration
AWS_REGION="eu-west-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPOSITORY="data-pipeline"

echo "🚀 Starting cleanup process..."

# Delete all images from ECR to allow repository deletion
echo "🧹 Cleaning up ECR repository..."
aws ecr batch-delete-image \
    --repository-name ${ECR_REPOSITORY} \
    --image-ids imageTag=latest \
    --region ${AWS_REGION} || true  # Ignore errors if repository doesn't exist

# Delete all files in S3 bucket
echo "🧹 Deleting all files in S3 bucket..."
BUCKET_NAME=$(aws s3 ls | grep "docker-pipeline-ml-ec2-lab" | awk '{print $3}')
if [ -z "$BUCKET_NAME" ]; then
    echo "⚠️ No matching S3 bucket found, skipping bucket cleanup"
else
    echo "Found bucket: ${BUCKET_NAME}"
    aws s3 rm "s3://${BUCKET_NAME}" --recursive || {
        echo "❌ Failed to delete contents of bucket ${BUCKET_NAME}"
        exit 1
    }
    echo "✅ Successfully deleted all files from bucket ${BUCKET_NAME}"
fi

# Change to infra directory
cd infra

# Initialize Terraform if needed
echo "🔧 Initializing Terraform..."
terraform init

# Destroy all infrastructure
echo "🗑️  Destroying all infrastructure..."
terraform destroy -auto-approve

echo "🧹 Cleaning up local Terraform state files..."
find . -type d -name ".terraform" -exec rm -rf {} +
find . -type d -name "terraform.tfstate.d" -exec rm -rf {} +
find . -type f \( \
    -name ".terraform.lock.hcl" \
    -o -name ".terraform.tfstate.lock.info" \
    -o -name "terraform.tfstate.backup" \
    -o -name "terraform.tfstate" \
    -o -name "myplan" \
\) -exec rm -f {} +

# Change back to root directory
cd ..

# Delete key.pem
echo "🧹 Deleting key.pem..."
rm -rf key.pem

echo "✅ Cleanup completed successfully!"
echo "All AWS resources and local Terraform state files have been destroyed."

#!/bin/bash

echo "Downloading NLLB translation model from S3..."

# Create cache directory
mkdir -p ~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M

# Download model from S3
aws s3 cp s3://finetuning-demo-models/nllb-200-distilled-600M/ ~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/ --recursive

echo "Download complete!"
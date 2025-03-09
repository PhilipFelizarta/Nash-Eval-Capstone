#!/bin/bash

# Set Vast.ai details
PORT=19648
IP=175.155.64.221
SSH_KEY="$HOME/.ssh/vastai_key"
LOCAL_PATH="/z/Capstone/Epsilon-Explore-Capstone"
REMOTE_PATH="/root/chess_project"

echo "🚀 Uploading files to Vast.ai..."

# Upload files and folders using SCP
scp -i "$SSH_KEY" -P "$PORT" \
    "$LOCAL_PATH/requirements.txt" \
    "$LOCAL_PATH/data/LumbrasGigaBase 2024.pgn.zst" \
    "$LOCAL_PATH/Dockerfile.pretrain" \
    "$LOCAL_PATH/docker-compose-pretrain.yml" \
    root@$IP:$REMOTE_PATH/

# Upload directories separately
scp -r -i "$SSH_KEY" -P "$PORT" "$LOCAL_PATH/core" root@$IP:$REMOTE_PATH/
scp -r -i "$SSH_KEY" -P "$PORT" "$LOCAL_PATH/pretraining" root@$IP:$REMOTE_PATH/

echo "✅ Files successfully uploaded to Vast.ai!"
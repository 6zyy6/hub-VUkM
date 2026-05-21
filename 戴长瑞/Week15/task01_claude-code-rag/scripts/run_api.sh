#!/bin/bash

# 启动脚本

# 1. 检查依赖
echo "Checking dependencies..."
python -c "import fastapi, pymilvus, kafka" 2>/dev/null && echo "Dependencies OK" || echo "Install dependencies: pip install -r requirements.txt"

# 2. 检查环境变量
if [ ! -f .env ]; then
    echo "Warning: .env not found. Copy .env.example to .env"
fi

# 3. 启动 FastAPI
echo "Starting FastAPI..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
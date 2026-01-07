#!/bin/bash
# install.sh - Полная установка RAG системы

set -e

echo "=== RAG System Installation ==="

# 1. Системные пакеты
echo "[1/7] Installing system dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3.11 python3.11-venv python3.11-dev \
    postgresql-15 postgresql-server-dev-15 \
    build-essential git curl \
    nginx

# 2. NVIDIA drivers (проверка)
if ! command -v nvidia-smi &> /dev/null; then
    echo "[2/7] WARNING: NVIDIA drivers not found. GPU features will be unavailable."
    echo "Please install drivers manually for your hardware."
fi

# 3. pgvector
echo "[3/7] Installing pgvector..."
cd /tmp
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# 4. PostgreSQL setup
echo "[4/7] Setting up PostgreSQL..."
sudo -u postgres psql -c "CREATE USER rag_user WITH PASSWORD 'changeme';" || echo "User rag_user already exists."
sudo -u postgres psql -c "CREATE DATABASE rag_docs OWNER rag_user;" || echo "Database rag_docs already exists."
sudo -u postgres psql -d rag_docs -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 5. Создание пользователя для сервисов
sudo useradd -r -s /bin/bash -d /opt/rag-system rag || echo "User rag already exists."
sudo mkdir -p /opt/rag-system
sudo chown -R rag:rag /opt/rag-system

# 6. Установка проекта
echo "[5/7] Installing RAG system..."
# Предполагается, что скрипт запускается из корня репозитория
sudo rsync -a . /opt/rag-system/ --exclude '.git' --exclude 'venv'
cd /opt/rag-system

# 7. Python environment
echo "[6/7] Setting up Python environment..."
sudo -u rag python3.11 -m venv venv
sudo -u rag ./venv/bin/pip install --upgrade pip wheel
sudo -u rag ./venv/bin/pip install -e ".[dev]"

# 8. Конфигурация
echo "[7/7] Finalizing configuration..."
sudo -u rag cp .env.example .env
echo "INFO: .env file created. Please edit /opt/rag-system/.env with your configuration."

echo "=== Installation complete ==="
echo "Next steps:"
echo "1. Edit /opt/rag-system/.env with your settings."
echo "2. Run 'sudo -u rag /opt/rag-system/venv/bin/python /opt/rag-system/scripts/download_models.py' to cache ML models."
echo "3. Run 'sudo -u rag psql -U rag_user -d rag_docs -f /opt/rag-system/scripts/setup_db.sql' to initialize the DB schema."
echo "4. Setup and enable systemd services and nginx."
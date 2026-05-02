#!/bin/bash
# Deployment script for AWS EC2
# Usage: ./deploy.sh

set -e

echo "🚀 Inventory Quantum Deployment"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    sudo yum update -y || sudo apt-get update -y
    sudo yum install -y docker || sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "✅ Docker installed"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your AWS credentials"
fi

# Build and run
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting application..."
docker-compose up -d

echo ""
echo "✅ Deployment complete!"
echo "📍 Application running at: http://$(curl -s ifconfig.me):8000"
echo "📊 Dashboard at: http://$(curl -s ifconfig.me):8000/static/index.html"
echo ""
echo "📝 Useful commands:"
echo "   docker-compose logs -f    # View logs"
echo "   docker-compose down       # Stop application"
echo "   docker-compose restart    # Restart application"

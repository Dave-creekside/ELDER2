#!/bin/bash

# ELDER Mind Installation Script
# This script sets up everything needed to run Elder

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        ELDER Mind Installation         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

echo "🔍 Detected: $OS ($ARCH)"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Docker installation helper
install_docker() {
    echo ""
    echo -e "${YELLOW}📦 Docker is required to run Elder's databases${NC}"
    
    if [[ "$OS" == "Darwin" ]]; then
        # macOS
        echo "📱 macOS detected"
        
        if [[ "$ARCH" == "arm64" ]]; then
            echo "🍎 Apple Silicon (M1/M2/M3) detected"
            DOCKER_URL="https://desktop.docker.com/mac/main/arm64/Docker.dmg"
        else
            echo "🖥️  Intel Mac detected"
            DOCKER_URL="https://desktop.docker.com/mac/main/amd64/Docker.dmg"
        fi
        
        echo ""
        echo "Options:"
        echo "1) Download Docker Desktop automatically"
        echo "2) Open Docker website in browser"
        echo "3) Skip (I'll install Docker myself)"
        echo ""
        read -p "Choose an option (1-3): " choice
        
        case $choice in
            1)
                echo "📥 Downloading Docker Desktop..."
                curl -L -o ~/Downloads/Docker.dmg "$DOCKER_URL"
                echo "✅ Downloaded to ~/Downloads/Docker.dmg"
                echo ""
                echo "📝 Next steps:"
                echo "1. Open ~/Downloads/Docker.dmg"
                echo "2. Drag Docker to Applications"
                echo "3. Launch Docker from Applications"
                echo "4. Wait for Docker to start (whale icon in menu bar)"
                echo "5. Run this install script again"
                echo ""
                read -p "Press Enter to open the Downloads folder..."
                open ~/Downloads/
                exit 0
                ;;
            2)
                echo "🌐 Opening Docker website..."
                open "https://www.docker.com/products/docker-desktop/"
                echo ""
                echo "Please download and install Docker, then run this script again."
                exit 0
                ;;
            3)
                echo "⚠️  Skipping Docker installation"
                echo "Please install Docker manually and run this script again."
                exit 1
                ;;
        esac
        
    elif [[ "$OS" == "Linux" ]]; then
        # Linux
        echo "🐧 Linux detected"
        echo ""
        echo "To install Docker on Linux, run:"
        echo ""
        echo "curl -fsSL https://get.docker.com | sh"
        echo "sudo usermod -aG docker $USER"
        echo ""
        echo "Then log out and back in, and run this script again."
        exit 1
    fi
}

# Check for Docker
echo ""
echo "🐳 Checking for Docker..."
if ! command_exists docker; then
    echo -e "${RED}❌ Docker not found${NC}"
    install_docker
else
    echo -e "${GREEN}✅ Docker is installed${NC}"
    
    # Check if Docker is running
    if ! docker ps >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Docker is installed but not running${NC}"
        
        if [[ "$OS" == "Darwin" ]]; then
            echo "Starting Docker..."
            open -a Docker
            echo "Waiting for Docker to start..."
            
            # Wait up to 30 seconds for Docker to start
            for i in {1..30}; do
                if docker ps >/dev/null 2>&1; then
                    echo -e "${GREEN}✅ Docker is now running${NC}"
                    break
                fi
                sleep 1
                echo -n "."
            done
            
            if ! docker ps >/dev/null 2>&1; then
                echo -e "${RED}❌ Docker failed to start${NC}"
                echo "Please start Docker manually and run this script again."
                exit 1
            fi
        else
            echo "Please start Docker and run this script again."
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Docker is running${NC}"
    fi
fi

# Check for Python 3
echo ""
echo "🐍 Checking Python..."
if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
else
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
fi

# Create virtual environment
echo ""
echo "🔧 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip >/dev/null 2>&1

# Install requirements
echo ""
echo "📦 Installing Python dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ] && [ ! -f "streamlined_consciousness/.env" ]; then
    echo "📝 Creating configuration files..."
    cp .env.example .env
    cp .env.example streamlined_consciousness/.env
    echo -e "${YELLOW}⚠️  Please edit .env to add your API keys${NC}"
else
    echo "✅ Configuration files already exist"
fi

# Use docker compose or docker-compose
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Start Docker containers
echo ""
echo "🚀 Starting database containers..."
$DOCKER_COMPOSE up -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for databases to start..."
echo -n "Neo4j"
for i in {1..30}; do
    if curl -s http://localhost:7474 >/dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

echo -n "Qdrant"
for i in {1..30}; do
    if curl -s http://localhost:6333/health >/dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Initialize databases
echo ""
echo "🧠 Initializing Elder's mind..."
./nuke_neo4j.sh

# Success message
echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     ✅ Installation Complete! ✅       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "To start Elder Mind, run:"
echo -e "${BLUE}./start.sh${NC}"
echo ""

# Check if using Ollama
if grep -q "LLM_PROVIDER=ollama" .env 2>/dev/null; then
    echo -e "${YELLOW}📝 Note: You're using Ollama${NC}"
    echo "Make sure Ollama is running and has a compatible model pulled"
    echo "Recommended models: mistral, mixtral, llama3.2"
fi

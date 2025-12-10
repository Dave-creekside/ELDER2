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

# Check for Python 3.10+
echo ""
echo "🐍 Checking Python..."

# Function to check python version
check_python_version() {
    local cmd=$1
    if command_exists "$cmd"; then
        local ver=$($cmd -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        local major=$(echo "$ver" | cut -d. -f1)
        local minor=$(echo "$ver" | cut -d. -f2)
        
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            echo "$cmd"
            return 0
        fi
    fi
    return 1
}

PYTHON_CMD=""

# Check python3 first
if check_python_version "python3" > /dev/null; then
    PYTHON_CMD="python3"
# Check specific versions
elif check_python_version "python3.11" > /dev/null; then
    PYTHON_CMD="python3.11"
elif check_python_version "python3.12" > /dev/null; then
    PYTHON_CMD="python3.12"
elif check_python_version "python3.10" > /dev/null; then
    PYTHON_CMD="python3.10"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}❌ Python 3.10+ not found${NC}"
    echo "The 'mcp' library requires Python 3.10 or higher."
    
    if [[ "$OS" == "Darwin" ]]; then
        echo ""
        echo "💡 Recommended: Install Python 3.11 via Homebrew"
        if command_exists brew; then
            echo "Run: brew install python@3.11"
            echo ""
            read -p "Would you like to try installing it now? (y/n) " choice
            if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
                brew install python@3.11
                PYTHON_CMD="python3.11"
            fi
        else
            echo "Please install Homebrew first: https://brew.sh"
        fi
    fi
    
    if [ -z "$PYTHON_CMD" ]; then
        echo "Please install Python 3.10+ and run this script again."
        exit 1
    fi
fi

VER=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✅ Using $PYTHON_CMD ($VER)${NC}"

# Create virtual environment
echo ""
echo "🔧 Setting up Python virtual environment..."

# Check if existing venv is compatible
if [ -d "venv" ]; then
    if [ -f "venv/bin/python" ]; then
        VENV_VER=$(venv/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        VENV_MAJOR=$(echo "$VENV_VER" | cut -d. -f1)
        VENV_MINOR=$(echo "$VENV_VER" | cut -d. -f2)
        
        if [ "$VENV_MAJOR" -eq 3 ] && [ "$VENV_MINOR" -ge 10 ]; then
            echo "✅ Virtual environment already exists and is compatible ($VENV_VER)"
        else
            echo -e "${YELLOW}⚠️  Existing virtual environment is too old ($VENV_VER)${NC}"
            echo "Recreating virtual environment..."
            rm -rf venv
            $PYTHON_CMD -m venv venv
            echo -e "${GREEN}✅ Virtual environment created${NC}"
        fi
    else
        echo "⚠️  Broken virtual environment detected. Recreating..."
        rm -rf venv
        $PYTHON_CMD -m venv venv
        echo -e "${GREEN}✅ Virtual environment created${NC}"
    fi
else
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
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

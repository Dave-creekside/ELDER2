#!/bin/bash

# ELDER Mind Startup Script

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🧠 Starting ELDER Mind...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please run ./install.sh first"
    exit 1
fi

# Check if .env file exists
if [ ! -f "streamlined_consciousness/.env" ]; then
    echo -e "${RED}❌ Configuration file not found!${NC}"
    echo "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Use docker compose or docker-compose
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Start Docker containers if not running
echo "🗄️ Checking databases..."
if ! docker ps | grep -q elder-neo4j; then
    echo "Starting Neo4j..."
    $DOCKER_COMPOSE up -d neo4j
fi

if ! docker ps | grep -q elder-qdrant; then
    echo "Starting Qdrant..."
    $DOCKER_COMPOSE up -d qdrant
fi

# Wait for services to be ready
echo "⏳ Waiting for services..."
MAX_ATTEMPTS=30
ATTEMPT=0

# Wait for Neo4j
while ! curl -s http://localhost:7474 > /dev/null 2>&1; do
    sleep 1
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo -e "${RED}❌ Neo4j failed to start${NC}"
        echo "Check logs with: docker logs elder-neo4j"
        exit 1
    fi
done
echo -e "${GREEN}✅ Neo4j is ready${NC}"

# Wait for Qdrant
ATTEMPT=0
while ! curl -s http://localhost:6333/health > /dev/null 2>&1; do
    sleep 1
    ATTEMPT=$((ATTEMPT + 1))
    if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
        echo -e "${RED}❌ Qdrant failed to start${NC}"
        echo "Check logs with: docker logs elder-qdrant"
        exit 1
    fi
done
echo -e "${GREEN}✅ Qdrant is ready${NC}"

# Check if API key is configured
if grep -q "your_anthropic_api_key_here" streamlined_consciousness/.env 2>/dev/null || \
   grep -q "your_openai_api_key_here" streamlined_consciousness/.env 2>/dev/null || \
   grep -q "your_gemini_api_key_here" streamlined_consciousness/.env 2>/dev/null || \
   grep -q "your_groq_api_key_here" streamlined_consciousness/.env 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: API keys may not be configured!${NC}"
    echo "Please check streamlined_consciousness/.env and add your API keys"
    echo ""
fi

# Start Elder with dashboard
echo ""
echo -e "${GREEN}✨ Elder Mind is awakening...${NC}"
echo ""
echo "🌐 Dashboard will be available at: http://localhost:5000"
echo "📊 System health: http://localhost:5000/health"
echo ""
echo "Commands:"
echo "  'chat' - Enter conversation mode"
echo "  'dream' - Enter dream state"
echo "  'status' - Check system status"
echo "  'clear' - Clear conversation history"
echo "  'exit' or 'quit' - Shutdown Elder"
echo ""

# Run the main program with dashboard
python streamlined_consciousness/run_with_dashboard.py

# Deactivate virtual environment when done
deactivate

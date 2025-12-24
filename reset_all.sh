#!/bin/bash

echo "☢️  INITIATING TOTAL SYSTEM RESET ☢️"
echo "This will wipe all databases, memories, and learned weights."
read -p "Are you sure? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Reset cancelled."
    exit 1
fi

echo ""
echo "1. Nuke Databases (Neo4j & Qdrant)..."
chmod +x nuke_neo4j.sh
./nuke_neo4j.sh

echo ""
echo "2. Clearing Learned Weights..."
if [ -d "adapters" ]; then
    rm -rf adapters
    echo "✅ Deleted adapters/ directory"
else
    echo "ℹ️  No adapters found to delete"
fi

echo ""
echo "3. Re-initializing Riemannian Infrastructure..."
./venv/bin/python initialize_riemannian.py

echo ""
echo "✨ SYSTEM RESET COMPLETE ✨"
echo "You can now run ./start.sh to begin a fresh lifecycle."

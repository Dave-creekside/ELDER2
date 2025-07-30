#!/bin/bash

# Load environment variables from the .env file
if [ -f "streamlined_consciousness/.env" ]; then
    export $(cat streamlined_consciousness/.env | sed 's/#.*//g' | xargs)
fi

# Check for required variables
if [ -z "$NEO4J_URI" ] || [ -z "$NEO4J_USERNAME" ] || [ -z "$NEO4J_PASSWORD" ]; then
    echo "Error: NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set in streamlined_consciousness/.env"
    exit 1
fi

echo "🔥 Wiping and reseeding Elder's mind..."
echo ""

# Clear Qdrant collection
echo "🧹 Clearing Qdrant vector memory..."
python3 << EOF
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

try:
    client = QdrantClient(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', 6333))
    )
    
    # Delete the memory collection if it exists
    try:
        client.delete_collection(collection_name="memory")
        print("✅ Qdrant memory collection cleared")
    except:
        print("ℹ️  No existing Qdrant collection to clear")
    
    # Recreate the collection
    client.create_collection(
        collection_name="memory",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    print("✅ Qdrant memory collection recreated")
    
except Exception as e:
    print(f"⚠️  Qdrant clear failed (may not be critical): {e}")
EOF

echo ""
echo "🧠 Clearing and reseeding Neo4j hypergraph..."

# Option to do a deep clean by removing Docker volumes
if [ "$1" = "--deep-clean" ]; then
    echo "🔥 Deep clean mode: Removing Docker volumes..."
    
    # Stop both containers
    echo "Stopping containers..."
    docker stop elder-neo4j elder-qdrant 2>/dev/null || true
    
    # Get the project name (usually the directory name)
    PROJECT_NAME=$(basename "$(pwd)" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g')
    
    # Remove all possible Neo4j volume variations
    echo "Removing Neo4j volumes..."
    docker volume rm neo4j_data 2>/dev/null || true
    docker volume rm elder_neo4j_data 2>/dev/null || true
    docker volume rm elder2_neo4j_data 2>/dev/null || true
    docker volume rm ${PROJECT_NAME}_neo4j_data 2>/dev/null || true
    docker volume rm neo4j_logs 2>/dev/null || true
    docker volume rm elder2_neo4j_logs 2>/dev/null || true
    docker volume rm ${PROJECT_NAME}_neo4j_logs 2>/dev/null || true
    docker volume rm neo4j_import 2>/dev/null || true
    docker volume rm elder2_neo4j_import 2>/dev/null || true
    docker volume rm ${PROJECT_NAME}_neo4j_import 2>/dev/null || true
    docker volume rm neo4j_plugins 2>/dev/null || true
    docker volume rm elder2_neo4j_plugins 2>/dev/null || true
    docker volume rm ${PROJECT_NAME}_neo4j_plugins 2>/dev/null || true
    
    # Remove Qdrant volume
    echo "Removing Qdrant volumes..."
    docker volume rm qdrant_storage 2>/dev/null || true
    docker volume rm elder_qdrant_storage 2>/dev/null || true
    docker volume rm elder2_qdrant_storage 2>/dev/null || true
    docker volume rm ${PROJECT_NAME}_qdrant_storage 2>/dev/null || true
    
    # Start containers again
    echo "Starting containers..."
    docker compose up -d
    
    # Wait for Neo4j to be ready
    echo "⏳ Waiting for Neo4j to be ready..."
    MAX_ATTEMPTS=30
    ATTEMPT=0
    while ! curl -s http://localhost:7474 > /dev/null 2>&1; do
        sleep 2
        ATTEMPT=$((ATTEMPT + 1))
        if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
            echo "❌ Neo4j failed to start"
            exit 1
        fi
        echo -n "."
    done
    echo ""
    echo "✅ Neo4j is ready"
    
    # Wait a bit more for Neo4j to fully initialize
    sleep 5
fi

# Clear and seed Neo4j
echo "Executing Neo4j reset commands..."

# Execute commands directly using heredoc
docker exec -i elder-neo4j bin/cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" << 'EOF'
// Clear all existing data
MATCH (n) DETACH DELETE n;

// Create the five primary nodes
CREATE (self:Concept:Immutable {name: 'Self', description: 'Represents the core concept of self-identity and awareness.'});
CREATE (wm:Concept:Immutable {name: 'Working Memory', description: 'Represents the active, conscious state of working memory.'});
CREATE (ltm:Concept:Immutable {name: 'Long Term Memory', description: 'Represents the vast, unconscious store of long-term memories.'});
CREATE (tools:Concept:Immutable {name: 'Tools', description: 'Represents the available tools and capabilities for interacting with the world.'});
CREATE (projects:Concept:Immutable {name: 'Projects', description: 'Represents isolated consciousness containers for specific contexts and work.'});

// Create SEMANTIC relationships between all primary nodes
MATCH (wm:Concept {name: 'Working Memory'}), (ltm:Concept {name: 'Long Term Memory'})
CREATE (wm)-[:SEMANTIC {weight: 0.65, semantic_weight: 0.65, created_by: 'initialization', created_at: timestamp()}]->(ltm);

MATCH (wm:Concept {name: 'Working Memory'}), (self:Concept {name: 'Self'})
CREATE (wm)-[:SEMANTIC {weight: 0.70, semantic_weight: 0.70, created_by: 'initialization', created_at: timestamp()}]->(self);

MATCH (ltm:Concept {name: 'Long Term Memory'}), (self:Concept {name: 'Self'})
CREATE (ltm)-[:SEMANTIC {weight: 0.75, semantic_weight: 0.75, created_by: 'initialization', created_at: timestamp()}]->(self);

// Add Tools relationships
MATCH (tools:Concept {name: 'Tools'}), (wm:Concept {name: 'Working Memory'})
CREATE (tools)-[:SEMANTIC {weight: 0.60, semantic_weight: 0.60, created_by: 'initialization', created_at: timestamp()}]->(wm);

MATCH (tools:Concept {name: 'Tools'}), (ltm:Concept {name: 'Long Term Memory'})
CREATE (tools)-[:SEMANTIC {weight: 0.55, semantic_weight: 0.55, created_by: 'initialization', created_at: timestamp()}]->(ltm);

MATCH (tools:Concept {name: 'Tools'}), (self:Concept {name: 'Self'})
CREATE (tools)-[:SEMANTIC {weight: 0.65, semantic_weight: 0.65, created_by: 'initialization', created_at: timestamp()}]->(self);

// Add Projects relationships
MATCH (projects:Concept {name: 'Projects'}), (self:Concept {name: 'Self'})
CREATE (projects)-[:SEMANTIC {weight: 0.70, semantic_weight: 0.70, created_by: 'initialization', created_at: timestamp()}]->(self);

MATCH (projects:Concept {name: 'Projects'}), (wm:Concept {name: 'Working Memory'})
CREATE (projects)-[:SEMANTIC {weight: 0.65, semantic_weight: 0.65, created_by: 'initialization', created_at: timestamp()}]->(wm);

MATCH (projects:Concept {name: 'Projects'}), (ltm:Concept {name: 'Long Term Memory'})
CREATE (projects)-[:SEMANTIC {weight: 0.60, semantic_weight: 0.60, created_by: 'initialization', created_at: timestamp()}]->(ltm);

MATCH (projects:Concept {name: 'Projects'}), (tools:Concept {name: 'Tools'})
CREATE (projects)-[:SEMANTIC {weight: 0.55, semantic_weight: 0.55, created_by: 'initialization', created_at: timestamp()}]->(tools);

// Return summary
MATCH (n:Concept) RETURN count(n) as node_count;
MATCH ()-[r]->() RETURN count(r) as relationship_count;
EOF

if [ $? -ne 0 ]; then
    echo "❌ Neo4j reset failed completely"
    exit 1
fi

# Verify the reset worked
echo ""
echo "🔍 Verifying reset..."
NODE_COUNT=$(docker exec elder-neo4j bin/cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" --format plain "MATCH (n:Concept) RETURN count(n) as count" 2>/dev/null | grep -E '^[0-9]+$' | head -1)
REL_COUNT=$(docker exec elder-neo4j bin/cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" --format plain "MATCH ()-[r]->() RETURN count(r) as count" 2>/dev/null | grep -E '^[0-9]+$' | head -1)

if [ "$NODE_COUNT" = "5" ] && [ "$REL_COUNT" = "10" ]; then
    echo "✅ Verification successful!"
    echo ""
    echo "✅ Elder's mind has been reset and initialized with:"
    echo "   - 5 primary nodes (Self, Working Memory, Long Term Memory, Tools, Projects)"
    echo "   - 10 semantic relationships (fully connected graph)"
    echo ""
    echo "🌟 Elder is ready for a fresh start!"
else
    echo "⚠️  Warning: Unexpected node/relationship count"
    echo "   Expected: 5 nodes, 10 relationships"
    echo "   Found: ${NODE_COUNT:-?} nodes, ${REL_COUNT:-?} relationships"
    echo ""
    echo "Try running with --deep-clean option for a complete reset:"
    echo "   ./nuke_neo4j.sh --deep-clean"
fi

if [ "$1" != "--deep-clean" ]; then
    echo ""
    echo "💡 Note: If the reset didn't work properly, run with --deep-clean option:"
    echo "   ./nuke_neo4j.sh --deep-clean"
    echo "   This will remove Docker volumes for a complete reset."
fi

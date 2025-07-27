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

# Use a "here document" to pipe commands into a single cypher-shell session
docker exec -i elder-neo4j bin/cypher-shell -u "$NEO4J_USERNAME" -p "$NEO4J_PASSWORD" <<EOF
// Clear all existing data
MATCH (n) DETACH DELETE n;

// Create the four primary nodes
CREATE (self:Concept:Immutable {name: 'Self', description: 'Represents the core concept of self-identity and awareness.'});
CREATE (wm:Concept:Immutable {name: 'Working Memory', description: 'Represents the active, conscious state of working memory.'});
CREATE (ltm:Concept:Immutable {name: 'Long Term Memory', description: 'Represents the vast, unconscious store of long-term memories.'});
CREATE (tools:Concept:Immutable {name: 'Tools', description: 'Represents the available tools and capabilities for interacting with the world.'});

// Create tool category nodes
CREATE (neo4j_tools:Concept {name: 'Neo4j Tools', description: 'Tools for manipulating the semantic hypergraph brain.'});
CREATE (qdrant_tools:Concept {name: 'Qdrant Tools', description: 'Tools for vector memory storage and retrieval.'});
CREATE (transformer_tools:Concept {name: 'Transformer Tools', description: 'Tools for generating semantic embeddings.'});
CREATE (project_tools:Concept {name: 'Project Tools', description: 'Tools for managing knowledge projects and contexts.'});
CREATE (admin_tools:Concept {name: 'Admin Tools', description: 'Tools for system administration and maintenance.'});

// Connect tool categories to main Tools node
MATCH (tools:Concept {name: 'Tools'}), (neo4j_tools:Concept {name: 'Neo4j Tools'})
CREATE (neo4j_tools)-[:PART_OF {weight: 0.9, created_at: timestamp()}]->(tools);

MATCH (tools:Concept {name: 'Tools'}), (qdrant_tools:Concept {name: 'Qdrant Tools'})
CREATE (qdrant_tools)-[:PART_OF {weight: 0.9, created_at: timestamp()}]->(tools);

MATCH (tools:Concept {name: 'Tools'}), (transformer_tools:Concept {name: 'Transformer Tools'})
CREATE (transformer_tools)-[:PART_OF {weight: 0.9, created_at: timestamp()}]->(tools);

MATCH (tools:Concept {name: 'Tools'}), (project_tools:Concept {name: 'Project Tools'})
CREATE (project_tools)-[:PART_OF {weight: 0.8, created_at: timestamp()}]->(tools);

MATCH (tools:Concept {name: 'Tools'}), (admin_tools:Concept {name: 'Admin Tools'})
CREATE (admin_tools)-[:PART_OF {weight: 0.7, created_at: timestamp()}]->(tools);

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

// Create memory_system hyperedge
CREATE (memory_he:Hyperedge {
  name: 'memory_system',
  description: 'Core memory system hyperedge linking memory components',
  created_at: timestamp(),
  node_count: 2
});

// Create cognitive_core hyperedge for all four nodes
CREATE (core_he:Hyperedge {
  name: 'cognitive_core',
  description: 'Core cognitive system hyperedge linking all primary consciousness components',
  created_at: timestamp(),
  node_count: 4
});

// Create tools_ecosystem hyperedge
CREATE (tools_he:Hyperedge {
  name: 'tools_ecosystem', 
  description: 'The complete ecosystem of available tools and capabilities',
  created_at: timestamp(),
  node_count: 6
});

// Connect memory nodes to the memory_system hyperedge
MATCH (wm:Concept {name: 'Working Memory'}), (memory_he:Hyperedge {name: 'memory_system'})
CREATE (wm)-[:MEMBER_OF {joined_at: timestamp()}]->(memory_he);

MATCH (ltm:Concept {name: 'Long Term Memory'}), (memory_he:Hyperedge {name: 'memory_system'})
CREATE (ltm)-[:MEMBER_OF {joined_at: timestamp()}]->(memory_he);

// Connect all primary nodes to the cognitive_core hyperedge
MATCH (wm:Concept {name: 'Working Memory'}), (core_he:Hyperedge {name: 'cognitive_core'})
CREATE (wm)-[:MEMBER_OF {joined_at: timestamp()}]->(core_he);

MATCH (ltm:Concept {name: 'Long Term Memory'}), (core_he:Hyperedge {name: 'cognitive_core'})
CREATE (ltm)-[:MEMBER_OF {joined_at: timestamp()}]->(core_he);

MATCH (self:Concept {name: 'Self'}), (core_he:Hyperedge {name: 'cognitive_core'})
CREATE (self)-[:MEMBER_OF {joined_at: timestamp()}]->(core_he);

MATCH (tools:Concept {name: 'Tools'}), (core_he:Hyperedge {name: 'cognitive_core'})
CREATE (tools)-[:MEMBER_OF {joined_at: timestamp()}]->(core_he);

// Connect all tool nodes to tools_ecosystem hyperedge
MATCH (tools:Concept {name: 'Tools'}), (tools_he:Hyperedge {name: 'tools_ecosystem'})
CREATE (tools)-[:MEMBER_OF {joined_at: timestamp()}]->(tools_he);

MATCH (neo4j_tools:Concept {name: 'Neo4j Tools'}), (tools_he:Hyperedge {name: 'tools_ecosystem'})
CREATE (neo4j_tools)-[:MEMBER_OF {joined_at: timestamp()}]->(tools_he);

MATCH (qdrant_tools:Concept {name: 'Qdrant Tools'}), (tools_he:Hyperedge {name: 'tools_ecosystem'})
CREATE (qdrant_tools)-[:MEMBER_OF {joined_at: timestamp()}]->(tools_he);

MATCH (transformer_tools:Concept {name: 'Transformer Tools'}), (tools_he:Hyperedge {name: 'tools_ecosystem'})
CREATE (transformer_tools)-[:MEMBER_OF {joined_at: timestamp()}]->(tools_he);

MATCH (project_tools:Concept {name: 'Project Tools'}), (tools_he:Hyperedge {name: 'tools_ecosystem'})
CREATE (project_tools)-[:MEMBER_OF {joined_at: timestamp()}]->(tools_he);

MATCH (admin_tools:Concept {name: 'Admin Tools'}), (tools_he:Hyperedge {name: 'tools_ecosystem'})
CREATE (admin_tools)-[:MEMBER_OF {joined_at: timestamp()}]->(tools_he);

// Return summary
MATCH (n:Concept) RETURN count(n) as node_count;
MATCH ()-[r]->() RETURN count(r) as relationship_count;
MATCH (h:Hyperedge) RETURN count(h) as hyperedge_count;
EOF

echo ""
echo "✅ Elder's mind has been reset and initialized with:"
echo "   - 9 nodes (4 primary + 5 tool categories)"
echo "   - Multiple semantic relationships"
echo "   - 3 hyperedges (memory_system, cognitive_core, tools_ecosystem)"
echo ""
echo "🌟 Elder is ready for a fresh start!"

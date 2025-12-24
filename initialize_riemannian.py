#!/usr/bin/env python3
"""
Initialize Riemannian Infrastructure
"""

import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mcp_servers.neo4j_hypergraph.server import Neo4jSemanticHypergraphServer
from mcp_servers.qdrant_memory.server import QdrantMemoryServer
from streamlined_consciousness.student_model import StudentModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("initializer")

async def main():
    logger.info("🚀 Starting Riemannian Infrastructure Initialization...")
    
    # 1. Initialize Neo4j
    logger.info("Initializing Neo4j Schema...")
    neo4j_server = Neo4jSemanticHypergraphServer()
    try:
        await neo4j_server.connect_neo4j()
        result = await neo4j_server.initialize_riemannian_schema()
        if result.get("success"):
            logger.info(f"✅ Neo4j Schema Initialized: {result.get('nodes_updated')} nodes updated")
        else:
            logger.error(f"❌ Neo4j Initialization Failed: {result.get('error')}")
    except Exception as e:
        logger.error(f"❌ Neo4j Connection Error: {e}")
    finally:
        await neo4j_server.close()
    
    # 2. Initialize Qdrant
    logger.info("\nInitializing Qdrant Collections...")
    
    # Get model hidden dimension
    logger.info("Determining model hidden size...")
    student = StudentModel()
    # We don't need to load weights, just get config
    from transformers import AutoConfig
    try:
        model_config = AutoConfig.from_pretrained(student.model_id, trust_remote_code=True)
        hidden_dim = getattr(model_config, "hidden_size", 
                     getattr(model_config, "d_model", 
                     getattr(model_config, "dim", 3072)))
        logger.info(f"🎯 Detected model hidden dimension: {hidden_dim}")
    except Exception as e:
        logger.warning(f"Could not detect dim automatically, defaulting to 3072: {e}")
        hidden_dim = 3072

    qdrant_server = QdrantMemoryServer()
    try:
        await qdrant_server.connect_qdrant()
        
        # Check if we need to recreate for new dimensions
        collections = await qdrant_server.client.get_collections()
        exists = any(c.name == "shadow_traces" for c in collections.collections)
        
        if exists:
            info = await qdrant_server.client.get_collection("shadow_traces")
            # Size can be in info.config.params.vectors or info.config.params.vectors['default']
            current_size = info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else 0
            
            if current_size != hidden_dim:
                logger.warning(f"⚠️ Existing collection has dim {current_size}, expected {hidden_dim}. Deleting...")
                await qdrant_server.client.delete_collection("shadow_traces")
                logger.info("🗑️ Deleted old collection")
                exists = False
        
        # Initialize with correct dimensions
        from qdrant_client.models import VectorParams, Distance
        
        if not exists:
            await qdrant_server.client.create_collection(
                collection_name="shadow_traces",
                vectors_config=VectorParams(size=hidden_dim, distance=Distance.COSINE)
            )
            logger.info(f"✅ Qdrant Collection Initialized: Created 'shadow_traces' with {hidden_dim} dimensions")
        else:
            logger.info(f"✅ Qdrant Collection verified (dim {hidden_dim})")
            
    except Exception as e:
        logger.error(f"❌ Qdrant Connection Error: {e}")
    finally:
        await qdrant_server.close()
        
    logger.info("\n✨ Initialization Complete")

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Qdrant Vector Memory MCP Server
Provides tools for managing semantic vector memory using Qdrant
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
    ServerCapabilities
)
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue, SearchRequest, CollectionInfo
)
import os
# import aiohttp  # Removed - no longer needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qdrant-memory-mcp")

class QdrantMemoryServer:
    def __init__(self):
        self.client = None
        self.server = Server("qdrant-memory")
        self.default_collection = "semantic_memory"
        # Sentence transformer model (lazy loaded)
        self.model = None
        self.model_name = None
        self.device = None
        self.setup_handlers()
    
    def _lazy_import_sentence_transformers(self):
        """Lazy import sentence_transformers only when needed"""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer
        except ImportError:
            raise ImportError("sentence_transformers is required but not installed")
    
    def _lazy_import_torch(self):
        """Lazy import torch only when needed"""
        try:
            import torch
            return torch
        except ImportError:
            raise ImportError("torch is required but not installed")
    
    async def _load_model(self, model_name: str = None):
        """Load sentence transformer model"""
        if self.model is not None:
            return  # Already loaded
        
        if model_name is None:
            model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        
        try:
            # Lazy import dependencies
            torch = self._lazy_import_torch()
            SentenceTransformer = self._lazy_import_sentence_transformers()
            
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading sentence transformer model {model_name} on {self.device}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_name = model_name
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
    
    async def connect_qdrant(self):
        """Connect to Qdrant database"""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY", None)
        
        try:
            # Always use HTTP URL for local development to avoid SSL issues
            if host in ["localhost", "127.0.0.1"]:
                url = f"http://{host}:{port}"
                self.client = AsyncQdrantClient(url=url, prefer_grpc=False)
            else:
                # For remote connections, use proper HTTPS
                self.client = AsyncQdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    prefer_grpc=False
                )
            
            # Test connection
            collections = await self.client.get_collections()
            logger.info(f"Connected to Qdrant successfully. Found {len(collections.collections)} collections.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="create_collection",
                    description="Create a new vector collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {"type": "string", "description": "Name of the collection"},
                            "vector_size": {"type": "integer", "description": "Size of vectors to store"},
                            "distance": {"type": "string", "description": "Distance metric (cosine, euclidean, dot)", "default": "cosine"}
                        },
                        "required": ["collection_name", "vector_size"]
                    }
                ),
                Tool(
                    name="store_memory",
                    description="Store a memory with its vector embedding",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Content to store"},
                            "vector": {"type": "array", "items": {"type": "number"}, "description": "Vector embedding"},
                            "metadata": {"type": "object", "description": "Additional metadata"},
                            "collection_name": {"type": "string", "description": "Collection to store in", "default": "semantic_memory"}
                        },
                        "required": ["content", "vector"]
                    }
                ),
                Tool(
                    name="search_memories",
                    description="Search for similar memories using vector similarity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_vector": {"type": "array", "items": {"type": "number"}, "description": "Query vector"},
                            "limit": {"type": "integer", "description": "Maximum number of results", "default": 10},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score", "default": 0.0},
                            "collection_name": {"type": "string", "description": "Collection to search in", "default": "semantic_memory"},
                            "filter_metadata": {"type": "object", "description": "Metadata filters"}
                        },
                        "required": ["query_vector"]
                    }
                ),
                Tool(
                    name="get_memory_by_id",
                    description="Retrieve a specific memory by its ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "ID of the memory to retrieve"},
                            "collection_name": {"type": "string", "description": "Collection to search in", "default": "semantic_memory"}
                        },
                        "required": ["memory_id"]
                    }
                ),
                Tool(
                    name="update_memory",
                    description="Update an existing memory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "ID of the memory to update"},
                            "content": {"type": "string", "description": "New content"},
                            "vector": {"type": "array", "items": {"type": "number"}, "description": "New vector embedding"},
                            "metadata": {"type": "object", "description": "New metadata"},
                            "collection_name": {"type": "string", "description": "Collection name", "default": "semantic_memory"}
                        },
                        "required": ["memory_id"]
                    }
                ),
                Tool(
                    name="delete_memory",
                    description="Delete a memory by its ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "ID of the memory to delete"},
                            "collection_name": {"type": "string", "description": "Collection name", "default": "semantic_memory"}
                        },
                        "required": ["memory_id"]
                    }
                ),
                Tool(
                    name="get_collection_info",
                    description="Get information about a collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {"type": "string", "description": "Collection name", "default": "semantic_memory"}
                        }
                    }
                ),
                Tool(
                    name="find_concept_cluster",
                    description="Find a cluster of related concepts around a query",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_vector": {"type": "array", "items": {"type": "number"}, "description": "Query vector"},
                            "cluster_size": {"type": "integer", "description": "Size of cluster to find", "default": 20},
                            "diversity_threshold": {"type": "number", "description": "Minimum diversity between cluster members", "default": 0.1},
                            "collection_name": {"type": "string", "description": "Collection to search in", "default": "semantic_memory"}
                        },
                        "required": ["query_vector"]
                    }
                ),
                Tool(
                    name="delete_collection",
                    description="Delete a Qdrant collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {"type": "string", "description": "Name of the collection to delete"},
                            "confirm": {"type": "boolean", "description": "Confirmation flag"}
                        },
                        "required": ["collection_name", "confirm"]
                    }
                ),
                Tool(
                    name="store_text_memory",
                    description="Store a text memory (generates embedding internally, no vectors sent to LLM)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text content to store"},
                            "metadata": {"type": "object", "description": "Additional metadata", "default": {}},
                            "collection_name": {"type": "string", "description": "Collection to store in", "default": "semantic_memory"}
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="search_text_memories",
                    description="Search memories using text query (generates embedding internally, no vectors sent to LLM)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_text": {"type": "string", "description": "Text to search for"},
                            "limit": {"type": "integer", "description": "Maximum number of results", "default": 5},
                            "score_threshold": {"type": "number", "description": "Minimum similarity score", "default": 0.0},
                            "collection_name": {"type": "string", "description": "Collection to search in", "default": "semantic_memory"}
                        },
                        "required": ["query_text"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "create_collection":
                    result = await self.create_collection(
                        arguments["collection_name"],
                        arguments["vector_size"],
                        arguments.get("distance", "cosine")
                    )
                elif name == "store_memory":
                    result = await self.store_memory(
                        arguments["content"],
                        arguments["vector"],
                        arguments.get("metadata", {}),
                        arguments.get("collection_name", self.default_collection)
                    )
                elif name == "search_memories":
                    result = await self.search_memories(
                        arguments["query_vector"],
                        arguments.get("limit", 10),
                        arguments.get("score_threshold", 0.0),
                        arguments.get("collection_name", self.default_collection),
                        arguments.get("filter_metadata", {})
                    )
                elif name == "get_memory_by_id":
                    result = await self.get_memory_by_id(
                        arguments["memory_id"],
                        arguments.get("collection_name", self.default_collection)
                    )
                elif name == "update_memory":
                    result = await self.update_memory(
                        arguments["memory_id"],
                        arguments.get("content"),
                        arguments.get("vector"),
                        arguments.get("metadata"),
                        arguments.get("collection_name", self.default_collection)
                    )
                elif name == "delete_memory":
                    result = await self.delete_memory(
                        arguments["memory_id"],
                        arguments.get("collection_name", self.default_collection)
                    )
                elif name == "get_collection_info":
                    result = await self.get_collection_info(
                        arguments.get("collection_name", self.default_collection)
                    )
                elif name == "find_concept_cluster":
                    result = await self.find_concept_cluster(
                        arguments["query_vector"],
                        arguments.get("cluster_size", 20),
                        arguments.get("diversity_threshold", 0.1),
                        arguments.get("collection_name", self.default_collection)
                    )
                elif name == "delete_collection":
                    result = await self.delete_collection(
                        arguments["collection_name"],
                        arguments.get("confirm", False)
                    )
                elif name == "store_text_memory":
                    result = await self.store_text_memory(
                        arguments["text"],
                        arguments.get("metadata", {}),
                        arguments.get("collection_name", self.default_collection)
                    )
                elif name == "search_text_memories":
                    result = await self.search_text_memories(
                        arguments["query_text"],
                        arguments.get("limit", 5),
                        arguments.get("score_threshold", 0.0),
                        arguments.get("collection_name", self.default_collection)
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def create_collection(self, collection_name: str, vector_size: int, distance: str) -> Dict[str, Any]:
        """Create a new vector collection"""
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map.get(distance, Distance.COSINE)
            )
        )
        
        return {
            "success": True,
            "collection_name": collection_name,
            "vector_size": vector_size,
            "distance": distance
        }
    
    async def store_memory(self, content: str, vector: List[float], 
                          metadata: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """Store a memory with its vector"""
        memory_id = str(uuid.uuid4())
        
        # Add content to metadata
        full_metadata = {
            "content": content,
            "timestamp": asyncio.get_event_loop().time(),
            **metadata
        }
        
        point = PointStruct(
            id=memory_id,
            vector=vector,
            payload=full_metadata
        )
        
        await self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        return {
            "success": True,
            "memory_id": memory_id,
            "collection_name": collection_name
        }
    
    async def search_memories(self, query_vector: List[float], limit: int, 
                            score_threshold: float, collection_name: str,
                            filter_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Search for similar memories"""
        search_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            search_filter = Filter(must=conditions)
        
        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=search_filter,
            with_payload=True
        )
        
        memories = []
        for result in results:
            memories.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "content"}
            })
        
        return {
            "success": True,
            "memories": memories,
            "count": len(memories)
        }
    
    async def get_memory_by_id(self, memory_id: str, collection_name: str) -> Dict[str, Any]:
        """Get a specific memory by ID"""
        results = await self.client.retrieve(
            collection_name=collection_name,
            ids=[memory_id],
            with_payload=True,
            with_vectors=True
        )
        
        if results:
            result = results[0]
            return {
                "success": True,
                "memory": {
                    "id": result.id,
                    "content": result.payload.get("content", ""),
                    "vector": result.vector,
                    "metadata": {k: v for k, v in result.payload.items() if k != "content"}
                }
            }
        else:
            return {"success": False, "message": "Memory not found"}
    
    async def update_memory(self, memory_id: str, content: Optional[str], 
                          vector: Optional[List[float]], metadata: Optional[Dict[str, Any]],
                          collection_name: str) -> Dict[str, Any]:
        """Update an existing memory"""
        # Get current memory
        current = await self.get_memory_by_id(memory_id, collection_name)
        if not current["success"]:
            return current
        
        # Update fields
        new_content = content if content is not None else current["memory"]["content"]
        new_vector = vector if vector is not None else current["memory"]["vector"]
        new_metadata = metadata if metadata is not None else current["memory"]["metadata"]
        
        # Store updated memory
        full_metadata = {
            "content": new_content,
            "timestamp": asyncio.get_event_loop().time(),
            **new_metadata
        }
        
        point = PointStruct(
            id=memory_id,
            vector=new_vector,
            payload=full_metadata
        )
        
        await self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        return {"success": True, "memory_id": memory_id}
    
    async def delete_memory(self, memory_id: str, collection_name: str) -> Dict[str, Any]:
        """Delete a memory"""
        await self.client.delete(
            collection_name=collection_name,
            points_selector=[memory_id]
        )
        
        return {"success": True, "memory_id": memory_id}
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = await self.client.get_collection(collection_name)
            return {
                "success": True,
                "collection_name": collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value if info.status else "unknown"
            }
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    async def find_concept_cluster(self, query_vector: List[float], cluster_size: int,
                                 diversity_threshold: float, collection_name: str) -> Dict[str, Any]:
        """Find a diverse cluster of concepts around a query"""
        # Get initial candidates (more than needed)
        initial_results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=cluster_size * 3,  # Get 3x more candidates
            with_payload=True,
            with_vectors=True
        )
        
        if not initial_results:
            return {"success": True, "cluster": [], "count": 0}
        
        # Select diverse cluster using simple greedy algorithm
        cluster = [initial_results[0]]  # Start with most similar
        
        for candidate in initial_results[1:]:
            if len(cluster) >= cluster_size:
                break
            
            # Check if candidate is diverse enough from existing cluster members
            is_diverse = True
            for cluster_member in cluster:
                # Simple cosine similarity check
                similarity = self._cosine_similarity(candidate.vector, cluster_member.vector)
                if similarity > (1.0 - diversity_threshold):
                    is_diverse = False
                    break
            
            if is_diverse:
                cluster.append(candidate)
        
        # Format results
        cluster_memories = []
        for result in cluster:
            cluster_memories.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload.get("content", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "content"}
            })
        
        return {
            "success": True,
            "cluster": cluster_memories,
            "count": len(cluster_memories)
        }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    async def delete_collection(self, collection_name: str, confirm: bool) -> Dict[str, Any]:
        """Delete a Qdrant collection"""
        if not confirm:
            return {
                "success": False,
                "message": "Deletion requires confirmation. Set confirm=True to proceed."
            }
        
        await self.client.delete_collection(collection_name=collection_name)
        
        return {
            "success": True,
            "collection_name": collection_name,
            "message": f"Collection '{collection_name}' deleted successfully"
        }
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence transformer library directly"""
        try:
            # Load model if not already loaded
            if self.model is None:
                await self._load_model()
            
            # Generate embedding
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise Exception(f"Embedding generation failed: {e}")
    
    async def store_text_memory(self, text: str, metadata: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """Store a text memory (generates embedding internally)"""
        try:
            # Generate embedding internally
            vector = await self._generate_embedding(text)
            
            # Store using existing method
            result = await self.store_memory(text, vector, metadata, collection_name)
            
            # Return clean result without vector data
            return {
                "success": True,
                "memory_id": result["memory_id"],
                "collection_name": result["collection_name"],
                "text": text,
                "metadata": metadata,
                "embedding_dimensions": len(vector)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_text_memories(self, query_text: str, limit: int, 
                                 score_threshold: float, collection_name: str) -> Dict[str, Any]:
        """Search memories using text query (generates embedding internally)"""
        try:
            # Generate embedding for query
            query_vector = await self._generate_embedding(query_text)
            
            # Search using existing method
            search_result = await self.search_memories(
                query_vector, limit, score_threshold, collection_name, {}
            )
            
            if not search_result["success"]:
                return search_result
            
            # Return clean results without vector data
            return {
                "success": True,
                "query_text": query_text,
                "results": search_result["memories"],
                "count": search_result["count"],
                "collection_name": collection_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close the Qdrant connection"""
        if self.client:
            await self.client.close()

async def main():
    """Main server function"""
    server_instance = QdrantMemoryServer()
    
    # Connect to Qdrant
    await server_instance.connect_qdrant()
    
    try:
        # Run the MCP server
        async with stdio_server() as (read_stream, write_stream):
            await server_instance.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="qdrant-memory",
                    server_version="1.0.0",
                    capabilities=ServerCapabilities(
                        tools={}
                    ),
                ),
            )
    finally:
        await server_instance.close()

if __name__ == "__main__":
    asyncio.run(main())

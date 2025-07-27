#!/usr/bin/env python3
"""
Sentence Transformers MCP Server with HTTP API
Provides tools for generating embeddings using sentence transformers with GPU acceleration
Also provides HTTP endpoints for direct API access
"""

import asyncio
import json
import logging
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
# Lazy imports - only import when actually needed
# import sentence_transformers  # Moved to lazy loading
# import torch  # Moved to lazy loading  
# import numpy as np  # Moved to lazy loading
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentence-transformers-mcp")

class SentenceTransformersServer:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.server = Server("sentence-transformers")
        self.device = None  # Will be set when model is loaded
        self._model_loading = False  # Prevent concurrent loading
        self.setup_handlers()
    
    def _lazy_import_torch(self):
        """Lazy import torch only when needed"""
        try:
            import torch
            return torch
        except ImportError:
            raise ImportError("torch is required but not installed")
    
    def _lazy_import_sentence_transformers(self):
        """Lazy import sentence_transformers only when needed"""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer
        except ImportError:
            raise ImportError("sentence_transformers is required but not installed")
    
    def _lazy_import_numpy(self):
        """Lazy import numpy only when needed"""
        try:
            import numpy as np
            return np
        except ImportError:
            raise ImportError("numpy is required but not installed")
    
    async def load_model(self, model_name: str = None):
        """Load a sentence transformer model"""
        if model_name is None:
            model_name = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
        
        try:
            # Lazy import dependencies
            torch = self._lazy_import_torch()
            SentenceTransformer = self._lazy_import_sentence_transformers()
            
            # Set device now that torch is available
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading model {model_name} on device {self.device}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model_name = model_name
            logger.info(f"Model {model_name} loaded successfully on {self.device}")
            
            # Log GPU info if available
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="load_model",
                    description="Load a sentence transformer model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {"type": "string", "description": "Name of the model to load", "default": "all-MiniLM-L6-v2"}
                        }
                    }
                ),
                Tool(
                    name="generate_embedding",
                    description="Generate embedding for a single text",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to embed"},
                            "normalize": {"type": "boolean", "description": "Whether to normalize the embedding", "default": True}
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="batch_embeddings",
                    description="Generate embeddings for multiple texts efficiently",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "texts": {"type": "array", "items": {"type": "string"}, "description": "List of texts to embed"},
                            "batch_size": {"type": "integer", "description": "Batch size for processing", "default": 32},
                            "normalize": {"type": "boolean", "description": "Whether to normalize embeddings", "default": True}
                        },
                        "required": ["texts"]
                    }
                ),
                Tool(
                    name="similarity_score",
                    description="Calculate similarity between two texts",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text1": {"type": "string", "description": "First text"},
                            "text2": {"type": "string", "description": "Second text"},
                            "metric": {"type": "string", "description": "Similarity metric (cosine, dot, euclidean)", "default": "cosine"}
                        },
                        "required": ["text1", "text2"]
                    }
                ),
                Tool(
                    name="find_most_similar",
                    description="Find the most similar text from a list of candidates",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_text": {"type": "string", "description": "Query text"},
                            "candidate_texts": {"type": "array", "items": {"type": "string"}, "description": "List of candidate texts"},
                            "top_k": {"type": "integer", "description": "Number of top results to return", "default": 5},
                            "metric": {"type": "string", "description": "Similarity metric", "default": "cosine"}
                        },
                        "required": ["query_text", "candidate_texts"]
                    }
                ),
                Tool(
                    name="get_model_info",
                    description="Get information about the currently loaded model",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="cluster_texts",
                    description="Cluster texts based on semantic similarity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "texts": {"type": "array", "items": {"type": "string"}, "description": "Texts to cluster"},
                            "num_clusters": {"type": "integer", "description": "Number of clusters", "default": 5},
                            "method": {"type": "string", "description": "Clustering method (kmeans, hierarchical)", "default": "kmeans"}
                        },
                        "required": ["texts"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "load_model":
                    result = await self.load_model_tool(arguments.get("model_name"))
                elif name == "generate_embedding":
                    result = await self.generate_embedding(
                        arguments["text"],
                        arguments.get("normalize", True)
                    )
                elif name == "batch_embeddings":
                    result = await self.batch_embeddings(
                        arguments["texts"],
                        arguments.get("batch_size", 32),
                        arguments.get("normalize", True)
                    )
                elif name == "similarity_score":
                    result = await self.similarity_score(
                        arguments["text1"],
                        arguments["text2"],
                        arguments.get("metric", "cosine")
                    )
                elif name == "find_most_similar":
                    result = await self.find_most_similar(
                        arguments["query_text"],
                        arguments["candidate_texts"],
                        arguments.get("top_k", 5),
                        arguments.get("metric", "cosine")
                    )
                elif name == "get_model_info":
                    result = await self.get_model_info()
                elif name == "cluster_texts":
                    result = await self.cluster_texts(
                        arguments["texts"],
                        arguments.get("num_clusters", 5),
                        arguments.get("method", "kmeans")
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def load_model_tool(self, model_name: Optional[str]) -> Dict[str, Any]:
        """Load model tool wrapper"""
        await self.load_model(model_name)
        return {
            "success": True,
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.model.get_sentence_embedding_dimension()
        }
    
    async def generate_embedding(self, text: str, normalize: bool) -> Dict[str, Any]:
        """Generate embedding for a single text"""
        if self.model is None:
            await self.load_model()
        
        embedding = self.model.encode(text, normalize_embeddings=normalize)
        
        return {
            "success": True,
            "text": text,
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        }
    
    async def batch_embeddings(self, texts: List[str], batch_size: int, normalize: bool) -> Dict[str, Any]:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            await self.load_model()
        
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )
        
        return {
            "success": True,
            "embeddings": embeddings.tolist(),
            "count": len(embeddings),
            "dimension": embeddings.shape[1] if len(embeddings) > 0 else 0
        }
    
    async def similarity_score(self, text1: str, text2: str, metric: str) -> Dict[str, Any]:
        """Calculate similarity between two texts"""
        if self.model is None:
            await self.load_model()
        
        np = self._lazy_import_numpy()
        embeddings = self.model.encode([text1, text2], normalize_embeddings=True)
        
        if metric == "cosine":
            similarity = float(np.dot(embeddings[0], embeddings[1]))
        elif metric == "dot":
            similarity = float(np.dot(embeddings[0], embeddings[1]))
        elif metric == "euclidean":
            similarity = float(1.0 / (1.0 + np.linalg.norm(embeddings[0] - embeddings[1])))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return {
            "success": True,
            "text1": text1,
            "text2": text2,
            "similarity": similarity,
            "metric": metric
        }
    
    async def find_most_similar(self, query_text: str, candidate_texts: List[str], 
                              top_k: int, metric: str) -> Dict[str, Any]:
        """Find most similar texts from candidates"""
        if self.model is None:
            await self.load_model()
        
        np = self._lazy_import_numpy()
        
        # Encode all texts
        all_texts = [query_text] + candidate_texts
        embeddings = self.model.encode(all_texts, normalize_embeddings=True)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Calculate similarities
        if metric == "cosine":
            similarities = np.dot(candidate_embeddings, query_embedding)
        elif metric == "dot":
            similarities = np.dot(candidate_embeddings, query_embedding)
        elif metric == "euclidean":
            distances = np.linalg.norm(candidate_embeddings - query_embedding, axis=1)
            similarities = 1.0 / (1.0 + distances)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": candidate_texts[idx],
                "similarity": float(similarities[idx]),
                "index": int(idx)
            })
        
        return {
            "success": True,
            "query_text": query_text,
            "results": results,
            "metric": metric
        }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {"success": False, "message": "No model loaded"}
        
        info = {
            "success": True,
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown')
        }
        
        torch = self._lazy_import_torch()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name()
            info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            info["gpu_memory_cached"] = torch.cuda.memory_reserved()
        
        return info
    
    async def cluster_texts(self, texts: List[str], num_clusters: int, method: str) -> Dict[str, Any]:
        """Cluster texts based on semantic similarity"""
        if self.model is None:
            await self.load_model()
        
        if len(texts) < num_clusters:
            return {"success": False, "message": "Number of texts must be >= number of clusters"}
        
        # Generate embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        if method == "kmeans":
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(embeddings)
            
        elif method == "hierarchical":
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_labels = clusterer.fit_predict(embeddings)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Organize results by cluster
        clusters = {}
        for i, (text, label) in enumerate(zip(texts, cluster_labels)):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                "text": text,
                "index": i
            })
        
        return {
            "success": True,
            "clusters": clusters,
            "num_clusters": len(clusters),
            "method": method
        }

# Global server instance for HTTP handler
_global_server_instance = None

class HTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for direct API access"""
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == '/health':
                # Health check endpoint
                self._send_json_response({"status": "healthy", "model_loaded": _global_server_instance.model is not None})
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"HTTP GET request error: {e}")
            self._send_error(500, str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            # Parse the path
            path = self.path
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))
            
            # Route to appropriate handler
            if path == '/similarity':
                result = self._handle_similarity(data)
            elif path == '/embeddings':
                result = self._handle_embeddings(data)
            elif path == '/embedding':
                result = self._handle_single_embedding(data)
            else:
                self._send_error(404, "Endpoint not found")
                return
            
            # Send response
            self._send_json_response(result)
            
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            self._send_error(500, str(e))
    
    def _handle_similarity(self, data):
        """Handle similarity calculation"""
        text1 = data.get('text1')
        text2 = data.get('text2')
        
        if not text1 or not text2:
            raise ValueError("Both text1 and text2 are required")
        
        # Use the global server instance
        if _global_server_instance.model is None:
            # Load model synchronously for HTTP requests
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_global_server_instance.load_model())
            loop.close()
        
        np = _global_server_instance._lazy_import_numpy()
        embeddings = _global_server_instance.model.encode([text1, text2], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        
        return {"similarity": similarity}
    
    def _handle_embeddings(self, data):
        """Handle batch embeddings"""
        texts = data.get('texts', [])
        
        if not texts:
            raise ValueError("texts array is required")
        
        # Load model if needed
        if _global_server_instance.model is None:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_global_server_instance.load_model())
            loop.close()
        
        embeddings = _global_server_instance.model.encode(texts, normalize_embeddings=True)
        return {"embeddings": embeddings.tolist()}
    
    def _handle_single_embedding(self, data):
        """Handle single embedding"""
        text = data.get('text')
        
        if not text:
            raise ValueError("text is required")
        
        # Load model if needed
        if _global_server_instance.model is None:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_global_server_instance.load_model())
            loop.close()
        
        embedding = _global_server_instance.model.encode(text, normalize_embeddings=True)
        return {"embedding": embedding.tolist()}
    
    def _send_json_response(self, data):
        """Send JSON response"""
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def _send_error(self, code, message):
        """Send error response"""
        error_data = {"error": message}
        response = json.dumps(error_data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"HTTP {format % args}")

def start_http_server(server_instance, port=8001):
    """Start HTTP server in a separate thread"""
    global _global_server_instance
    _global_server_instance = server_instance
    
    httpd = HTTPServer(('localhost', port), HTTPRequestHandler)
    logger.info(f"Starting HTTP server on port {port}")
    
    # Run in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    
    return httpd

async def main():
    """Main server function"""
    server_instance = SentenceTransformersServer()
    
    # Start HTTP server for direct API access
    http_server = start_http_server(server_instance, port=8001)
    
    # NOTE: Model is now loaded lazily when first needed
    # This saves ~200MB of memory on startup
    logger.info("Sentence Transformers server starting with lazy loading enabled")
    logger.info("HTTP API available at http://localhost:8001")
    
    try:
        # Run the MCP server
        async with stdio_server() as (read_stream, write_stream):
            await server_instance.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="sentence-transformers",
                    server_version="1.0.0",
                    capabilities=ServerCapabilities(
                        tools={}
                    ),
                ),
            )
    finally:
        # Shutdown HTTP server
        http_server.shutdown()
        
        # Cleanup GPU memory if model was loaded
        if server_instance.model is not None:
            try:
                torch = server_instance._lazy_import_torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cleaned up")
            except ImportError:
                pass  # torch not available, no cleanup needed

if __name__ == "__main__":
    asyncio.run(main())

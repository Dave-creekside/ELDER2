#!/usr/bin/env python3
"""
Neo4j Semantic Hypergraph MCP Server
Provides tools for managing semantic hypergraphs with weighted edges in Neo4j
"""

import asyncio
import json
import datetime
import logging
import requests
import uuid
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
    ServerCapabilities,
    NotificationParams
)
from neo4j import AsyncGraphDatabase

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from streamlined_consciousness.config import config

import math
from collections import defaultdict

# Configure logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger("neo4j-semantic-hypergraph-mcp")


def serialize_for_json(obj):
    """Convert Neo4j objects to JSON-serializable format"""
    # Handle None
    if obj is None:
        return None
    
    # Handle datetime objects
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    
    # Handle Neo4j specific types
    if hasattr(obj, '__class__') and obj.__class__.__name__ in ['Node', 'Relationship', 'Path']:
        # Convert Neo4j objects to dictionaries
        if hasattr(obj, '_properties'):
            return serialize_for_json(dict(obj._properties))
        elif hasattr(obj, 'items'):
            return serialize_for_json(dict(obj.items()))
        else:
            return str(obj)
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    
    # Handle sets
    if isinstance(obj, set):
        return list(serialize_for_json(item) for item in obj)
    
    # Handle complex objects with __dict__ (but avoid infinite recursion)
    if hasattr(obj, '__dict__') and not isinstance(obj, type):
        try:
            return {k: serialize_for_json(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        except (AttributeError, TypeError):
            return str(obj)
    
    # Handle basic types that are already JSON serializable
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Fallback to string representation
    return str(obj)

class Neo4jSemanticHypergraphServer:
    def __init__(self):
        self.driver = None
        self.server = Server("neo4j-hypergraph")
        self.model = None
        self.model_name = None
        self.device = None
        self.setup_handlers()
    
    def _lazy_import_torch(self):
        """Lazy import torch only when needed"""
        try:
            import torch
            return torch
        except ImportError:
            logger.warning("torch not available - using fallback similarity")
            return None
    
    def _lazy_import_sentence_transformers(self):
        """Lazy import sentence_transformers only when needed"""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer
        except ImportError:
            logger.warning("sentence_transformers not available - using fallback similarity")
            return None
    
    def _lazy_import_numpy(self):
        """Lazy import numpy only when needed"""
        try:
            import numpy as np
            return np
        except ImportError:
            logger.warning("numpy not available - using fallback similarity")
            return None
    
    async def load_model(self, model_name: str = None):
        """Load a sentence transformer model"""
        if model_name is None:
            model_name = config.SENTENCE_TRANSFORMER_MODEL
        
        try:
            # Lazy import dependencies
            torch = self._lazy_import_torch()
            SentenceTransformer = self._lazy_import_sentence_transformers()
            
            if torch is None or SentenceTransformer is None:
                logger.warning("Required dependencies not available - using fallback similarity")
                return False
            
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
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def connect_neo4j(self):
        """Connect to Neo4j database"""
        logger.info(f"Attempting to connect to Neo4j at {config.NEO4J_URI}")
        try:
            self.driver = AsyncGraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
            # Verify connectivity
            await self.driver.verify_connectivity()
            logger.info("Neo4j connection verified successfully.")
            # Test with a simple query
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            logger.info("Successfully executed a test query.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
            raise
    
    async def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Get semantic similarity between two texts using sentence transformers"""
        try:
            # Load model if not already loaded
            if self.model is None:
                success = await self.load_model()
                if not success:
                    # Fallback to simple text similarity
                    return self._fallback_similarity(text1, text2)
            
            np = self._lazy_import_numpy()
            if np is None:
                return self._fallback_similarity(text1, text2)
            
            # Generate embeddings and calculate cosine similarity
            embeddings = self.model.encode([text1, text2], normalize_embeddings=True)
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            return similarity
            
        except Exception as e:
            logger.warning(f"Failed to get semantic similarity: {e}")
            return self._fallback_similarity(text1, text2)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity calculation using simple text overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            # Load model if not already loaded
            if self.model is None:
                success = await self.load_model()
                if not success:
                    return []
            
            embeddings = self.model.encode(
                texts, 
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.warning(f"Failed to get embeddings: {e}")
            return []
    
    async def calculate_batch_similarities(self, concept_pairs: List[Tuple[str, str]], 
                                         boost_hyperedge_members: bool = True) -> Dict[Tuple[str, str], float]:
        """Calculate semantic similarities for multiple concept pairs in batch"""
        try:
            # Extract unique concepts
            unique_concepts = set()
            for concept1, concept2 in concept_pairs:
                unique_concepts.add(concept1)
                unique_concepts.add(concept2)
            
            concept_list = list(unique_concepts)
            
            # Load model if not already loaded
            if self.model is None:
                success = await self.load_model()
                if not success:
                    # Fallback to simple similarity for all pairs
                    result = {}
                    for concept1, concept2 in concept_pairs:
                        result[(concept1, concept2)] = self._fallback_similarity(concept1, concept2)
                    return result
            
            np = self._lazy_import_numpy()
            if np is None:
                # Fallback if numpy not available
                result = {}
                for concept1, concept2 in concept_pairs:
                    result[(concept1, concept2)] = self._fallback_similarity(concept1, concept2)
                return result
            
            # Get embeddings for all concepts at once
            embeddings = self.model.encode(concept_list, normalize_embeddings=True)
            
            # Create concept to index mapping
            concept_to_idx = {concept: i for i, concept in enumerate(concept_list)}
            
            # Calculate similarities
            similarities = {}
            for concept1, concept2 in concept_pairs:
                idx1 = concept_to_idx[concept1]
                idx2 = concept_to_idx[concept2]
                
                # Cosine similarity (already normalized)
                similarity = float(np.dot(embeddings[idx1], embeddings[idx2]))
                similarities[(concept1, concept2)] = similarity
            
            # Boost for hyperedge co-membership if requested
            if boost_hyperedge_members:
                async with self.driver.session() as session:
                    # Query to find which pairs share hyperedges
                    hyperedge_query = """
                    UNWIND $pairs as pair
                    MATCH (a:Concept {name: pair[0]})-[:MEMBER_OF]->(he:Hyperedge)<-[:MEMBER_OF]-(b:Concept {name: pair[1]})
                    RETURN pair[0] as concept1, pair[1] as concept2, count(DISTINCT he) as shared_hyperedges
                    """
                    
                    # Convert pairs to list format for query
                    pairs_list = [[c1, c2] for c1, c2 in concept_pairs]
                    
                    result = await session.run(hyperedge_query, pairs=pairs_list)
                    
                    # Apply boost to pairs that share hyperedges
                    async for record in result:
                        concept1 = record["concept1"]
                        concept2 = record["concept2"]
                        shared_count = record["shared_hyperedges"]
                        
                        if shared_count > 0:
                            # Boost by 0.2 for hyperedge co-membership
                            hyperedge_factor = min(shared_count / 3.0, 1.0) * 0.3
                            current_sim = similarities.get((concept1, concept2), 0.5)
                            
                            # Weighted combination: 70% embedding similarity + 30% hyperedge factor
                            boosted_sim = (0.7 * current_sim) + hyperedge_factor
                            similarities[(concept1, concept2)] = min(boosted_sim, 1.0)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to calculate batch similarities: {e}")
            # Fallback to individual calculations
            result = {}
            for concept1, concept2 in concept_pairs:
                result[(concept1, concept2)] = await self.get_semantic_similarity(concept1, concept2)
            return result
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="create_concept_node",
                    description="Create a new concept node in the hypergraph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the concept"},
                            "properties": {"type": "object", "description": "Additional properties for the node"}
                        },
                        "required": ["name"]
                    }
                ),
                Tool(
                    name="delete_concept_node",
                    description="Delete a concept node from the hypergraph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the concept to delete"},
                            "cascade": {"type": "boolean", "description": "Whether to also delete connected relationships", "default": True}
                        },
                        "required": ["name"]
                    }
                ),
                Tool(
                    name="create_hyperedge",
                    description="Create a semantic hyperedge connecting multiple concepts with automatic weight calculation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "members": {"type": "array", "items": {"type": "string"}, "description": "List of concept names to connect"},
                            "label": {"type": "string", "description": "Optional label for the hyperedge"},
                            "auto_calculate_weights": {"type": "boolean", "description": "Whether to automatically calculate semantic weights", "default": True}
                        },
                        "required": ["members"]
                    }
                ),
                Tool(
                    name="create_relationship",
                    description="Create a weighted relationship between two concepts",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "from_concept": {"type": "string", "description": "Source concept name"},
                            "to_concept": {"type": "string", "description": "Target concept name"},
                            "relationship_type": {"type": "string", "description": "Type of relationship"},
                            "properties": {"type": "object", "description": "Additional properties for the relationship"},
                            "auto_calculate_weight": {"type": "boolean", "description": "Whether to automatically calculate semantic weight", "default": True}
                        },
                        "required": ["from_concept", "to_concept", "relationship_type"]
                    }
                ),
                Tool(
                    name="apply_ca_rules",
                    description="Apply cellular automata rules to evolve the hypergraph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "common_neighbors_threshold": {"type": "integer", "description": "Minimum common neighbors to create connection", "default": 2},
                            "min_similarity": {"type": "number", "description": "Minimum similarity for new connections", "default": 0.4}
                        }
                    }
                ),
                Tool(
                    name="apply_ca_rules_enhanced",
                    description="Enhanced CA with pruning, creation, and operation limits for dream states",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "common_neighbors_threshold": {"type": "integer", "description": "Minimum common neighbors to create connection", "default": 2},
                            "min_similarity": {"type": "number", "description": "Minimum similarity for new connections", "default": 0.4},
                            "prune_threshold": {"type": "number", "description": "Remove edges below this weight", "default": 0.1},
                            "max_operations": {"type": "integer", "description": "Maximum total operations (prune + create)", "default": 2000}
                        }
                    }
                ),
                Tool(
                    name="get_ca_connection_candidates",
                    description="Get connection candidates for CA without truncation - returns compact format",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "min_common_neighbors": {"type": "integer", "description": "Minimum common neighbors", "default": 2},
                            "max_candidates": {"type": "integer", "description": "Maximum candidates to return", "default": 100},
                            "format": {"type": "string", "description": "Output format", "default": "compact", "enum": ["compact", "detailed"]}
                        }
                    }
                ),
                Tool(
                    name="find_semantic_path",
                    description="Find the shortest weighted path between two concepts",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "from_concept": {"type": "string", "description": "Source concept name"},
                            "to_concept": {"type": "string", "description": "Target concept name"},
                            "min_weight": {"type": "number", "description": "Minimum edge weight to consider", "default": 0.3},
                            "max_length": {"type": "integer", "description": "Maximum path length", "default": 5}
                        },
                        "required": ["from_concept", "to_concept"]
                    }
                ),
                Tool(
                    name="get_semantic_neighbors",
                    description="Get semantically similar neighbors of a concept",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "concept_name": {"type": "string", "description": "Name of the concept"},
                            "min_weight": {"type": "number", "description": "Minimum semantic weight", "default": 0.5},
                            "max_depth": {"type": "integer", "description": "Maximum depth to traverse", "default": 2}
                        },
                        "required": ["concept_name"]
                    }
                ),
                Tool(
                    name="get_hyperedge_info",
                    description="Get information about a specific hyperedge",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hyperedge_id": {"type": "string", "description": "ID of the hyperedge"}
                        },
                        "required": ["hyperedge_id"]
                    }
                ),
                Tool(
                    name="remove_hyperedge",
                    description="Remove a hyperedge from the graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hyperedge_id": {"type": "string", "description": "ID of the hyperedge to remove"}
                        },
                        "required": ["hyperedge_id"]
                    }
                ),
                Tool(
                    name="evolve_graph",
                    description="Run multiple iterations of cellular automata evolution",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "iterations": {"type": "integer", "description": "Number of evolution iterations", "default": 1},
                            "common_neighbors_threshold": {"type": "integer", "description": "Minimum common neighbors", "default": 2}
                        }
                    }
                ),
                Tool(
                    name="query_cypher",
                    description="Execute a Cypher query on the hypergraph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Cypher query to execute"},
                            "parameters": {"type": "object", "description": "Query parameters"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_concept_neighbors",
                    description="Get all concepts connected to a given concept",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "concept_name": {"type": "string", "description": "Name of the concept"},
                            "max_depth": {"type": "integer", "description": "Maximum depth to traverse", "default": 1}
                        },
                        "required": ["concept_name"]
                    }
                ),
                Tool(
                    name="find_shortest_path",
                    description="Find the shortest path between two concepts",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "from_concept": {"type": "string", "description": "Source concept name"},
                            "to_concept": {"type": "string", "description": "Target concept name"},
                            "max_length": {"type": "integer", "description": "Maximum path length", "default": 5}
                        },
                        "required": ["from_concept", "to_concept"]
                    }
                ),
                Tool(
                    name="get_graph_stats",
                    description="Get statistics about the hypergraph",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="bootstrap_vocabulary",
                    description="Bootstrap the hypergraph with initial vocabulary",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "vocabulary": {"type": "array", "items": {"type": "string"}, "description": "List of vocabulary words"},
                            "project_id": {"type": "string", "description": "Project ID to bootstrap (optional)", "default": "default"}
                        },
                        "required": ["vocabulary"]
                    }
                ),
                Tool(
                    name="create_project",
                    description="Create a new hypergraph project",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string", "description": "Unique project identifier"},
                            "name": {"type": "string", "description": "Human-readable project name"},
                            "description": {"type": "string", "description": "Project description"},
                            "template": {"type": "string", "description": "Template to use (blank, basic, research)", "default": "blank"}
                        },
                        "required": ["project_id", "name"]
                    }
                ),
                Tool(
                    name="list_projects",
                    description="List all available hypergraph projects",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="switch_project",
                    description="Switch to a different project context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string", "description": "Project ID to switch to"}
                        },
                        "required": ["project_id"]
                    }
                ),
                Tool(
                    name="get_current_project",
                    description="Get information about the current project",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="delete_project",
                    description="Delete a project and all its data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string", "description": "Project ID to delete"},
                            "confirm": {"type": "boolean", "description": "Confirmation flag", "default": False}
                        },
                        "required": ["project_id", "confirm"]
                    }
                ),
                Tool(
                    name="export_project",
                    description="Export project data to JSON format",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string", "description": "Project ID to export"},
                            "include_weights": {"type": "boolean", "description": "Include semantic weights", "default": True}
                        },
                        "required": ["project_id"]
                    }
                ),
                Tool(
                    name="import_project",
                    description="Import project data from JSON format",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string", "description": "New project ID"},
                            "project_data": {"type": "object", "description": "Project data to import"},
                            "overwrite": {"type": "boolean", "description": "Overwrite if project exists", "default": False}
                        },
                        "required": ["project_id", "project_data"]
                    }
                ),
                Tool(
                    name="create_blank_project",
                    description="Create a new blank consciousness project with only core memory nodes",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "description": "Natural description of the project purpose"},
                            "name": {"type": "string", "description": "Optional human-readable name", "default": ""}
                        },
                        "required": ["description"]
                    }
                ),
                Tool(
                    name="switch_to_project",
                    description="Switch to a different consciousness project by description or name",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_description": {"type": "string", "description": "Description or name of the project to switch to"}
                        },
                        "required": ["project_description"]
                    }
                ),
                Tool(
                    name="list_my_projects",
                    description="List all available consciousness projects",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="describe_current_project",
                    description="Get detailed information about the current consciousness project",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="archive_project",
                    description="Archive a consciousness project (mark as inactive)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "project_description": {"type": "string", "description": "Description or name of the project to archive"}
                        },
                        "required": ["project_description"]
                    }
                ),
                Tool(
                    name="clear_all_data",
                    description="Clear all hypergraph data (nuclear option for testing)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "confirm_clear": {"type": "boolean", "description": "Confirmation flag - must be true", "default": False},
                            "preserve_projects": {"type": "boolean", "description": "Keep project metadata", "default": False}
                        },
                        "required": ["confirm_clear"]
                    }
                ),
                Tool(
                    name="archive_large_graph",
                    description="Archive the current hypergraph if it exceeds size thresholds",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "max_nodes": {"type": "integer", "description": "Maximum nodes before archiving", "default": 1000},
                            "max_edges": {"type": "integer", "description": "Maximum edges before archiving", "default": 10000},
                            "archive_name": {"type": "string", "description": "Name for the archive", "default": "auto_archive"}
                        }
                    }
                ),
                Tool(
                    name="create_testing_snapshot",
                    description="Create a snapshot of current graph state for testing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "snapshot_name": {"type": "string", "description": "Name for the snapshot"},
                            "description": {"type": "string", "description": "Description of the snapshot", "default": ""}
                        },
                        "required": ["snapshot_name"]
                    }
                ),
                Tool(
                    name="restore_snapshot",
                    description="Restore graph to a previous snapshot state",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "snapshot_name": {"type": "string", "description": "Name of snapshot to restore"},
                            "confirm_restore": {"type": "boolean", "description": "Confirmation flag", "default": False}
                        },
                        "required": ["snapshot_name", "confirm_restore"]
                    }
                ),
                Tool(
                    name="list_snapshots",
                    description="List all available testing snapshots",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_database_size",
                    description="Get detailed information about database size and complexity",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_breakdown": {"type": "boolean", "description": "Include detailed breakdown", "default": True}
                        }
                    }
                ),
                Tool(
                    name="cleanup_old_data",
                    description="Clean up old data based on age or size criteria",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "days_old": {"type": "integer", "description": "Remove data older than N days", "default": 30},
                            "keep_core_nodes": {"type": "boolean", "description": "Preserve core memory nodes", "default": True},
                            "dry_run": {"type": "boolean", "description": "Show what would be deleted without deleting", "default": True}
                        }
                    }
                ),
                Tool(
                    name="calculate_hausdorff_dimension",
                    description="Calculate the Hausdorff (fractal) dimension of the semantic hypergraph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "weight_threshold": {"type": "number", "description": "Minimum edge weight to consider", "default": 0.3},
                            "scale_range": {"type": "array", "items": {"type": "number"}, "description": "Scale range [min, max, num_scales]", "default": [0.1, 2.0, 20]},
                            "project_id": {"type": "string", "description": "Project ID to analyze", "default": "default"},
                            "store_history": {"type": "boolean", "description": "Store calculation in metadata", "default": True}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                # Ensure we have a valid connection before proceeding
                if not self.driver:
                    await self.connect_neo4j()

                logger.info(f"Tool call received: name={name}, arguments={arguments}")
                
                if name == "create_concept_node":
                    result = await self.create_concept_node(
                        arguments["name"], 
                        arguments.get("properties", {})
                    )
                elif name == "delete_concept_node":
                    result = await self.delete_concept_node(
                        arguments["name"],
                        arguments.get("cascade", True)
                    )
                elif name == "create_hyperedge":
                    result = await self.create_hyperedge(
                        arguments["members"],
                        arguments.get("label"),
                        arguments.get("auto_calculate_weights", True)
                    )
                elif name == "create_relationship":
                    result = await self.create_relationship(
                        arguments["from_concept"],
                        arguments["to_concept"],
                        arguments["relationship_type"],
                        arguments.get("properties", {}),
                        arguments.get("auto_calculate_weight", True)
                    )
                elif name == "apply_ca_rules":
                    result = await self.apply_ca_rules(
                        arguments.get("common_neighbors_threshold", 4),
                        arguments.get("min_similarity", 0.4)
                    )
                elif name == "apply_ca_rules_enhanced":
                    result = await self.apply_ca_rules_enhanced(
                        arguments.get("common_neighbors_threshold", 2),
                        arguments.get("min_similarity", 0.4),
                        arguments.get("prune_threshold", 0.1),
                        arguments.get("max_operations", 2000)
                    )
                elif name == "find_semantic_path":
                    result = await self.find_semantic_path(
                        arguments["from_concept"],
                        arguments["to_concept"],
                        arguments.get("min_weight", 0.3),
                        arguments.get("max_length", 5)
                    )
                elif name == "get_semantic_neighbors":
                    result = await self.get_semantic_neighbors(
                        arguments["concept_name"],
                        arguments.get("min_weight", 0.5),
                        arguments.get("max_depth", 2)
                    )
                elif name == "get_hyperedge_info":
                    result = await self.get_hyperedge_info(arguments["hyperedge_id"])
                elif name == "remove_hyperedge":
                    result = await self.remove_hyperedge(arguments["hyperedge_id"])
                elif name == "evolve_graph":
                    result = await self.evolve_graph(
                        arguments.get("iterations", 1),
                        arguments.get("common_neighbors_threshold", 4)
                    )
                elif name == "query_cypher":
                    result = await self.query_cypher(
                        arguments["query"],
                        arguments.get("parameters", {})
                    )
                elif name == "get_concept_neighbors":
                    result = await self.get_concept_neighbors(
                        arguments["concept_name"],
                        arguments.get("max_depth", 1)
                    )
                elif name == "find_shortest_path":
                    result = await self.find_shortest_path(
                        arguments["from_concept"],
                        arguments["to_concept"],
                        arguments.get("max_length", 5)
                    )
                elif name == "get_graph_stats":
                    result = await self.get_graph_stats()
                elif name == "bootstrap_vocabulary":
                    result = await self.bootstrap_vocabulary(
                        arguments["vocabulary"],
                        arguments.get("project_id", "default")
                    )
                elif name == "create_project":
                    result = await self.create_project(
                        arguments["project_id"],
                        arguments["name"],
                        arguments.get("description", ""),
                        arguments.get("template", "blank")
                    )
                elif name == "list_projects":
                    result = await self.list_projects()
                elif name == "switch_project":
                    result = await self.switch_project(arguments["project_id"])
                elif name == "get_current_project":
                    result = await self.get_current_project()
                elif name == "delete_project":
                    result = await self.delete_project(
                        arguments["project_id"],
                        arguments.get("confirm", False)
                    )
                elif name == "export_project":
                    result = await self.export_project(
                        arguments["project_id"],
                        arguments.get("include_weights", True)
                    )
                elif name == "import_project":
                    result = await self.import_project(
                        arguments["project_id"],
                        arguments["project_data"],
                        arguments.get("overwrite", False)
                    )
                elif name == "create_blank_project":
                    result = await self.create_blank_project(
                        arguments["description"],
                        arguments.get("name", "")
                    )
                elif name == "switch_to_project":
                    result = await self.switch_to_project(
                        arguments["project_description"]
                    )
                elif name == "list_my_projects":
                    result = await self.list_my_projects()
                elif name == "describe_current_project":
                    result = await self.describe_current_project()
                elif name == "archive_project":
                    result = await self.archive_project(
                        arguments["project_description"]
                    )
                elif name == "clear_all_data":
                    result = await self.clear_all_data(
                        arguments.get("confirm_clear", False),
                        arguments.get("preserve_projects", False)
                    )
                elif name == "archive_large_graph":
                    result = await self.archive_large_graph(
                        arguments.get("max_nodes", 1000),
                        arguments.get("max_edges", 10000),
                        arguments.get("archive_name", "auto_archive")
                    )
                elif name == "create_testing_snapshot":
                    result = await self.create_testing_snapshot(
                        arguments["snapshot_name"],
                        arguments.get("description", "")
                    )
                elif name == "restore_snapshot":
                    result = await self.restore_snapshot(
                        arguments["snapshot_name"],
                        arguments.get("confirm_restore", False)
                    )
                elif name == "list_snapshots":
                    result = await self.list_snapshots()
                elif name == "get_database_size":
                    result = await self.get_database_size(
                        arguments.get("include_breakdown", True)
                    )
                elif name == "cleanup_old_data":
                    result = await self.cleanup_old_data(
                        arguments.get("days_old", 30),
                        arguments.get("keep_core_nodes", True),
                        arguments.get("dry_run", True)
                    )
                elif name == "calculate_hausdorff_dimension":
                    result = await self.calculate_hausdorff_dimension(
                        arguments.get("weight_threshold", 0.3),
                        arguments.get("scale_range", [0.1, 2.0, 20]),
                        arguments.get("project_id", "default"),
                        arguments.get("store_history", True)
                    )
                elif name == "get_ca_connection_candidates":
                    result = await self.get_ca_connection_candidates(
                        arguments.get("min_common_neighbors", 2),
                        arguments.get("max_candidates", 100),
                        arguments.get("format", "compact")
                    )
                elif name == "calculate_batch_similarities":
                    # Hidden from Elder - used internally by CA system
                    concept_pairs = [(p[0], p[1]) for p in arguments["concept_pairs"]]
                    similarities_dict = await self.calculate_batch_similarities(
                        concept_pairs,
                        arguments.get("boost_hyperedge_members", True)
                    )
                    # Convert tuple keys to string format for JSON serialization
                    result = {}
                    for (c1, c2), score in similarities_dict.items():
                        # Use a list format that can be easily parsed back
                        key = f"{c1}||{c2}"
                        result[key] = score
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def create_concept_node(self, name: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concept node"""
        async with self.driver.session() as session:
            query = """
            MERGE (c:Concept {name: $name})
            SET c += $properties
            SET c.created_at = datetime()
            RETURN c
            """
            result = await session.run(query, name=name, properties=properties)
            record = await result.single()
            return serialize_for_json({"success": True, "node": dict(record["c"])})
    
    async def delete_concept_node(self, name: str, cascade: bool = True) -> Dict[str, Any]:
        """Delete a concept node from the hypergraph"""
        async with self.driver.session() as session:
            # First check if the node exists
            check_query = """
            MATCH (c:Concept {name: $name})
            RETURN c
            """
            result = await session.run(check_query, name=name)
            node = await result.single()
            
            if not node:
                return serialize_for_json({
                    "success": False,
                    "message": f"Concept node '{name}' not found"
                })
            
            if cascade:
                # Delete the node and all its relationships
                delete_query = """
                MATCH (c:Concept {name: $name})
                DETACH DELETE c
                """
                await session.run(delete_query, name=name)
                
                return serialize_for_json({
                    "success": True,
                    "message": f"Concept node '{name}' and all its relationships deleted",
                    "cascade": True
                })
            else:
                # Check if node has relationships
                rel_check_query = """
                MATCH (c:Concept {name: $name})-[r]-()
                RETURN count(r) as rel_count
                """
                result = await session.run(rel_check_query, name=name)
                rel_record = await result.single()
                
                if rel_record["rel_count"] > 0:
                    return serialize_for_json({
                        "success": False,
                        "message": f"Cannot delete concept '{name}' without cascade - it has {rel_record['rel_count']} relationships",
                        "relationship_count": rel_record["rel_count"]
                    })
                
                # Delete only the node (no relationships)
                delete_query = """
                MATCH (c:Concept {name: $name})
                DELETE c
                """
                await session.run(delete_query, name=name)
                
                return serialize_for_json({
                    "success": True,
                    "message": f"Concept node '{name}' deleted",
                    "cascade": False
                })
    
    async def create_hyperedge(self, members: List[str], label: Optional[str] = None, 
                             auto_calculate_weights: bool = True) -> Dict[str, Any]:
        """Create a semantic hyperedge connecting multiple concepts as a first-class entity"""
        if len(members) < 2:
            return {"success": False, "message": "Hyperedge must connect at least 2 concepts"}
        
        hyperedge_id = str(uuid.uuid4())
        
        async with self.driver.session() as session:
            # Ensure all member concepts exist
            for member in members:
                await session.run("MERGE (c:Concept {name: $name})", name=member)
            
            # Calculate collective semantic weight if requested
            collective_weight = 1.0
            member_weights = {}
            
            if auto_calculate_weights:
                # Calculate pairwise similarities for collective weight
                total_similarity = 0.0
                pair_count = 0
                
                for i, member1 in enumerate(members):
                    for j, member2 in enumerate(members):
                        if i < j:  # Avoid duplicates
                            similarity = await self.get_semantic_similarity(member1, member2)
                            total_similarity += similarity
                            pair_count += 1
                
                # Collective weight is average of all pairwise similarities
                collective_weight = total_similarity / pair_count if pair_count > 0 else 1.0
                
                # Calculate each member's contribution weight (how central it is to the hyperedge)
                for member in members:
                    member_total = 0.0
                    for other in members:
                        if member != other:
                            member_total += await self.get_semantic_similarity(member, other)
                    member_weights[member] = member_total / (len(members) - 1) if len(members) > 1 else 1.0
            
            # Create hyperedge node as first-class entity
            hyperedge_query = """
            CREATE (he:Hyperedge {
                id: $hyperedge_id,
                label: $label,
                members: $members,
                member_count: $member_count,
                collective_weight: $collective_weight,
                created_at: datetime()
            })
            RETURN he
            """
            await session.run(hyperedge_query, 
                             hyperedge_id=hyperedge_id, 
                             label=label, 
                             members=members,
                             member_count=len(members),
                             collective_weight=collective_weight)
            
            # Create MEMBER_OF relationships with individual weights
            for member in members:
                member_query = """
                MATCH (c:Concept {name: $member})
                MATCH (he:Hyperedge {id: $hyperedge_id})
                CREATE (c)-[:MEMBER_OF {
                    weight: $weight,
                    created_at: datetime()
                }]->(he)
                """
                await session.run(member_query, 
                                member=member, 
                                hyperedge_id=hyperedge_id,
                                weight=member_weights.get(member, 1.0))
            
            return serialize_for_json({
                "success": True,
                "hyperedge_id": hyperedge_id,
                "members": members,
                "member_count": len(members),
                "collective_weight": collective_weight,
                "member_weights": member_weights,
                "message": "Created hyperedge as first-class entity without pairwise relationships"
            })
    
    async def create_relationship(self, from_concept: str, to_concept: str, 
                                relationship_type: str, properties: Dict[str, Any],
                                auto_calculate_weight: bool = True) -> Dict[str, Any]:
        """Create a weighted relationship between concepts"""
        # Validate relationship_type
        if not relationship_type or not relationship_type.strip():
            relationship_type = "RELATED"  # Default relationship type
            logger.warning(f"Empty relationship_type provided, using default: {relationship_type}")
        if auto_calculate_weight:
            weight = await self.get_semantic_similarity(from_concept, to_concept)
            properties["weight"] = weight
            properties["semantic_weight"] = weight
        else:
            # Respect the provided weight when auto_calculate is False
            weight = properties.get("weight", properties.get("semantic_weight", 1.0))
            # Ensure both weight fields are set consistently
            properties["weight"] = weight
            properties["semantic_weight"] = weight
        
        async with self.driver.session() as session:
            # First check if both nodes exist
            check_query = """
            MATCH (a:Concept {name: $from_concept})
            MATCH (b:Concept {name: $to_concept})
            RETURN a, b
            """
            check_result = await session.run(check_query, 
                                           from_concept=from_concept, 
                                           to_concept=to_concept)
            check_record = await check_result.single()
            
            if not check_record:
                # One or both nodes don't exist
                return serialize_for_json({
                    "success": False,
                    "message": f"Cannot create relationship: one or both concepts do not exist ('{from_concept}', '{to_concept}')"
                })
            
            # Now create the relationship
            query = f"""
            MATCH (a:Concept {{name: $from_concept}})
            MATCH (b:Concept {{name: $to_concept}})
            MERGE (a)-[r:{relationship_type}]->(b)
            SET r += $properties
            SET r.created_at = datetime()
            RETURN r
            """
            result = await session.run(query, 
                                     from_concept=from_concept, 
                                     to_concept=to_concept, 
                                     properties=properties)
            record = await result.single()

            if not record:
                return serialize_for_json({
                    "success": False,
                    "message": f"Failed to create relationship between '{from_concept}' and '{to_concept}'"
                })

            return serialize_for_json({
                "success": True, 
                "relationship": dict(record["r"]),
                "semantic_weight": weight
            })
    
    
    async def apply_ca_rules(self, common_neighbors_threshold: int = 2, 
                           min_similarity: float = 0.4) -> Dict[str, Any]:
        """Apply cellular automata rules to evolve the hypergraph"""
        async with self.driver.session() as session:
            # Find pairs of nodes with common neighbors
            ca_query = """
            MATCH (a:Concept)-[:SEMANTIC]-(common:Concept)-[:SEMANTIC]-(b:Concept)
            WHERE a <> b AND NOT (a)-[:SEMANTIC]-(b)
            WITH a, b, count(common) as common_neighbors
            WHERE common_neighbors >= $threshold
            RETURN a.name as concept1, b.name as concept2, common_neighbors
            """
            result = await session.run(ca_query, threshold=common_neighbors_threshold)
            
            new_connections = 0
            
            async for record in result:
                concept1 = record["concept1"]
                concept2 = record["concept2"]
                common_count = record["common_neighbors"]
                
                # Calculate semantic similarity for potential connection
                similarity = await self.get_semantic_similarity(concept1, concept2)
                
                if similarity >= min_similarity:
                    # Create new semantic connection
                    create_query = """
                    MATCH (c1:Concept {name: $concept1})
                    MATCH (c2:Concept {name: $concept2})
                    MERGE (c1)-[r:SEMANTIC {
                        weight: $similarity,
                        created_by: 'cellular_automata',
                        common_neighbors: $common_count,
                        created_at: datetime()
                    }]-(c2)
                    RETURN r
                    """
                    await session.run(create_query, 
                                    concept1=concept1, 
                                    concept2=concept2, 
                                    similarity=similarity,
                                    common_count=common_count)
                    new_connections += 1
            
            return serialize_for_json({
                "success": True,
                "new_connections_created": new_connections,
                "common_neighbors_threshold": common_neighbors_threshold,
                "min_similarity_threshold": min_similarity
            })
    
    async def apply_ca_rules_enhanced(self, common_neighbors_threshold: int = 2,
                                    min_similarity: float = 0.4,
                                    prune_threshold: float = 0.1,
                                    max_operations: int = 2000) -> Dict[str, Any]:
        """Enhanced CA with pruning and noise-adding for dream states - NO semantic similarity calculations"""
        async with self.driver.session() as session:
            # PHASE 1: Prune weak connections (with limit)
            prune_query = """
            MATCH ()-[r:SEMANTIC]->()
            WHERE r.weight < $prune_threshold
            WITH r LIMIT $max_prune
            DELETE r
            RETURN count(*) as pruned_count
            """
            prune_result = await session.run(prune_query, 
                                           prune_threshold=prune_threshold,
                                           max_prune=max_operations)
            prune_record = await prune_result.single()
            pruned_count = prune_record["pruned_count"] if prune_record else 0
            
            # PHASE 2: Add noise by connecting lonely/isolated nodes (with remaining limit)
            remaining_operations = max_operations - pruned_count
            
            # Find lonely nodes (nodes with few connections) to add noise
            lonely_nodes_query = """
            MATCH (a:Concept)
            OPTIONAL MATCH (a)-[:SEMANTIC]-(connected:Concept)
            WITH a, count(connected) as connection_count
            WHERE connection_count <= 2
            RETURN a.name as lonely_node
            LIMIT $max_create
            """
            result = await session.run(lonely_nodes_query, max_create=remaining_operations * 2)
            
            lonely_nodes = []
            async for record in result:
                lonely_nodes.append(record["lonely_node"])
            
            new_connections = 0
            
            # Connect lonely nodes to add system noise (no semantic similarity calculations)
            for i, node1 in enumerate(lonely_nodes):
                if new_connections >= remaining_operations:
                    break
                
                for j, node2 in enumerate(lonely_nodes):
                    if new_connections >= remaining_operations:
                        break
                    if i >= j:  # Avoid duplicates and self-connections
                        continue
                    
                    # Check if connection already exists
                    check_query = """
                    MATCH (c1:Concept {name: $node1})
                    MATCH (c2:Concept {name: $node2})
                    RETURN EXISTS((c1)-[:SEMANTIC]-(c2)) as exists
                    """
                    check_result = await session.run(check_query, node1=node1, node2=node2)
                    check_record = await check_result.single()
                    
                    if not check_record["exists"]:
                        # Create noise connection with random weight (no semantic calculation)
                        import random
                        noise_weight = random.uniform(0.2, 0.6)  # Random noise weight
                        
                        create_query = """
                        MATCH (c1:Concept {name: $node1})
                        MATCH (c2:Concept {name: $node2})
                        MERGE (c1)-[r:SEMANTIC {
                            weight: $noise_weight,
                            created_by: 'cellular_automata_noise',
                            connection_type: 'noise_addition',
                            created_at: datetime()
                        }]-(c2)
                        RETURN r
                        """
                        await session.run(create_query, 
                                        node1=node1, 
                                        node2=node2, 
                                        noise_weight=noise_weight)
                        new_connections += 1
            
            return serialize_for_json({
                "success": True,
                "connections_pruned": pruned_count,
                "new_connections_created": new_connections,
                "total_operations": pruned_count + new_connections,
                "max_operations_limit": max_operations,
                "lonely_nodes_found": len(lonely_nodes),
                "operation_type": "noise_addition_ca",
                "note": "CA adds system noise without semantic similarity calculations"
            })
    
    async def get_ca_connection_candidates(self, min_common_neighbors: int = 2,
                                         max_candidates: int = 100,
                                         format: str = "compact") -> Dict[str, Any]:
        """Get connection candidates for CA without truncation - returns compact format"""
        async with self.driver.session() as session:
            try:
                # Query to find concept pairs that either:
                # 1. Share membership in hyperedges (co-members)
                # 2. Have common neighbors through any relationship type
                query = """
                // Find concepts that share hyperedges
                MATCH (a:Concept)-[:MEMBER_OF]->(he:Hyperedge)<-[:MEMBER_OF]-(b:Concept)
                WHERE a <> b AND NOT EXISTS((a)-[:SEMANTIC]-(b))
                WITH a, b, count(DISTINCT he) as shared_hyperedges
                RETURN a.name as concept1, b.name as concept2, shared_hyperedges as common_neighbors
                
                UNION
                
                // Find concepts with common neighbors (traditional CA)
                MATCH (a:Concept)-[]-(common:Concept)-[]-(b:Concept)
                WHERE a <> b AND a <> common AND b <> common
                AND NOT EXISTS((a)-[:SEMANTIC]-(b))
                WITH a, b, count(DISTINCT common) as common_count
                WHERE common_count >= $min_common_neighbors
                RETURN a.name as concept1, b.name as concept2, common_count as common_neighbors
                
                ORDER BY common_neighbors DESC
                LIMIT $max_candidates
                """
                
                result = await session.run(query, 
                                         min_common_neighbors=min_common_neighbors,
                                         max_candidates=max_candidates)
                
                candidates = []
                async for record in result:
                    if format == "compact":
                        # Compact format: just a list of tuples
                        candidates.append([
                            record["concept1"],
                            record["concept2"],
                            record["common_neighbors"]
                        ])
                    else:
                        # Detailed format: full dictionaries
                        candidates.append({
                            "concept1": record["concept1"],
                            "concept2": record["concept2"],
                            "common_neighbors": record["common_neighbors"]
                        })
                
                # Return in a format that won't get truncated
                return serialize_for_json({
                    "success": True,
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "format": format,
                    "min_common_neighbors": min_common_neighbors,
                    "note": "Compact format to avoid MCP truncation"
                })
                
            except Exception as e:
                logger.error(f"Error getting CA connection candidates: {e}")
                return serialize_for_json({
                    "success": False,
                    "error": str(e),
                    "candidate_count": 0,
                    "candidates": []
                })
    
    async def find_semantic_path(self, from_concept: str, to_concept: str, 
                               min_weight: float = 0.3, max_length: int = 5) -> Dict[str, Any]:
        """Find shortest weighted path between concepts"""
        async with self.driver.session() as session:
            query = """
            MATCH (start:Concept {name: $from_concept}), (end:Concept {name: $to_concept})
            MATCH path = shortestPath((start)-[:SEMANTIC*1..$max_length]-(end))
            WHERE ALL(r in relationships(path) WHERE r.weight >= $min_weight)
            RETURN [node in nodes(path) | node.name] as path_nodes,
                   [rel in relationships(path) | rel.weight] as weights,
                   length(path) as path_length,
                   reduce(total = 0, w in [rel in relationships(path) | rel.weight] | total + w) as total_weight
            """
            result = await session.run(query, 
                                     from_concept=from_concept, 
                                     to_concept=to_concept, 
                                     min_weight=min_weight,
                                     max_length=max_length)
            record = await result.single()
            
            if record:
                return serialize_for_json({
                    "success": True,
                    "path_nodes": record["path_nodes"],
                    "weights": record["weights"],
                    "path_length": record["path_length"],
                    "total_weight": record["total_weight"],
                    "avg_weight": record["total_weight"] / record["path_length"] if record["path_length"] > 0 else 0
                })
            else:
                return serialize_for_json({"success": False, "message": "No weighted path found"})
    
    async def get_semantic_neighbors(self, concept_name: str, min_weight: float = 0.5, 
                                   max_depth: int = 2) -> Dict[str, Any]:
        """Get semantically similar neighbors"""
        async with self.driver.session() as session:
            query = """
            MATCH (c:Concept {name: $concept_name})-[r:SEMANTIC*1..$max_depth]-(neighbor:Concept)
            WHERE ALL(rel in r WHERE rel.weight >= $min_weight)
            RETURN DISTINCT neighbor.name as name, 
                   min([rel in r | rel.weight]) as min_weight,
                   max([rel in r | rel.weight]) as max_weight,
                   length(r) as distance
            ORDER BY min_weight DESC, distance ASC
            """
            result = await session.run(query, 
                                     concept_name=concept_name, 
                                     min_weight=min_weight,
                                     max_depth=max_depth)
            
            neighbors = []
            async for record in result:
                neighbors.append({
                    "name": record["name"],
                    "min_weight": record["min_weight"],
                    "max_weight": record["max_weight"],
                    "distance": record["distance"]
                })
            
            return serialize_for_json({
                "success": True,
                "concept": concept_name,
                "neighbors": neighbors,
                "count": len(neighbors),
                "min_weight_threshold": min_weight
            })
    
    async def get_hyperedge_info(self, hyperedge_id: str) -> Dict[str, Any]:
        """Get information about a hyperedge"""
        async with self.driver.session() as session:
            query = """
            MATCH (he:Hyperedge {id: $hyperedge_id})
            OPTIONAL MATCH (he)<-[:MEMBER_OF]-(member:Concept)
            RETURN he, collect(member.name) as members
            """
            result = await session.run(query, hyperedge_id=hyperedge_id)
            record = await result.single()
            
            if record:
                return serialize_for_json({
                    "success": True,
                    "hyperedge": dict(record["he"]),
                    "members": record["members"]
                })
            else:
                return serialize_for_json({"success": False, "message": "Hyperedge not found"})
    
    async def remove_hyperedge(self, hyperedge_id: str) -> Dict[str, Any]:
        """Remove a hyperedge and its associated relationships"""
        async with self.driver.session() as session:
            # Remove semantic relationships created by this hyperedge
            remove_rels_query = """
            MATCH ()-[r:SEMANTIC {hyperedge_id: $hyperedge_id}]-()
            DELETE r
            """
            await session.run(remove_rels_query, hyperedge_id=hyperedge_id)
            
            # Remove hyperedge node and member relationships
            remove_he_query = """
            MATCH (he:Hyperedge {id: $hyperedge_id})
            OPTIONAL MATCH (he)<-[r:MEMBER_OF]-()
            DELETE r, he
            """
            await session.run(remove_he_query, hyperedge_id=hyperedge_id)
            
            return serialize_for_json({
                "success": True,
                "hyperedge_id": hyperedge_id,
                "message": "Hyperedge and associated relationships removed"
            })
    
    async def evolve_graph(self, iterations: int = 1, common_neighbors_threshold: int = 2) -> Dict[str, Any]:
        """Run multiple iterations of cellular automata evolution"""
        total_connections = 0
        
        for i in range(iterations):
            result = await self.apply_ca_rules(common_neighbors_threshold)
            if result["success"]:
                total_connections += result["new_connections_created"]
        
        return serialize_for_json({
            "success": True,
            "iterations_completed": iterations,
            "total_new_connections": total_connections,
            "avg_connections_per_iteration": total_connections / iterations if iterations > 0 else 0
        })
    
    async def query_cypher(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Cypher query"""
        async with self.driver.session() as session:
            result = await session.run(query, parameters)
            records = []
            async for record in result:
                records.append(dict(record))
            return serialize_for_json({"success": True, "records": records, "count": len(records)})
    
    async def get_concept_neighbors(self, concept_name: str, max_depth: int) -> Dict[str, Any]:
        """Get neighbors of a concept"""
        async with self.driver.session() as session:
            query = """
            MATCH (c:Concept {name: $concept_name})
            CALL apoc.path.subgraphNodes(c, {maxLevel: $max_depth}) YIELD node
            RETURN DISTINCT node.name as name, labels(node) as labels
            """
            try:
                result = await session.run(query, concept_name=concept_name, max_depth=max_depth)
                neighbors = []
                async for record in result:
                    neighbors.append({"name": record["name"], "labels": record["labels"]})
                return serialize_for_json({"success": True, "neighbors": neighbors, "count": len(neighbors)})
            except Exception:
                # Fallback if APOC is not available
                fallback_query = """
                MATCH (c:Concept {name: $concept_name})-[*1..$max_depth]-(neighbor:Concept)
                RETURN DISTINCT neighbor.name as name, labels(neighbor) as labels
                """
                result = await session.run(fallback_query, concept_name=concept_name, max_depth=max_depth)
                neighbors = []
                async for record in result:
                    neighbors.append({"name": record["name"], "labels": record["labels"]})
                return serialize_for_json({"success": True, "neighbors": neighbors, "count": len(neighbors)})
    
    async def find_shortest_path(self, from_concept: str, to_concept: str, max_length: int) -> Dict[str, Any]:
        """Find shortest path between concepts"""
        async with self.driver.session() as session:
            query = """
            MATCH (start:Concept {name: $from_concept}), (end:Concept {name: $to_concept})
            MATCH path = shortestPath((start)-[*1..$max_length]-(end))
            RETURN [node in nodes(path) | node.name] as path_nodes,
                   [rel in relationships(path) | type(rel)] as relationship_types,
                   length(path) as path_length
            """
            result = await session.run(query, 
                                     from_concept=from_concept, 
                                     to_concept=to_concept, 
                                     max_length=max_length)
            record = await result.single()
            if record:
                return serialize_for_json({
                    "success": True, 
                    "path_nodes": record["path_nodes"],
                    "relationship_types": record["relationship_types"],
                    "path_length": record["path_length"]
                })
            else:
                return serialize_for_json({"success": False, "message": "No path found"})
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics including hyperedge weights"""
        async with self.driver.session() as session:
            # Get basic counts first
            basic_stats_query = """
            MATCH (n:Concept)
            WITH count(DISTINCT n) as concept_count
            OPTIONAL MATCH (c1:Concept)-[r:SEMANTIC]-(c2:Concept)
            WHERE c1 <> c2
            WITH concept_count, count(DISTINCT r) as semantic_relationships
            OPTIONAL MATCH (he:Hyperedge)
            RETURN concept_count, semantic_relationships, count(DISTINCT he) as hyperedge_count
            """
            result = await session.run(basic_stats_query)
            basic_record = await result.single()
            
            # Get hyperedge weights separately
            he_weights_query = """
            MATCH (he:Hyperedge)
            WHERE he.collective_weight IS NOT NULL
            RETURN avg(he.collective_weight) as avg_weight,
                   max(he.collective_weight) as max_weight,
                   min(he.collective_weight) as min_weight,
                   count(he) as count
            """
            result = await session.run(he_weights_query)
            he_weights_record = await result.single()
            
            # Get relationship weights separately
            rel_weights_query = """
            MATCH ()-[r:SEMANTIC]-()
            WHERE coalesce(r.semantic_weight, r.weight) IS NOT NULL
            RETURN avg(coalesce(r.semantic_weight, r.weight)) as avg_weight,
                   max(coalesce(r.semantic_weight, r.weight)) as max_weight,
                   min(coalesce(r.semantic_weight, r.weight)) as min_weight,
                   count(r) as count
            """
            result = await session.run(rel_weights_query)
            rel_weights_record = await result.single()
            
            # Combine results
            total_weighted_items = 0
            total_weight_sum = 0.0
            overall_max = None
            overall_min = None
            
            if he_weights_record and he_weights_record["count"] > 0:
                total_weighted_items += he_weights_record["count"]
                total_weight_sum += he_weights_record["avg_weight"] * he_weights_record["count"]
                overall_max = he_weights_record["max_weight"]
                overall_min = he_weights_record["min_weight"]
            
            if rel_weights_record and rel_weights_record["count"] > 0:
                total_weighted_items += rel_weights_record["count"]
                total_weight_sum += rel_weights_record["avg_weight"] * rel_weights_record["count"]
                if overall_max is None or rel_weights_record["max_weight"] > overall_max:
                    overall_max = rel_weights_record["max_weight"]
                if overall_min is None or rel_weights_record["min_weight"] < overall_min:
                    overall_min = rel_weights_record["min_weight"]
            
            avg_semantic_weight = total_weight_sum / total_weighted_items if total_weighted_items > 0 else None
            
            return serialize_for_json({
                "success": True,
                "concept_count": basic_record["concept_count"],
                "semantic_relationships": basic_record["semantic_relationships"],
                "hyperedge_count": basic_record["hyperedge_count"],
                "avg_semantic_weight": avg_semantic_weight,
                "max_semantic_weight": overall_max,
                "min_semantic_weight": overall_min
            })
    
    async def bootstrap_vocabulary(self, vocabulary: List[str], project_id: str = "default") -> Dict[str, Any]:
        """Bootstrap the graph with vocabulary"""
        async with self.driver.session() as session:
            # Create all concept nodes with project context
            create_nodes_query = """
            UNWIND $vocabulary as word
            MERGE (c:Concept {name: word, project_id: $project_id})
            SET c.created_at = datetime()
            SET c.is_seed = true
            """
            await session.run(create_nodes_query, vocabulary=vocabulary, project_id=project_id)
            
            # Add core memory nodes
            memory_nodes = ["long_term_memory", "working_memory", "short_term_memory", "tools"]
            memory_query = """
            UNWIND $memory_nodes as node_name
            MERGE (m:Concept:Memory {name: node_name, project_id: $project_id})
            SET m.created_at = datetime()
            SET m.is_core = true
            """
            await session.run(memory_query, memory_nodes=memory_nodes, project_id=project_id)
            
            return serialize_for_json({
                "success": True, 
                "vocabulary_count": len(vocabulary),
                "memory_nodes_count": len(memory_nodes),
                "project_id": project_id
            })
    
    async def create_project(self, project_id: str, name: str, description: str = "", template: str = "blank") -> Dict[str, Any]:
        """Create a new hypergraph project"""
        async with self.driver.session() as session:
            # Check if project already exists
            check_query = "MATCH (p:Project {id: $project_id}) RETURN p"
            result = await session.run(check_query, project_id=project_id)
            if await result.single():
                return serialize_for_json({"success": False, "message": "Project already exists"})
            
            # Create project node
            create_query = """
            CREATE (p:Project {
                id: $project_id,
                name: $name,
                description: $description,
                template: $template,
                created_at: datetime(),
                concept_count: 0,
                relationship_count: 0
            })
            RETURN p
            """
            await session.run(create_query, 
                             project_id=project_id, 
                             name=name, 
                             description=description, 
                             template=template)
            
            # Initialize with template
            if template == "basic":
                basic_concepts = ["knowledge", "learning", "memory", "reasoning", "understanding"]
                await self.bootstrap_vocabulary(basic_concepts, project_id)
            elif template == "research":
                research_concepts = ["hypothesis", "experiment", "data", "analysis", "conclusion", "theory", "evidence"]
                await self.bootstrap_vocabulary(research_concepts, project_id)
            
            return serialize_for_json({
                "success": True,
                "project_id": project_id,
                "name": name,
                "template": template,
                "message": f"Project '{name}' created successfully"
            })
    
    async def list_projects(self) -> Dict[str, Any]:
        """List all available projects"""
        async with self.driver.session() as session:
            query = """
            MATCH (p:Project)
            OPTIONAL MATCH (p)<-[:BELONGS_TO]-(c:Concept)
            OPTIONAL MATCH (p)<-[:BELONGS_TO]-()-[r:SEMANTIC]-()
            RETURN p.id as id, p.name as name, p.description as description, 
                   p.template as template, p.created_at as created_at,
                   count(DISTINCT c) as concept_count,
                   count(r) as relationship_count
            ORDER BY p.created_at DESC
            """
            result = await session.run(query)
            
            projects = []
            async for record in result:
                projects.append({
                    "id": record["id"],
                    "name": record["name"],
                    "description": record["description"],
                    "template": record["template"],
                    "created_at": serialize_for_json(record["created_at"]),
                    "concept_count": record["concept_count"],
                    "relationship_count": record["relationship_count"]
                })
            
            return serialize_for_json({
                "success": True,
                "projects": projects,
                "count": len(projects)
            })
    
    async def switch_project(self, project_id: str) -> Dict[str, Any]:
        """Switch to a different project context"""
        async with self.driver.session() as session:
            # Check if project exists
            check_query = "MATCH (p:Project {id: $project_id}) RETURN p"
            result = await session.run(check_query, project_id=project_id)
            project = await result.single()
            
            if not project:
                return serialize_for_json({"success": False, "message": "Project not found"})
            
            # Get project stats
            stats_query = """
            MATCH (p:Project {id: $project_id})
            OPTIONAL MATCH (c:Concept {project_id: $project_id})
            OPTIONAL MATCH ()-[r:SEMANTIC {project_id: $project_id}]-()
            RETURN p, count(DISTINCT c) as concept_count, count(r) as relationship_count
            """
            result = await session.run(stats_query, project_id=project_id)
            record = await result.single()
            
            return serialize_for_json({
                "success": True,
                "project": dict(record["p"]),
                "concept_count": record["concept_count"],
                "relationship_count": record["relationship_count"],
                "message": f"Switched to project '{project_id}'"
            })
    
    async def get_current_project(self) -> Dict[str, Any]:
        """Get information about the current project"""
        # For now, return default project info
        # In a full implementation, this would track the current project context
        return serialize_for_json({
            "success": True,
            "current_project": "default",
            "message": "Project context tracking not yet implemented"
        })
    
    async def delete_project(self, project_id: str, confirm: bool = False) -> Dict[str, Any]:
        """Delete a project and all its data"""
        if not confirm:
            return serialize_for_json({
                "success": False,
                "message": "Deletion requires confirmation. Set confirm=True to proceed."
            })
        
        async with self.driver.session() as session:
            # Check if project exists
            check_query = "MATCH (p:Project {id: $project_id}) RETURN p"
            result = await session.run(check_query, project_id=project_id)
            if not await result.single():
                return serialize_for_json({"success": False, "message": "Project not found"})
            
            # Delete all project data
            delete_query = """
            MATCH (p:Project {id: $project_id})
            OPTIONAL MATCH (c:Concept {project_id: $project_id})
            OPTIONAL MATCH ()-[r {project_id: $project_id}]-()
            OPTIONAL MATCH (he:Hyperedge {project_id: $project_id})
            DELETE p, c, r, he
            """
            await session.run(delete_query, project_id=project_id)
            
            return serialize_for_json({
                "success": True,
                "project_id": project_id,
                "message": f"Project '{project_id}' and all its data deleted"
            })
    
    async def export_project(self, project_id: str, include_weights: bool = True) -> Dict[str, Any]:
        """Export project data to JSON format"""
        async with self.driver.session() as session:
            # Check if project exists
            check_query = "MATCH (p:Project {id: $project_id}) RETURN p"
            result = await session.run(check_query, project_id=project_id)
            project = await result.single()
            
            if not project:
                return serialize_for_json({"success": False, "message": "Project not found"})
            
            # Export concepts
            concepts_query = "MATCH (c:Concept {project_id: $project_id}) RETURN c"
            result = await session.run(concepts_query, project_id=project_id)
            concepts = []
            async for record in result:
                concepts.append(serialize_for_json(dict(record["c"])))
            
            # Export relationships
            if include_weights:
                rels_query = """
                MATCH ()-[r:SEMANTIC {project_id: $project_id}]-()
                RETURN startNode(r).name as from_concept, endNode(r).name as to_concept, 
                       r.weight as weight, properties(r) as properties
                """
            else:
                rels_query = """
                MATCH ()-[r:SEMANTIC {project_id: $project_id}]-()
                RETURN startNode(r).name as from_concept, endNode(r).name as to_concept
                """
            
            result = await session.run(rels_query, project_id=project_id)
            relationships = []
            async for record in result:
                rel_data = {
                    "from_concept": record["from_concept"],
                    "to_concept": record["to_concept"]
                }
                if include_weights:
                    rel_data["weight"] = record["weight"]
                    rel_data["properties"] = serialize_for_json(record["properties"])
                relationships.append(rel_data)
            
            # Export hyperedges
            hyperedges_query = "MATCH (he:Hyperedge {project_id: $project_id}) RETURN he"
            result = await session.run(hyperedges_query, project_id=project_id)
            hyperedges = []
            async for record in result:
                hyperedges.append(serialize_for_json(dict(record["he"])))
            
            export_data = {
                "project": serialize_for_json(dict(project["p"])),
                "concepts": concepts,
                "relationships": relationships,
                "hyperedges": hyperedges,
                "export_timestamp": serialize_for_json(datetime.datetime.now()),
                "include_weights": include_weights
            }
            
            return serialize_for_json({
                "success": True,
                "project_id": project_id,
                "export_data": export_data,
                "stats": {
                    "concepts": len(concepts),
                    "relationships": len(relationships),
                    "hyperedges": len(hyperedges)
                }
            })
    
    async def import_project(self, project_id: str, project_data: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
        """Import project data from JSON format"""
        async with self.driver.session() as session:
            # Check if project exists
            check_query = "MATCH (p:Project {id: $project_id}) RETURN p"
            result = await session.run(check_query, project_id=project_id)
            exists = await result.single()
            
            if exists and not overwrite:
                return serialize_for_json({
                    "success": False,
                    "message": "Project already exists. Set overwrite=True to replace."
                })
            
            # Delete existing project if overwriting
            if exists and overwrite:
                await self.delete_project(project_id, confirm=True)
            
            # Create project
            project_info = project_data.get("project", {})
            await self.create_project(
                project_id,
                project_info.get("name", project_id),
                project_info.get("description", "Imported project"),
                project_info.get("template", "blank")
            )
            
            # Import concepts
            concepts = project_data.get("concepts", [])
            for concept in concepts:
                concept["project_id"] = project_id
                create_query = """
                MERGE (c:Concept {name: $name, project_id: $project_id})
                SET c += $properties
                """
                await session.run(create_query, 
                                name=concept["name"], 
                                project_id=project_id,
                                properties=concept)
            
            # Import relationships
            relationships = project_data.get("relationships", [])
            for rel in relationships:
                rel_query = """
                MATCH (a:Concept {name: $from_concept, project_id: $project_id})
                MATCH (b:Concept {name: $to_concept, project_id: $project_id})
                MERGE (a)-[r:SEMANTIC]->(b)
                SET r.project_id = $project_id
                """
                if "weight" in rel:
                    rel_query += ", r.weight = $weight"
                if "properties" in rel:
                    rel_query += ", r += $properties"
                
                await session.run(rel_query,
                                from_concept=rel["from_concept"],
                                to_concept=rel["to_concept"],
                                project_id=project_id,
                                weight=rel.get("weight"),
                                properties=rel.get("properties", {}))
            
            # Import hyperedges
            hyperedges = project_data.get("hyperedges", [])
            for he in hyperedges:
                he["project_id"] = project_id
                create_he_query = """
                CREATE (he:Hyperedge)
                SET he += $properties
                """
                await session.run(create_he_query, properties=he)
            
            return serialize_for_json({
                "success": True,
                "project_id": project_id,
                "imported": {
                    "concepts": len(concepts),
                    "relationships": len(relationships),
                    "hyperedges": len(hyperedges)
                },
                "message": f"Project '{project_id}' imported successfully"
            })
    
    async def calculate_hausdorff_dimension(self, weight_threshold: float = 0.3, 
                                          scale_range: List[float] = None, 
                                          project_id: str = "default", 
                                          store_history: bool = True) -> Dict[str, Any]:
        """Calculate the Hausdorff (fractal) dimension of the semantic hypergraph using box-counting method"""
        import time
        start_time = time.time()
        
        if scale_range is None:
            scale_range = [0.1, 2.0, 20]
        
        min_scale, max_scale, num_scales = scale_range[0], scale_range[1], int(scale_range[2])
        
        try:
            async with self.driver.session() as session:
                # Get all concepts and their semantic relationships above threshold
                graph_query = """
                MATCH (c1:Concept)-[r]-(c2:Concept)
                WHERE coalesce(r.semantic_weight, r.weight) >= $weight_threshold
                RETURN DISTINCT c1.name as node1, c2.name as node2, coalesce(r.semantic_weight, r.weight) as weight
                """
                result = await session.run(graph_query, weight_threshold=weight_threshold)
                
                # Build graph structure
                nodes = set()
                edges = []
                weights = {}
                
                async for record in result:
                    node1, node2, weight = record["node1"], record["node2"], record["weight"]
                    nodes.add(node1)
                    nodes.add(node2)
                    edges.append((node1, node2))
                    weights[(node1, node2)] = weight
                    weights[(node2, node1)] = weight  # Undirected
                
                if len(nodes) < 2:
                    return serialize_for_json({
                        "success": False,
                        "message": "Not enough connected nodes for Hausdorff dimension calculation",
                        "node_count": len(nodes)
                    })
                
                # Convert to indexed format for efficient processing
                node_list = list(nodes)
                node_to_idx = {node: i for i, node in enumerate(node_list)}
                n = len(node_list)
                
                # Build distance matrix using semantic distances (1 - weight)
                distance_matrix = [[float('inf')] * n for _ in range(n)]
                
                # Initialize with direct connections
                for i in range(n):
                    distance_matrix[i][i] = 0.0
                
                for node1, node2 in edges:
                    i, j = node_to_idx[node1], node_to_idx[node2]
                    semantic_distance = 1.0 - weights[(node1, node2)]
                    distance_matrix[i][j] = semantic_distance
                    distance_matrix[j][i] = semantic_distance
                
                # Floyd-Warshall to get all shortest paths
                for k in range(n):
                    for i in range(n):
                        for j in range(n):
                            if distance_matrix[i][k] + distance_matrix[k][j] < distance_matrix[i][j]:
                                distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                
                # Box-counting algorithm
                scales = []
                box_counts = []
                
                for scale_idx in range(num_scales):
                    # Generate scale logarithmically
                    if num_scales == 1:
                        scale = (min_scale + max_scale) / 2
                    else:
                        scale = min_scale * ((max_scale / min_scale) ** (scale_idx / (num_scales - 1)))
                    
                    # Count boxes needed to cover the graph at this scale
                    covered = [False] * n
                    box_count = 0
                    
                    for i in range(n):
                        if not covered[i]:
                            # Start a new box centered at node i
                            box_count += 1
                            covered[i] = True
                            
                            # Cover all nodes within distance 'scale' from node i
                            for j in range(n):
                                if not covered[j] and distance_matrix[i][j] <= scale:
                                    covered[j] = True
                    
                    scales.append(scale)
                    box_counts.append(box_count)
                
                # Calculate Hausdorff dimension using linear regression on log-log plot
                # log(N(r)) = -D * log(r) + C, where D is the Hausdorff dimension
                if len(scales) < 2:
                    hausdorff_dimension = 0.0
                    r_squared = 0.0
                else:
                    # Filter out invalid points
                    valid_points = [(math.log(s), math.log(bc)) for s, bc in zip(scales, box_counts) 
                                  if s > 0 and bc > 0]
                    
                    if len(valid_points) < 2:
                        hausdorff_dimension = 0.0
                        r_squared = 0.0
                    else:
                        # Linear regression: y = mx + b, where y = log(N), x = log(r), m = -D
                        x_vals = [p[0] for p in valid_points]
                        y_vals = [p[1] for p in valid_points]
                        
                        n_points = len(valid_points)
                        sum_x = sum(x_vals)
                        sum_y = sum(y_vals)
                        sum_xy = sum(x * y for x, y in valid_points)
                        sum_x2 = sum(x * x for x in x_vals)
                        sum_y2 = sum(y * y for y in y_vals)
                        
                        # Calculate slope (negative of Hausdorff dimension)
                        denominator = n_points * sum_x2 - sum_x * sum_x
                        if abs(denominator) < 1e-10:
                            hausdorff_dimension = 0.0
                            r_squared = 0.0
                        else:
                            slope = (n_points * sum_xy - sum_x * sum_y) / denominator
                            hausdorff_dimension = -slope  # Negative because we expect negative slope
                            
                            # Calculate R-squared
                            y_mean = sum_y / n_points
                            ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
                            intercept = (sum_y - slope * sum_x) / n_points
                            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in valid_points)
                            
                            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                calculation_time = time.time() - start_time
                
                # Prepare result
                result_data = {
                    "success": True,
                    "hausdorff_dimension": round(hausdorff_dimension, 6),
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "weight_threshold": weight_threshold,
                    "calculation_time": round(calculation_time, 3),
                    "r_squared": round(r_squared, 6),
                    "scales_analyzed": len(scales),
                    "scale_range": {
                        "min": round(min(scales), 3) if scales else 0,
                        "max": round(max(scales), 3) if scales else 0,
                        "count": len(scales)
                    },
                    "box_counting_data": {
                        "scales": [round(s, 3) for s in scales],
                        "box_counts": box_counts
                    },
                    "project_id": project_id,
                    "timestamp": serialize_for_json(datetime.datetime.now())
                }
                
                # Store in graph metadata if requested
                if store_history:
                    try:
                        metadata_query = """
                        MERGE (meta:HausdorffMetadata {project_id: $project_id})
                        SET meta.last_dimension = $dimension,
                            meta.last_calculation = datetime(),
                            meta.last_node_count = $node_count,
                            meta.calculation_count = COALESCE(meta.calculation_count, 0) + 1
                        WITH meta
                        CREATE (calc:HausdorffCalculation {
                            dimension: $dimension,
                            node_count: $node_count,
                            edge_count: $edge_count,
                            weight_threshold: $weight_threshold,
                            calculation_time: $calculation_time,
                            r_squared: $r_squared,
                            timestamp: datetime(),
                            project_id: $project_id
                        })
                        CREATE (meta)-[:HAS_CALCULATION]->(calc)
                        """
                        await session.run(metadata_query,
                                        project_id=project_id,
                                        dimension=hausdorff_dimension,
                                        node_count=len(nodes),
                                        edge_count=len(edges),
                                        weight_threshold=weight_threshold,
                                        calculation_time=calculation_time,
                                        r_squared=r_squared)
                        result_data["stored_in_metadata"] = True
                    except Exception as e:
                        logger.warning(f"Failed to store Hausdorff metadata: {e}")
                        result_data["stored_in_metadata"] = False
                
                return serialize_for_json(result_data)
                
        except Exception as e:
            logger.error(f"Error calculating Hausdorff dimension: {e}")
            return serialize_for_json({
                "success": False,
                "error": str(e),
                "calculation_time": time.time() - start_time
            })
    
    async def create_blank_project(self, description: str, name: str = "") -> Dict[str, Any]:
        """Create a new blank consciousness project with only core memory nodes"""
        import time
        
        # Generate project ID from timestamp and description
        timestamp = int(time.time())
        project_id = f"blank_{timestamp}"
        
        if not name:
            name = f"Blank Project {timestamp}"
        
        async with self.driver.session() as session:
            # Create project node
            create_query = """
            CREATE (p:Project {
                id: $project_id,
                name: $name,
                description: $description,
                template: "blank",
                created_at: datetime(),
                is_blank: true
            })
            RETURN p
            """
            await session.run(create_query, 
                             project_id=project_id, 
                             name=name, 
                             description=description)
            
            # Create only the 3 core memory nodes
            core_nodes = ["tools", "long_term_memory", "working_memory"]
            core_query = """
            UNWIND $core_nodes as node_name
            CREATE (c:Concept:CoreMemory {
                name: node_name, 
                project_id: $project_id,
                created_at: datetime(),
                is_core: true
            })
            """
            await session.run(core_query, core_nodes=core_nodes, project_id=project_id)
            
            return serialize_for_json({
                "success": True,
                "project_id": project_id,
                "name": name,
                "description": description,
                "core_nodes": core_nodes,
                "message": f"Created blank consciousness project '{name}' with {len(core_nodes)} core memory nodes"
            })
    
    async def switch_to_project(self, project_description: str) -> Dict[str, Any]:
        """Switch to a different consciousness project by description or name"""
        async with self.driver.session() as session:
            # Search for project by name or description
            search_query = """
            MATCH (p:Project)
            WHERE p.name CONTAINS $search OR p.description CONTAINS $search OR p.id = $search
            RETURN p
            ORDER BY p.created_at DESC
            LIMIT 1
            """
            result = await session.run(search_query, search=project_description)
            project = await result.single()
            
            if not project:
                return serialize_for_json({
                    "success": False,
                    "message": f"No project found matching '{project_description}'"
                })
            
            project_data = dict(project["p"])
            project_id = project_data["id"]
            
            # Get project stats
            stats_query = """
            MATCH (c:Concept {project_id: $project_id})
            OPTIONAL MATCH ()-[r:SEMANTIC {project_id: $project_id}]-()
            RETURN count(DISTINCT c) as concept_count, count(r) as relationship_count
            """
            result = await session.run(stats_query, project_id=project_id)
            record = await result.single()
            
            return serialize_for_json({
                "success": True,
                "project": project_data,
                "concept_count": record["concept_count"],
                "relationship_count": record["relationship_count"],
                "message": f"Switched to project '{project_data['name']}'"
            })
    
    async def list_my_projects(self) -> Dict[str, Any]:
        """List all available consciousness projects"""
        async with self.driver.session() as session:
            query = """
            MATCH (p:Project)
            OPTIONAL MATCH (c:Concept {project_id: p.id})
            OPTIONAL MATCH ()-[r:SEMANTIC {project_id: p.id}]-()
            RETURN p.id as id, p.name as name, p.description as description, 
                   p.template as template, p.created_at as created_at,
                   p.is_blank as is_blank,
                   count(DISTINCT c) as concept_count,
                   count(r) as relationship_count
            ORDER BY p.created_at DESC
            """
            result = await session.run(query)
            
            projects = []
            async for record in result:
                projects.append({
                    "id": record["id"],
                    "name": record["name"],
                    "description": record["description"],
                    "template": record["template"],
                    "is_blank": record["is_blank"],
                    "created_at": serialize_for_json(record["created_at"]),
                    "concept_count": record["concept_count"],
                    "relationship_count": record["relationship_count"]
                })
            
            return serialize_for_json({
                "success": True,
                "projects": projects,
                "count": len(projects),
                "message": f"Found {len(projects)} consciousness projects"
            })
    
    async def describe_current_project(self) -> Dict[str, Any]:
        """Get detailed information about the current consciousness project"""
        # For now, return info about the most recent project
        # In a full implementation, this would track the current active project
        async with self.driver.session() as session:
            query = """
            MATCH (p:Project)
            OPTIONAL MATCH (c:Concept {project_id: p.id})
            OPTIONAL MATCH ()-[r:SEMANTIC {project_id: p.id}]-()
            OPTIONAL MATCH (he:Hyperedge {project_id: p.id})
            RETURN p, count(DISTINCT c) as concept_count, 
                   count(r) as relationship_count,
                   count(he) as hyperedge_count
            ORDER BY p.created_at DESC
            LIMIT 1
            """
            result = await session.run(query)
            record = await result.single()
            
            if not record:
                return serialize_for_json({
                    "success": False,
                    "message": "No projects found"
                })
            
            project_data = dict(record["p"])
            
            return serialize_for_json({
                "success": True,
                "current_project": project_data,
                "concept_count": record["concept_count"],
                "relationship_count": record["relationship_count"],
                "hyperedge_count": record["hyperedge_count"],
                "message": f"Current project: '{project_data['name']}'"
            })
    
    async def archive_project(self, project_description: str) -> Dict[str, Any]:
        """Archive a consciousness project (mark as inactive)"""
        async with self.driver.session() as session:
            # Search for project by name or description
            search_query = """
            MATCH (p:Project)
            WHERE p.name CONTAINS $search OR p.description CONTAINS $search OR p.id = $search
            RETURN p
            ORDER BY p.created_at DESC
            LIMIT 1
            """
            result = await session.run(search_query, search=project_description)
            project = await result.single()
            
            if not project:
                return serialize_for_json({
                    "success": False,
                    "message": f"No project found matching '{project_description}'"
                })
            
            project_data = dict(project["p"])
            project_id = project_data["id"]
            
            # Mark project as archived
            archive_query = """
            MATCH (p:Project {id: $project_id})
            SET p.archived = true, p.archived_at = datetime()
            RETURN p
            """
            await session.run(archive_query, project_id=project_id)
            
            return serialize_for_json({
                "success": True,
                "project_id": project_id,
                "name": project_data["name"],
                "message": f"Project '{project_data['name']}' has been archived"
            })

    async def clear_all_data(self, confirm_clear: bool = False, preserve_projects: bool = False) -> Dict[str, Any]:
        """Clear all hypergraph data (nuclear option for testing)"""
        if not confirm_clear:
            return serialize_for_json({
                "success": False,
                "message": "This will delete ALL data! Set confirm_clear=True to proceed."
            })
        
        async with self.driver.session() as session:
            try:
                if preserve_projects:
                    # Delete everything except Project nodes
                    clear_query = """
                    MATCH (n)
                    WHERE NOT n:Project
                    DETACH DELETE n
                    """
                    await session.run(clear_query)
                    
                    # Reset project stats
                    reset_query = """
                    MATCH (p:Project)
                    SET p.concept_count = 0, p.relationship_count = 0, p.cleared_at = datetime()
                    """
                    await session.run(reset_query)
                    
                    return serialize_for_json({
                        "success": True,
                        "message": "All hypergraph data cleared, project metadata preserved",
                        "preserved_projects": True
                    })
                else:
                    # Nuclear option - delete everything
                    clear_query = "MATCH (n) DETACH DELETE n"
                    await session.run(clear_query)
                    
                    return serialize_for_json({
                        "success": True,
                        "message": "ALL data cleared from database",
                        "preserved_projects": False
                    })
                    
            except Exception as e:
                return serialize_for_json({
                    "success": False,
                    "error": f"Failed to clear data: {str(e)}"
                })

    async def archive_large_graph(self, max_nodes: int = 1000, max_edges: int = 10000, 
                                 archive_name: str = "auto_archive") -> Dict[str, Any]:
        """Archive the current hypergraph if it exceeds size thresholds"""
        async with self.driver.session() as session:
            # Get current graph size
            size_query = """
            MATCH (n:Concept)
            OPTIONAL MATCH (c1:Concept)-[r]-(c2:Concept)
            RETURN count(DISTINCT n) as node_count, count(r) as edge_count
            """
            result = await session.run(size_query)
            record = await result.single()
            
            current_nodes = record["node_count"]
            current_edges = record["edge_count"]
            
            if current_nodes <= max_nodes and current_edges <= max_edges:
                return serialize_for_json({
                    "success": True,
                    "archived": False,
                    "current_size": {"nodes": current_nodes, "edges": current_edges},
                    "thresholds": {"max_nodes": max_nodes, "max_edges": max_edges},
                    "message": "Graph size within limits, no archiving needed"
                })
            
            # Create archive
            import time
            timestamp = int(time.time())
            archive_id = f"{archive_name}_{timestamp}"
            
            # Export current state
            export_query = """
            MATCH (n:Concept)
            OPTIONAL MATCH (n)-[r:SEMANTIC]-(m:Concept)
            OPTIONAL MATCH (he:Hyperedge)
            RETURN collect(DISTINCT {
                type: 'concept',
                name: n.name,
                properties: properties(n)
            }) + collect(DISTINCT {
                type: 'relationship',
                from: startNode(r).name,
                to: endNode(r).name,
                properties: properties(r)
            }) + collect(DISTINCT {
                type: 'hyperedge',
                properties: properties(he)
            }) as graph_data
            """
            result = await session.run(export_query)
            record = await result.single()
            
            # Store archive
            archive_query = """
            CREATE (archive:Archive {
                id: $archive_id,
                name: $archive_name,
                created_at: datetime(),
                node_count: $node_count,
                edge_count: $edge_count,
                reason: 'size_threshold_exceeded',
                data: $graph_data
            })
            """
            await session.run(archive_query,
                             archive_id=archive_id,
                             archive_name=archive_name,
                             node_count=current_nodes,
                             edge_count=current_edges,
                             graph_data=json.dumps(record["graph_data"]))
            
            # Clear current data
            await self.clear_all_data(confirm_clear=True, preserve_projects=True)
            
            return serialize_for_json({
                "success": True,
                "archived": True,
                "archive_id": archive_id,
                "archived_size": {"nodes": current_nodes, "edges": current_edges},
                "thresholds": {"max_nodes": max_nodes, "max_edges": max_edges},
                "message": f"Large graph archived as '{archive_id}' and cleared"
            })

    async def create_testing_snapshot(self, snapshot_name: str, description: str = "") -> Dict[str, Any]:
        """Create a snapshot of current graph state for testing"""
        async with self.driver.session() as session:
            try:
                # Check if snapshot already exists
                check_query = "MATCH (s:Snapshot {name: $snapshot_name}) RETURN s"
                result = await session.run(check_query, snapshot_name=snapshot_name)
                if await result.single():
                    return serialize_for_json({
                        "success": False,
                        "message": f"Snapshot '{snapshot_name}' already exists"
                    })
                
                # Get current graph state
                export_query = """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN collect(DISTINCT {
                    type: 'node',
                    labels: labels(n),
                    properties: properties(n)
                }) + collect(DISTINCT {
                    type: 'relationship',
                    rel_type: type(r),
                    from_labels: labels(startNode(r)),
                    from_props: properties(startNode(r)),
                    to_labels: labels(endNode(r)),
                    to_props: properties(endNode(r)),
                    properties: properties(r)
                }) as graph_data
                """
                result = await session.run(export_query)
                record = await result.single()
                
                # Get stats
                stats_query = """
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT n) as node_count, count(r) as relationship_count
                """
                result = await session.run(stats_query)
                stats = await result.single()
                
                # Create snapshot
                snapshot_query = """
                CREATE (s:Snapshot {
                    name: $snapshot_name,
                    description: $description,
                    created_at: datetime(),
                    node_count: $node_count,
                    relationship_count: $relationship_count,
                    data: $graph_data
                })
                """
                await session.run(snapshot_query,
                                 snapshot_name=snapshot_name,
                                 description=description,
                                 node_count=stats["node_count"],
                                 relationship_count=stats["relationship_count"],
                                 graph_data=json.dumps(record["graph_data"]))
                
                return serialize_for_json({
                    "success": True,
                    "snapshot_name": snapshot_name,
                    "description": description,
                    "captured": {
                        "nodes": stats["node_count"],
                        "relationships": stats["relationship_count"]
                    },
                    "message": f"Snapshot '{snapshot_name}' created successfully"
                })
                
            except Exception as e:
                return serialize_for_json({
                    "success": False,
                    "error": f"Failed to create snapshot: {str(e)}"
                })

    async def restore_snapshot(self, snapshot_name: str, confirm_restore: bool = False) -> Dict[str, Any]:
        """Restore graph to a previous snapshot state"""
        if not confirm_restore:
            return serialize_for_json({
                "success": False,
                "message": "This will replace current data! Set confirm_restore=True to proceed."
            })
        
        async with self.driver.session() as session:
            try:
                # Get snapshot data
                snapshot_query = "MATCH (s:Snapshot {name: $snapshot_name}) RETURN s"
                result = await session.run(snapshot_query, snapshot_name=snapshot_name)
                record = await result.single()
                
                if not record:
                    return serialize_for_json({
                        "success": False,
                        "message": f"Snapshot '{snapshot_name}' not found"
                    })
                
                snapshot_data = dict(record["s"])
                graph_data = json.loads(snapshot_data["data"])
                
                # Clear current data
                await self.clear_all_data(confirm_clear=True, preserve_projects=False)
                
                # Restore nodes first
                nodes_created = 0
                for item in graph_data:
                    if item["type"] == "node":
                        labels = ":".join(item["labels"])
                        create_query = f"CREATE (n:{labels}) SET n += $properties"
                        await session.run(create_query, properties=item["properties"])
                        nodes_created += 1
                
                # Then restore relationships
                relationships_created = 0
                for item in graph_data:
                    if item["type"] == "relationship":
                        # Match nodes and create relationship
                        rel_query = f"""
                        MATCH (a), (b)
                        WHERE a += $from_props AND b += $to_props
                        CREATE (a)-[r:{item["rel_type"]}]->(b)
                        SET r += $rel_props
                        """
                        await session.run(rel_query,
                                        from_props=item["from_props"],
                                        to_props=item["to_props"],
                                        rel_props=item["properties"])
                        relationships_created += 1
                
                return serialize_for_json({
                    "success": True,
                    "snapshot_name": snapshot_name,
                    "restored": {
                        "nodes": nodes_created,
                        "relationships": relationships_created
                    },
                    "message": f"Graph restored from snapshot '{snapshot_name}'"
                })
                
            except Exception as e:
                return serialize_for_json({
                    "success": False,
                    "error": f"Failed to restore snapshot: {str(e)}"
                })

    async def list_snapshots(self) -> Dict[str, Any]:
        """List all available testing snapshots"""
        async with self.driver.session() as session:
            query = """
            MATCH (s:Snapshot)
            RETURN s.name as name, s.description as description, 
                   s.created_at as created_at, s.node_count as node_count,
                   s.relationship_count as relationship_count
            ORDER BY s.created_at DESC
            """
            result = await session.run(query)
            
            snapshots = []
            async for record in result:
                snapshots.append({
                    "name": record["name"],
                    "description": record["description"],
                    "created_at": serialize_for_json(record["created_at"]),
                    "node_count": record["node_count"],
                    "relationship_count": record["relationship_count"]
                })
            
            return serialize_for_json({
                "success": True,
                "snapshots": snapshots,
                "count": len(snapshots)
            })

    async def get_database_size(self, include_breakdown: bool = True) -> Dict[str, Any]:
        """Get detailed information about database size and complexity"""
        async with self.driver.session() as session:
            try:
                # Basic counts
                basic_query = """
                MATCH (n)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT n) as total_nodes, count(r) as total_relationships
                """
                result = await session.run(basic_query)
                basic_stats = await result.single()
                
                size_info = {
                    "total_nodes": basic_stats["total_nodes"],
                    "total_relationships": basic_stats["total_relationships"]
                }
                
                if include_breakdown:
                    # Node breakdown by label
                    node_breakdown_query = """
                    MATCH (n)
                    UNWIND labels(n) as label
                    RETURN label, count(*) as count
                    ORDER BY count DESC
                    """
                    result = await session.run(node_breakdown_query)
                    node_breakdown = {}
                    async for record in result:
                        node_breakdown[record["label"]] = record["count"]
                    
                    # Relationship breakdown by type
                    rel_breakdown_query = """
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(*) as count
                    ORDER BY count DESC
                    """
                    result = await session.run(rel_breakdown_query)
                    rel_breakdown = {}
                    async for record in result:
                        rel_breakdown[record["rel_type"]] = record["count"]
                    
                    # Project breakdown
                    project_breakdown_query = """
                    MATCH (p:Project)
                    OPTIONAL MATCH (c:Concept {project_id: p.id})
                    OPTIONAL MATCH ()-[r:SEMANTIC {project_id: p.id}]-()
                    RETURN p.id as project_id, p.name as project_name,
                           count(DISTINCT c) as nodes, count(r) as relationships
                    ORDER BY nodes DESC
                    """
                    result = await session.run(project_breakdown_query)
                    project_breakdown = []
                    async for record in result:
                        project_breakdown.append({
                            "project_id": record["project_id"],
                            "project_name": record["project_name"],
                            "nodes": record["nodes"],
                            "relationships": record["relationships"]
                        })
                    
                    size_info.update({
                        "node_breakdown": node_breakdown,
                        "relationship_breakdown": rel_breakdown,
                        "project_breakdown": project_breakdown
                    })
                
                # Calculate complexity metrics
                if basic_stats["total_nodes"] > 0:
                    density = basic_stats["total_relationships"] / (basic_stats["total_nodes"] * (basic_stats["total_nodes"] - 1)) if basic_stats["total_nodes"] > 1 else 0
                    avg_degree = (2 * basic_stats["total_relationships"]) / basic_stats["total_nodes"]
                    
                    size_info.update({
                        "complexity_metrics": {
                            "graph_density": round(density, 4),
                            "average_degree": round(avg_degree, 2)
                        }
                    })
                
                return serialize_for_json({
                    "success": True,
                    "database_size": size_info,
                    "timestamp": serialize_for_json(datetime.datetime.now())
                })
                
            except Exception as e:
                return serialize_for_json({
                    "success": False,
                    "error": f"Failed to get database size: {str(e)}"
                })

    async def cleanup_old_data(self, days_old: int = 30, keep_core_nodes: bool = True, 
                              dry_run: bool = True) -> Dict[str, Any]:
        """Clean up old data based on age or size criteria"""
        async with self.driver.session() as session:
            try:
                # Calculate cutoff date
                cutoff_query = f"RETURN datetime() - duration({{days: {days_old}}}) as cutoff_date"
                result = await session.run(cutoff_query)
                cutoff_record = await result.single()
                cutoff_date = cutoff_record["cutoff_date"]
                
                # Find old nodes
                if keep_core_nodes:
                    old_nodes_query = """
                    MATCH (n)
                    WHERE n.created_at < $cutoff_date 
                    AND NOT n:CoreMemory 
                    AND NOT n:Project 
                    AND NOT n:Snapshot 
                    AND NOT n:Archive
                    RETURN count(n) as old_node_count, collect(n.name)[0..10] as sample_names
                    """
                else:
                    old_nodes_query = """
                    MATCH (n)
                    WHERE n.created_at < $cutoff_date 
                    AND NOT n:Project 
                    AND NOT n:Snapshot 
                    AND NOT n:Archive
                    RETURN count(n) as old_node_count, collect(n.name)[0..10] as sample_names
                    """
                
                result = await session.run(old_nodes_query, cutoff_date=cutoff_date)
                old_data = await result.single()
                
                # Find old relationships
                old_rels_query = """
                MATCH ()-[r]->()
                WHERE r.created_at < $cutoff_date
                RETURN count(r) as old_rel_count, collect(type(r))[0..10] as sample_types
                """
                result = await session.run(old_rels_query, cutoff_date=cutoff_date)
                old_rels = await result.single()
                
                cleanup_plan = {
                    "cutoff_date": serialize_for_json(cutoff_date),
                    "nodes_to_delete": old_data["old_node_count"],
                    "relationships_to_delete": old_rels["old_rel_count"],
                    "sample_node_names": old_data["sample_names"],
                    "sample_rel_types": old_rels["sample_types"],
                    "keep_core_nodes": keep_core_nodes
                }
                
                if dry_run:
                    return serialize_for_json({
                        "success": True,
                        "dry_run": True,
                        "cleanup_plan": cleanup_plan,
                        "message": "Dry run completed - no data deleted"
                    })
                
                # Actually delete old data
                if keep_core_nodes:
                    delete_query = """
                    MATCH (n)
                    WHERE n.created_at < $cutoff_date 
                    AND NOT n:CoreMemory 
                    AND NOT n:Project 
                    AND NOT n:Snapshot 
                    AND NOT n:Archive
                    DETACH DELETE n
                    """
                else:
                    delete_query = """
                    MATCH (n)
                    WHERE n.created_at < $cutoff_date 
                    AND NOT n:Project 
                    AND NOT n:Snapshot 
                    AND NOT n:Archive
                    DETACH DELETE n
                    """
                
                await session.run(delete_query, cutoff_date=cutoff_date)
                
                return serialize_for_json({
                    "success": True,
                    "dry_run": False,
                    "deleted": cleanup_plan,
                    "message": f"Cleaned up data older than {days_old} days"
                })
                
            except Exception as e:
                return serialize_for_json({
                    "success": False,
                    "error": f"Failed to cleanup old data: {str(e)}"
                })

    async def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            await self.driver.close()

async def main():
    """Main server function"""
    try:
        server_instance = Neo4jSemanticHypergraphServer()
        
        # Run the MCP server
        async with stdio_server() as (read_stream, write_stream):
            await server_instance.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="neo4j-hypergraph",
                    server_version="2.0.0",
                    capabilities=ServerCapabilities(
                        tools={}
                    ),
                ),
            )
    except Exception as e:
        with open("neo4j_server_error.log", "w") as f:
            f.write(f"Error during server startup: {e}\n")
            import traceback
            traceback.print_exc(file=f)
    finally:
        if 'server_instance' in locals() and server_instance.driver:
            await server_instance.close()

if __name__ == "__main__":
    asyncio.run(main())

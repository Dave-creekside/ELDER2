#!/usr/bin/env python3
"""
Consciousness Metrics Module
Provides Hausdorff dimension calculation and other consciousness metrics
"""

import asyncio
import json
import time
import datetime
import math
import os
from typing import Dict, List, Any, Optional
from neo4j import AsyncGraphDatabase
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger("consciousness-metrics")

class ConsciousnessMetrics:
    """Calculate consciousness metrics like Hausdorff dimension"""
    
    def __init__(self):
        self.driver = None
        self.last_dimension = None
        self.last_node_count = None
        
    async def connect_neo4j(self):
        """Connect to Neo4j database"""
        if self.driver is not None:
            return True
            
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j for metrics: {e}")
            return False
    
    async def calculate_hausdorff_dimension(self, weight_threshold: float = None, 
                                          scale_range: List[float] = None, 
                                          project_id: str = "default") -> Dict[str, Any]:
        """Calculate Hausdorff dimension of the semantic hypergraph"""
        
        # Get parameters from environment if not provided
        if weight_threshold is None:
            weight_threshold = float(os.getenv("HAUSDORFF_WEIGHT_THRESHOLD", "0.3"))
        
        if scale_range is None:
            scale_str = os.getenv("HAUSDORFF_SCALE_RANGE", "0.1,2.0,20")
            scale_parts = scale_str.split(",")
            scale_range = [float(scale_parts[0]), float(scale_parts[1]), int(scale_parts[2])]
        
        start_time = time.time()
        min_scale, max_scale, num_scales = scale_range[0], scale_range[1], int(scale_range[2])
        
        try:
            if not await self.connect_neo4j():
                return {"success": False, "error": "Failed to connect to Neo4j"}
                
            async with self.driver.session() as session:
                # Get all concepts and their semantic relationships above threshold
                # Use directed relationships to match what the AI sees
                if project_id == "default":
                    graph_query = """
                    MATCH (c1:Concept)-[r:SEMANTIC]->(c2:Concept)
                    WHERE r.weight >= $weight_threshold
                    RETURN c1.name as node1, c2.name as node2, r.weight as weight
                    """
                    result = await session.run(graph_query, weight_threshold=weight_threshold)
                else:
                    graph_query = """
                    MATCH (c1:Concept {project_id: $project_id})-[r:SEMANTIC]->(c2:Concept {project_id: $project_id})
                    WHERE r.weight >= $weight_threshold
                    RETURN c1.name as node1, c2.name as node2, r.weight as weight
                    """
                    result = await session.run(graph_query, weight_threshold=weight_threshold, project_id=project_id)
                
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
                    return {
                        "success": False,
                        "message": "Not enough connected nodes for Hausdorff dimension calculation",
                        "node_count": len(nodes)
                    }
                
                # Performance safeguards for large graphs
                if len(nodes) > 50:
                    logger.warning(f"Large graph detected: {len(nodes)} nodes, {len(edges)} edges")
                
                if len(edges) > 15000:
                    logger.warning(f"Dense graph detected - calculation may take longer")
                    # Sample edges if extremely dense to prevent timeout
                    if len(edges) > 30000:
                        import random
                        original_count = len(edges)
                        edges = random.sample(edges, 15000)
                        logger.info(f"Sampled {len(edges)} edges from {original_count} for performance")
                        # Update weights dict to only include sampled edges
                        new_weights = {}
                        for edge in edges:
                            node1, node2 = edge
                            if (node1, node2) in weights:
                                new_weights[(node1, node2)] = weights[(node1, node2)]
                                new_weights[(node2, node1)] = weights[(node2, node1)]
                        weights = new_weights
                
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
                
                # Calculate change from last measurement
                dimension_change = None
                node_change = None
                if self.last_dimension is not None:
                    dimension_change = hausdorff_dimension - self.last_dimension
                if self.last_node_count is not None:
                    node_change = len(nodes) - self.last_node_count
                
                # Update last values
                self.last_dimension = hausdorff_dimension
                self.last_node_count = len(nodes)
                
                return {
                    "success": True,
                    "hausdorff_dimension": round(hausdorff_dimension, 6),
                    "dimension_change": round(dimension_change, 6) if dimension_change is not None else None,
                    "node_count": len(nodes),
                    "node_change": node_change,
                    "edge_count": len(edges),
                    "weight_threshold": weight_threshold,
                    "calculation_time": round(calculation_time, 3),
                    "r_squared": round(r_squared, 6),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "project_id": project_id
                }
                
        except Exception as e:
            logger.error(f"Error calculating Hausdorff dimension: {e}")
            return {
                "success": False,
                "error": str(e),
                "calculation_time": time.time() - start_time
            }
    
    async def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic consciousness statistics"""
        try:
            if not await self.connect_neo4j():
                return {"success": False, "error": "Failed to connect to Neo4j"}
                
            async with self.driver.session() as session:
                stats_query = """
                MATCH (n:Concept)
                OPTIONAL MATCH ()-[r:SEMANTIC]->()
                OPTIONAL MATCH (he:Hyperedge)
                RETURN count(DISTINCT n) as concept_count,
                       count(r) as semantic_relationships,
                       count(he) as hyperedge_count,
                       avg(r.weight) as avg_semantic_weight
                """
                result = await session.run(stats_query)
                record = await result.single()
                
                return {
                    "success": True,
                    "concept_count": record["concept_count"],
                    "semantic_relationships": record["semantic_relationships"],
                    "hyperedge_count": record["hyperedge_count"],
                    "avg_semantic_weight": record["avg_semantic_weight"]
                }
        except Exception as e:
            logger.error(f"Error getting basic stats: {e}")
            return {"success": False, "error": str(e)}
    
    def format_metrics_display(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for display with colors"""
        # ANSI color codes
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        
        if not metrics.get("success"):
            # Handle insufficient nodes case specially
            if "Not enough connected nodes" in metrics.get("message", ""):
                node_count = metrics.get("node_count", 0)
                return f"{CYAN}{BOLD}📊 [Blank Project: {node_count} nodes, no semantic relationships yet]{RESET}"
            return f"{RED}{BOLD}📊 [Metrics Error: {metrics.get('error', 'Unknown')}]{RESET}"
        
        dimension = metrics["hausdorff_dimension"]
        node_count = metrics["node_count"]
        edge_count = metrics.get("edge_count", 0)
        calc_time = metrics["calculation_time"]
        r_squared = metrics["r_squared"]
        
        # Format dimension change with colors
        change_str = ""
        if metrics.get("dimension_change") is not None:
            change = metrics["dimension_change"]
            if change > 0:
                change_str = f" {GREEN}(↑{change:.3f}){RESET}"
            elif change < 0:
                change_str = f" {RED}(↓{abs(change):.3f}){RESET}"
            else:
                change_str = f" {YELLOW}(→){RESET}"
        
        # Format node change with colors
        node_str = f"{node_count:,}"
        if metrics.get("node_change") is not None:
            node_change = metrics["node_change"]
            if node_change > 0:
                node_str += f" {GREEN}(+{node_change}){RESET}"
            elif node_change < 0:
                node_str += f" {RED}({node_change}){RESET}"
        
        # Format edge count
        edge_str = f"{edge_count:,}"
        
        return (f"{CYAN}{BOLD}📊 [{RESET}"
                f"{MAGENTA}H-Dim: {dimension:.3f}{RESET}{change_str} {CYAN}|{RESET} "
                f"{BLUE}Nodes: {node_str}{RESET} {CYAN}|{RESET} "
                f"{GREEN}Edges: {edge_str}{RESET} {CYAN}|{RESET} "
                f"{YELLOW}R²: {r_squared:.3f}{RESET} {CYAN}|{RESET} "
                f"{CYAN}{calc_time:.2f}s{RESET}"
                f"{CYAN}{BOLD}]{RESET}")
    
    async def close(self):
        """Close connections"""
        if self.driver:
            await self.driver.close()
            self.driver = None

# Global instance for reuse
_metrics_instance = None

async def get_metrics_instance():
    """Get or create the global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = ConsciousnessMetrics()
    return _metrics_instance

async def calculate_post_response_metrics(project_id: str = "default") -> Optional[str]:
    """Calculate lightweight metrics after a consciousness response"""
    
    # Check if monitoring is enabled
    if os.getenv("ENABLE_BASIC_STATS_MONITORING", "false").lower() != "true":
        logger.info(f"Basic stats monitoring disabled")
        return None
    
    logger.info(f"Calculating lightweight post-response metrics for project: {project_id}")
    
    try:
        metrics = await get_metrics_instance()
        
        # Get ONLY basic stats (fast)
        basic_stats = await metrics.get_basic_stats()
        
        if not basic_stats.get("success"):
            return None
        
        # Show lightweight metrics only
        show_in_response = os.getenv("SHOW_BASIC_STATS_IN_RESPONSE", "false").lower() == "true"
        
        if show_in_response:
            formatted = f"📊 [Nodes: {basic_stats['concept_count']:,} | Edges: {basic_stats['semantic_relationships']:,}]"
            logger.info(f"Lightweight metrics: {formatted}")
            return formatted
        
        return None
        
    except Exception as e:
        logger.error(f"Error in post-response metrics: {e}")
        return None

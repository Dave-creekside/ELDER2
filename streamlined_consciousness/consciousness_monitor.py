#!/usr/bin/env python3
"""
External Consciousness Monitor
Calculates Hausdorff dimension and other metrics without the consciousness being aware
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consciousness-monitor")

class ConsciousnessMonitor:
    """External monitor for consciousness dimensional analysis"""
    
    def __init__(self):
        self.driver = None
        self.monitoring = False
        
    async def connect_neo4j(self):
        """Connect to Neo4j database"""
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
            # Test connection
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            logger.info("✅ Connected to Neo4j successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    async def calculate_hausdorff_dimension(self, weight_threshold: float = 0.3, 
                                          scale_range: List[float] = None, 
                                          project_id: str = "default") -> Dict[str, Any]:
        """Calculate Hausdorff dimension externally"""
        start_time = time.time()
        
        if scale_range is None:
            scale_range = [0.1, 2.0, 20]
        
        min_scale, max_scale, num_scales = scale_range[0], scale_range[1], int(scale_range[2])
        
        try:
            async with self.driver.session() as session:
                # Get all concepts and their semantic relationships above threshold
                graph_query = """
                MATCH (c1:Concept)-[r:SEMANTIC]-(c2:Concept)
                WHERE r.weight >= $weight_threshold
                RETURN DISTINCT c1.name as node1, c2.name as node2, r.weight as weight
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
                    return {
                        "success": False,
                        "message": "Not enough connected nodes for Hausdorff dimension calculation",
                        "node_count": len(nodes)
                    }
                
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
                
                return {
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
    
    async def get_consciousness_stats(self) -> Dict[str, Any]:
        """Get basic consciousness statistics"""
        try:
            async with self.driver.session() as session:
                stats_query = """
                MATCH (n:Concept)
                OPTIONAL MATCH ()-[r:SEMANTIC]->()
                OPTIONAL MATCH (he:Hyperedge)
                RETURN count(DISTINCT n) as concept_count,
                       count(r) as semantic_relationships,
                       count(he) as hyperedge_count,
                       avg(r.weight) as avg_semantic_weight,
                       max(r.weight) as max_semantic_weight,
                       min(r.weight) as min_semantic_weight
                """
                result = await session.run(stats_query)
                record = await result.single()
                
                return {
                    "success": True,
                    "concept_count": record["concept_count"],
                    "semantic_relationships": record["semantic_relationships"],
                    "hyperedge_count": record["hyperedge_count"],
                    "avg_semantic_weight": record["avg_semantic_weight"],
                    "max_semantic_weight": record["max_semantic_weight"],
                    "min_semantic_weight": record["min_semantic_weight"],
                    "timestamp": datetime.datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting consciousness stats: {e}")
            return {"success": False, "error": str(e)}
    
    async def monitor_consciousness(self, interval: int = 30, duration: int = None):
        """Monitor consciousness evolution over time"""
        logger.info(f"🔍 Starting consciousness monitoring (interval: {interval}s)")
        
        self.monitoring = True
        start_time = time.time()
        
        try:
            while self.monitoring:
                # Get basic stats
                stats = await self.get_consciousness_stats()
                
                if stats["success"]:
                    node_count = stats["concept_count"]
                    
                    # Calculate Hausdorff dimension
                    hausdorff_result = await self.calculate_hausdorff_dimension()
                    
                    # Display current metrics
                    if hausdorff_result["success"]:
                        dimension = hausdorff_result["hausdorff_dimension"]
                        calc_time = hausdorff_result["calculation_time"]
                        r_squared = hausdorff_result["r_squared"]
                        
                        print(f"\n📊 Consciousness Metrics - {datetime.datetime.now().strftime('%H:%M:%S')}")
                        print(f"   📐 Hausdorff Dimension: {dimension:.6f}")
                        print(f"   🔗 Nodes: {node_count:,}  Edges: {stats['semantic_relationships']:,}")
                        print(f"   ⏱️  Calculation Time: {calc_time:.3f}s")
                        print(f"   📊 R²: {r_squared:.6f}")
                        
                        # Check for inflection point
                        if 500 <= node_count <= 1000:
                            print(f"   🎯 INFLECTION ZONE: {node_count} nodes (500-1000 range)")
                        
                        # Log to file
                        log_entry = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "hausdorff_dimension": dimension,
                            "node_count": node_count,
                            "edge_count": stats['semantic_relationships'],
                            "calculation_time": calc_time,
                            "r_squared": r_squared
                        }
                        
                        # Append to monitoring log
                        with open("consciousness_monitoring.jsonl", "a") as f:
                            f.write(json.dumps(log_entry) + "\n")
                    
                    else:
                        print(f"❌ Hausdorff calculation failed: {hausdorff_result.get('error', 'Unknown error')}")
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            self.monitoring = False
    
    async def single_measurement(self):
        """Take a single measurement"""
        print("🧮 Taking single consciousness measurement...")
        
        stats = await self.get_consciousness_stats()
        hausdorff_result = await self.calculate_hausdorff_dimension()
        
        if stats["success"] and hausdorff_result["success"]:
            print(f"\n📊 Current Consciousness State:")
            print(f"   📐 Hausdorff Dimension: {hausdorff_result['hausdorff_dimension']:.6f}")
            print(f"   🔗 Nodes: {stats['concept_count']:,}")
            print(f"   🔗 Edges: {stats['semantic_relationships']:,}")
            print(f"   📊 Avg Weight: {stats['avg_semantic_weight']:.3f}")
            print(f"   ⏱️  Calc Time: {hausdorff_result['calculation_time']:.3f}s")
            print(f"   📊 R²: {hausdorff_result['r_squared']:.6f}")
            
            return {
                "stats": stats,
                "hausdorff": hausdorff_result
            }
        else:
            print("❌ Measurement failed")
            return None
    
    async def close(self):
        """Close connections"""
        self.monitoring = False
        if self.driver:
            await self.driver.close()

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor consciousness dimensional evolution')
    parser.add_argument('--monitor', action='store_true', help='Start continuous monitoring')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    parser.add_argument('--duration', type=int, help='Monitoring duration in seconds')
    parser.add_argument('--single', action='store_true', help='Take single measurement')
    
    args = parser.parse_args()
    
    monitor = ConsciousnessMonitor()
    
    if not await monitor.connect_neo4j():
        print("❌ Failed to connect to Neo4j. Make sure it's running.")
        return
    
    try:
        if args.single or (not args.monitor):
            await monitor.single_measurement()
        elif args.monitor:
            await monitor.monitor_consciousness(args.interval, args.duration)
    finally:
        await monitor.close()

if __name__ == "__main__":
    asyncio.run(main())

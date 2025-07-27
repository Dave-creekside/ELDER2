"""
Semantic Cellular Automata System
Intelligent hypergraph evolution with quality controls and safety monitoring
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from .ca_quality import QualityAnalyzer, QualityMetrics, ConnectionCandidate
from .ca_monitor import CAMonitor, CAOperationStatus
from .ca_parameters import AdaptiveParameters, CAParameters, CAPhase

logger = logging.getLogger("semantic-ca")

@dataclass
class CAResult:
    """Result of a CA operation"""
    success: bool
    connections_created: int
    connections_pruned: int
    total_operations: int
    phase: CAPhase
    session_id: str
    execution_time: float
    quality_improvement: float
    error_message: Optional[str] = None
    emergency_stop: bool = False

class SemanticCellularAutomata:
    """
    Intelligent Semantic Cellular Automata with quality controls and monitoring
    """
    
    def __init__(self, neo4j_tool_executor):
        """
        Initialize with a reference to neo4j tool executor for graph operations
        """
        self.neo4j_executor = neo4j_tool_executor
        self.quality_analyzer = QualityAnalyzer()
        self.monitor = CAMonitor()
        self.adaptive_params = AdaptiveParameters()
        
        # State tracking
        self.current_phase = CAPhase.MAINTENANCE
        self.last_graph_stats = {}
        
        # Setup emergency stop callback
        self.monitor.add_emergency_stop_callback(self._handle_emergency_stop)
        
    async def _get_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics"""
        try:
            result = await self.neo4j_executor("get_graph_stats", {})
            if isinstance(result, str):
                stats = json.loads(result)
            else:
                stats = result
            
            if stats.get("success"):
                return stats
            else:
                logger.warning(f"Failed to get graph stats: {stats}")
                return {}
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {}
    
    async def _get_batch_similarities(self, concept_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        """
        Get semantic similarities for multiple concept pairs in batch
        """
        try:
            # Convert to list format for the tool call
            pairs_list = [[c1, c2] for c1, c2 in concept_pairs]
            
            logger.debug(f"Calling batch similarity calculation for {len(pairs_list)} pairs")
            
            # Call the batch similarity calculation
            result = await self.neo4j_executor("calculate_batch_similarities", {
                "concept_pairs": pairs_list,
                "boost_hyperedge_members": True
            })
            
            # Debug logging
            logger.debug(f"Batch similarity raw result type: {type(result)}")
            logger.debug(f"Batch similarity raw result length: {len(result) if result else 'None'}")
            
            if not result:
                logger.error("Batch similarity returned empty result")
                raise ValueError("Empty result from batch similarity calculation")
            
            # Handle the result based on type
            if isinstance(result, str):
                logger.debug(f"Result is string, first 200 chars: {result[:200]}...")
                try:
                    result_data = json.loads(result)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON result: {e}")
                    logger.error(f"Raw result: {result[:500]}...")
                    raise
            else:
                result_data = result
            
            logger.debug(f"Parsed result data type: {type(result_data)}")
            
            # Convert back to tuple keys
            similarities = {}
            
            # Handle different response formats
            if isinstance(result_data, dict):
                for key, score in result_data.items():
                    # Handle new pipe-separated format "concept1||concept2"
                    if isinstance(key, str) and "||" in key:
                        parts = key.split("||")
                        if len(parts) == 2:
                            similarities[(parts[0], parts[1])] = score
                        else:
                            logger.warning(f"Unexpected key format: {key}")
                    # Handle old string tuple format "(concept1, concept2)"
                    elif isinstance(key, str) and key.startswith("("):
                        # Parse the string tuple
                        key_parts = key.strip("()").split(", ")
                        if len(key_parts) == 2:
                            c1 = key_parts[0].strip("'\"")
                            c2 = key_parts[1].strip("'\"")
                            similarities[(c1, c2)] = score
                    elif isinstance(key, (list, tuple)) and len(key) == 2:
                        # Handle list/tuple keys
                        similarities[(key[0], key[1])] = score
                    else:
                        logger.warning(f"Unexpected key format: {key}")
            else:
                logger.error(f"Unexpected result format: {type(result_data)}")
                raise ValueError(f"Expected dict result, got {type(result_data)}")
            
            logger.info(f"Batch similarity calculation successful: {len(similarities)} pairs processed")
            
            # Log first few similarities for verification
            for i, ((c1, c2), score) in enumerate(similarities.items()):
                if i < 3:
                    logger.debug(f"  Similarity({c1}, {c2}) = {score:.3f}")
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Batch similarity calculation failed, falling back to individual calculations: {e}")
            # Fallback to individual calculations
            similarities = {}
            for c1, c2 in concept_pairs:
                try:
                    # Try to use the semantic similarity tool first
                    similarity = await self._get_semantic_similarity(c1, c2)
                except Exception as e2:
                    logger.warning(f"Individual similarity calculation also failed for {c1}-{c2}: {e2}")
                    # Last resort: use heuristic
                    similarity = self._heuristic_similarity(c1, c2)
                similarities[(c1, c2)] = similarity
            
            logger.info(f"Fallback completed: {len(similarities)} pairs processed individually")
            return similarities
    
    async def _get_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """
        Get semantic similarity between two concepts (kept for compatibility)
        """
        # This is now just a wrapper that calls batch with a single pair
        batch_result = await self._get_batch_similarities([(concept1, concept2)])
        return batch_result.get((concept1, concept2), self._heuristic_similarity(concept1, concept2))
    
    def _heuristic_similarity(self, concept1: str, concept2: str) -> float:
        """
        Improved heuristic similarity as fallback
        """
        # Normalize concepts
        words1 = set(concept1.lower().replace('_', ' ').split())
        words2 = set(concept2.lower().replace('_', ' ').split())
        
        # Expanded semantic relationships
        semantic_groups = [
            # Consciousness-related
            {'consciousness', 'awareness', 'mind', 'cognition', 'sentience', 'perception'},
            # Memory-related
            {'memory', 'remember', 'recall', 'retention', 'storage', 'long', 'short', 'working'},
            # Knowledge-related
            {'knowledge', 'understanding', 'learning', 'wisdom', 'insight', 'comprehension'},
            # Thought-related
            {'thought', 'thinking', 'reasoning', 'logic', 'analysis', 'reflection'},
            # Dream-related
            {'dream', 'sleep', 'unconscious', 'subconscious', 'imagination'},
            # Tool-related
            {'tools', 'instrument', 'method', 'technique', 'utility', 'function'},
            # Temporal
            {'temporal', 'time', 'chronological', 'sequence', 'duration'},
            # Spatial
            {'spatial', 'space', 'dimension', 'topology', 'geometry'},
        ]
        
        # Check if words belong to same semantic group
        for group in semantic_groups:
            group_words1 = words1.intersection(group)
            group_words2 = words2.intersection(group)
            if group_words1 and group_words2:
                # Strong similarity if in same semantic group
                return 0.7 + (0.2 if group_words1 == group_words2 else 0.1)
        
        # Check for direct word overlap
        if words1.intersection(words2):
            overlap_ratio = len(words1.intersection(words2)) / min(len(words1), len(words2))
            return min(0.5 + overlap_ratio * 0.4, 0.9)
        
        # Check for substring matches
        for w1 in words1:
            for w2 in words2:
                if len(w1) > 3 and len(w2) > 3:
                    if w1 in w2 or w2 in w1:
                        return 0.6
                    # Check for common roots (simple stemming)
                    if w1[:4] == w2[:4]:
                        return 0.5
        
        # Default low similarity
        return 0.2
    
    async def _find_connection_candidates(self, min_common_neighbors: int = 2,
                                        max_candidates: int = 100) -> List[Dict[str, Any]]:
        """
        Find potential connection candidates based on shared hyperedges or common neighbors
        """
        try:
            # Try the dedicated CA tool first (avoids truncation)
            result = await self.neo4j_executor("get_ca_connection_candidates", {
                "min_common_neighbors": min_common_neighbors,
                "max_candidates": max_candidates,
                "format": "compact"
            })
            
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result
            
            if result_data.get("success") and result_data.get("candidates"):
                # Convert compact format to expected format
                candidates = []
                for candidate in result_data["candidates"]:
                    if isinstance(candidate, list) and len(candidate) >= 3:
                        candidates.append({
                            "concept1": candidate[0],
                            "concept2": candidate[1],
                            "common_neighbors": candidate[2]
                        })
                
                logger.info(f"   Found {len(candidates)} connection candidates using dedicated CA tool")
                return candidates
            
        except Exception as e:
            logger.warning(f"Dedicated CA tool failed, falling back to query_cypher: {e}")
            
        # Fallback to old method if dedicated tool fails
        try:
            # Query to find concept pairs that either:
            # 1. Share membership in hyperedges (co-members)
            # 2. Have common neighbors through any relationship type
            # But don't have a direct SEMANTIC connection
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
            
            result = await self.neo4j_executor("query_cypher", {
                "query": query,
                "parameters": {
                    "min_common_neighbors": min_common_neighbors,
                    "max_candidates": max_candidates
                }
            })
            
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result
            
            if result_data.get("success"):
                records = result_data.get("records", [])
                
                # Handle MCP truncation issue
                if isinstance(records, str) and "[list - truncated for LLM]" in records:
                    logger.warning("Neo4j results were truncated by MCP, retrying with direct query...")
                    # The query found results but they were truncated
                    # Try a more limited query to avoid truncation
                    limited_query = """
                    MATCH (a:Concept)-[:MEMBER_OF]->(he:Hyperedge)<-[:MEMBER_OF]-(b:Concept)
                    WHERE a <> b AND NOT EXISTS((a)-[:SEMANTIC]-(b))
                    RETURN a.name as concept1, b.name as concept2, 1 as common_neighbors
                    LIMIT 20
                    """
                    
                    retry_result = await self.neo4j_executor("query_cypher", {
                        "query": limited_query,
                        "parameters": {}
                    })
                    
                    if isinstance(retry_result, str):
                        retry_data = json.loads(retry_result)
                    else:
                        retry_data = retry_result
                    
                    if retry_data.get("success") and retry_data.get("records"):
                        # Use the retry records directly
                        records = retry_data.get("records", [])
                        
                        # Check if records is still a string after retry
                        if isinstance(records, str):
                            logger.warning("Retry records are still a string, attempting to parse...")
                            try:
                                # MCP might be returning the count as a string
                                if records.isdigit():
                                    logger.error(f"Records is just a number: {records}")
                                    return []
                                # Try parsing as JSON
                                records = json.loads(records)
                                logger.info(f"Successfully parsed retry records")
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse retry records: {e}")
                                logger.error(f"Records string: {records[:200]}...")
                                return []
                        
                        logger.info(f"   Retry successful, got {len(records)} records")
                        # Fall through to validation with the new records
                    else:
                        logger.warning("Retry also failed, returning empty")
                        return []
                elif isinstance(records, str):
                    # Some other string format, not truncation
                    logger.warning(f"Neo4j returned unexpected string: {records[:200]}...")
                    return []
                
                if not isinstance(records, list):
                    logger.warning(f"Invalid records format - expected list, got {type(records)}")
                    return []
                
                # Validate each record is a dictionary
                valid_records = []
                for i, record in enumerate(records):
                    if isinstance(record, dict):
                        # Check for required fields
                        if "concept1" in record and "concept2" in record:
                            valid_records.append(record)
                        else:
                            logger.debug(f"Record {i} missing required fields: {record}")
                    else:
                        logger.debug(f"Record {i} is not a dictionary: {type(record)}")
                
                logger.info(f"   Validated {len(valid_records)} of {len(records)} candidate records")
                return valid_records
            else:
                logger.warning(f"Failed to find connection candidates: {result_data}")
                return []
                
        except Exception as e:
            logger.error(f"Error finding connection candidates: {e}")
            return []
    
    async def _get_node_connection_count(self, concept_name: str) -> int:
        """Get the number of connections for a specific node"""
        try:
            query = """
            MATCH (c:Concept {name: $concept_name})-[]-(connected:Concept)
            RETURN count(DISTINCT connected) as connection_count
            """
            
            result = await self.neo4j_executor("query_cypher", {
                "query": query,
                "parameters": {"concept_name": concept_name}
            })
            
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result
            
            if result_data.get("success") and result_data.get("records"):
                return result_data["records"][0].get("connection_count", 0)
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error getting connection count for {concept_name}: {e}")
            return 0
    
    async def _create_semantic_connection(self, concept1: str, concept2: str, 
                                        semantic_weight: float, 
                                        common_neighbors: int) -> bool:
        """Create a new semantic connection between concepts"""
        try:
            result = await self.neo4j_executor("create_relationship", {
                "from_concept": concept1,
                "to_concept": concept2,
                "relationship_type": "SEMANTIC",
                "properties": {
                    "weight": semantic_weight,
                    "semantic_weight": semantic_weight,
                    "created_by": "semantic_ca",
                    "common_neighbors": common_neighbors,
                    "created_at": time.time()
                },
                "auto_calculate_weight": False  # We're providing our own weight
            })
            
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result
            
            success = result_data.get("success", False)
            
            # Record the operation in monitor
            self.monitor.record_operation(
                operation_type="creation",
                concept1=concept1,
                concept2=concept2,
                semantic_weight=semantic_weight,
                success=success
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error creating semantic connection {concept1}-{concept2}: {e}")
            self.monitor.record_operation(
                operation_type="creation",
                concept1=concept1,
                concept2=concept2,
                success=False,
                error_message=str(e)
            )
            return False
    
    async def _prune_weak_connections(self, prune_threshold: float, 
                                    max_prune_operations: int) -> int:
        """Prune connections below the threshold"""
        try:
            # Query to find weak connections
            query = """
            MATCH ()-[r:SEMANTIC]->()
            WHERE coalesce(r.semantic_weight, r.weight) < $prune_threshold
            WITH r LIMIT $max_prune
            DELETE r
            RETURN count(*) as pruned_count
            """
            
            result = await self.neo4j_executor("query_cypher", {
                "query": query,
                "parameters": {
                    "prune_threshold": prune_threshold,
                    "max_prune": max_prune_operations
                }
            })
            
            if isinstance(result, str):
                result_data = json.loads(result)
            else:
                result_data = result
            
            if result_data.get("success") and result_data.get("records"):
                pruned_count = result_data["records"][0].get("pruned_count", 0)
                
                # Record pruning operations
                for _ in range(pruned_count):
                    self.monitor.record_operation(
                        operation_type="pruning",
                        success=True
                    )
                
                return pruned_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Error pruning weak connections: {e}")
            self.monitor.record_operation(
                operation_type="pruning",
                success=False,
                error_message=str(e)
            )
            return 0
    
    async def _analyze_connection_candidates(self, raw_candidates: List[Dict[str, Any]],
                                           ca_params: CAParameters) -> List[ConnectionCandidate]:
        """
        Analyze and score connection candidates
        """
        analyzed_candidates = []
        
        for candidate_data in raw_candidates:
            concept1 = candidate_data["concept1"]
            concept2 = candidate_data["concept2"]
            common_neighbors = candidate_data["common_neighbors"]
            
            # Get semantic similarity
            semantic_similarity = await self._get_semantic_similarity(concept1, concept2)
            
            # Get existing connection counts
            connections1 = await self._get_node_connection_count(concept1)
            connections2 = await self._get_node_connection_count(concept2)
            
            # Analyze the candidate
            candidate = self.quality_analyzer.analyze_connection_candidate(
                concept1, concept2, semantic_similarity, common_neighbors,
                connections1, connections2
            )
            
            # Check if we should create this connection
            if self.quality_analyzer.should_create_connection(
                candidate, ca_params.min_quality_score, ca_params.max_connections_per_node
            ):
                analyzed_candidates.append(candidate)
        
        # Sort by priority (highest first)
        analyzed_candidates.sort(key=lambda c: c.creation_priority, reverse=True)
        
        return analyzed_candidates
    
    async def apply_semantic_ca_rules(self, phase: CAPhase = CAPhase.MAINTENANCE,
                                    custom_params: Optional[CAParameters] = None) -> CAResult:
        """
        Apply semantic cellular automata rules with full intelligence and safety
        """
        session_id = self.monitor.start_session(f"semantic_ca_{phase.value}")
        start_time = time.time()
        
        # Get current graph state
        initial_stats = await self._get_graph_stats()
        if not initial_stats:
            return CAResult(
                success=False,
                connections_created=0,
                connections_pruned=0,
                total_operations=0,
                phase=phase,
                session_id=session_id,
                execution_time=time.time() - start_time,
                quality_improvement=0.0,
                error_message="Failed to get initial graph stats"
            )
        
        # Calculate quality metrics
        initial_quality = self.quality_analyzer.calculate_quality_metrics(initial_stats)
        
        # Use fixed parameters for each phase (no adaptive adjustments)
        if custom_params:
            ca_params = custom_params
        else:
            # Get base parameters directly without adaptation
            ca_params = self.adaptive_params.base_parameters[phase]
        
        logger.info(f"🔬 Starting Semantic CA: {phase.value}")
        
        # Different parameter display for different phases
        if phase == CAPhase.PRE_DREAM:
            logger.info(f"   Parameters: similarity≥{ca_params.min_similarity}, "
                       f"neighbors≥{ca_params.common_neighbors_threshold}, "
                       f"max_ops={ca_params.max_operations} (EXPANSION - no pruning)")
        else:
            logger.info(f"   Parameters: similarity≥{ca_params.min_similarity}, "
                       f"neighbors≥{ca_params.common_neighbors_threshold}, "
                       f"prune<{ca_params.prune_threshold}, max_ops={ca_params.max_operations}")
        
        connections_created = 0
        connections_pruned = 0
        total_operations = 0
        
        try:
            # PRE-DREAM: Skip pruning entirely, focus on expansion
            if phase == CAPhase.PRE_DREAM:
                logger.info("   🌱 Pre-dream expansion mode: skipping pruning, connecting all candidates")
                # Skip pruning phase entirely for pre-dream
                self.monitor.current_status = CAOperationStatus.CREATING_CONNECTIONS
            else:
                # PHASE 1: Pruning (for post-dream and maintenance)
                self.monitor.current_status = CAOperationStatus.PRUNING
                
                max_prune_ops = min(ca_params.max_operations // 2, 200)
                if max_prune_ops > 0:
                    pruned = await self._prune_weak_connections(
                        ca_params.prune_threshold, max_prune_ops
                    )
                    connections_pruned += pruned
                    total_operations += pruned
                    
                    logger.info(f"   Pruned {pruned} weak connections (threshold: {ca_params.prune_threshold})")
                
                # Check if we should continue
                should_continue, reason = self.monitor.should_continue_operations()
                if not should_continue:
                    logger.warning(f"⚠️ CA stopped after pruning: {reason}")
                    return await self._finalize_ca_result(
                        session_id, start_time, phase, connections_created, connections_pruned,
                        total_operations, initial_quality, reason
                    )
                
                # PHASE 2: Connection Creation (semantic analysis)
                self.monitor.current_status = CAOperationStatus.CREATING_CONNECTIONS
            
            max_creation_ops = ca_params.max_operations - total_operations
            if max_creation_ops > 0:
                # Find connection candidates
                raw_candidates = await self._find_connection_candidates(
                    ca_params.common_neighbors_threshold,
                    min(max_creation_ops * 3, 200)  # Search more candidates than we'll create
                )
                
                if raw_candidates:
                    logger.info(f"   Found {len(raw_candidates)} connection candidates")
                    
                    # PRE-DREAM: Simple expansion mode - connect ALL candidates
                    if phase == CAPhase.PRE_DREAM:
                        logger.info("   🌱 Expansion mode: connecting ALL candidates without complex analysis")
                        
                        # Simple limit check
                        creation_limit = min(
                            len(raw_candidates),
                            ca_params.max_new_connections,
                            max_creation_ops
                        )
                        
                        # Prepare candidates for batch processing
                        valid_candidates = []
                        concept_pairs = []
                        
                        for i, candidate_data in enumerate(raw_candidates[:creation_limit]):
                            try:
                                # Handle potential string parsing issues
                                if isinstance(candidate_data, str):
                                    try:
                                        candidate_data = json.loads(candidate_data)
                                    except json.JSONDecodeError:
                                        logger.error(f"Failed to parse candidate data: {candidate_data}")
                                        continue
                                
                                # Extract data safely
                                concept1 = candidate_data.get("concept1")
                                concept2 = candidate_data.get("concept2") 
                                common_neighbors = candidate_data.get("common_neighbors", 1)
                                
                                if not concept1 or not concept2:
                                    logger.warning(f"Invalid candidate data: {candidate_data}")
                                    continue
                                
                                valid_candidates.append({
                                    "concept1": concept1,
                                    "concept2": concept2,
                                    "common_neighbors": common_neighbors
                                })
                                concept_pairs.append((concept1, concept2))
                                
                            except Exception as e:
                                logger.error(f"Error processing candidate {i}: {e}, data: {candidate_data}")
                                continue
                        
                        if concept_pairs:
                            logger.info(f"   Calculating batch similarities for {len(concept_pairs)} candidate pairs...")
                            
                            # Batch calculate all similarities at once
                            similarities = await self._get_batch_similarities(concept_pairs)
                            
                            logger.info(f"   Batch calculation complete - received {len(similarities)} results")
                            
                            # Log first few similarities for debugging
                            for i, (pair, similarity) in enumerate(similarities.items()):
                                if i < 5:
                                    logger.debug(f"   Similarity({pair[0]}, {pair[1]}) = {similarity:.3f}")
                            
                            # Create connections for pairs above threshold
                            connections_to_create = []
                            for candidate in valid_candidates:
                                concept1 = candidate["concept1"]
                                concept2 = candidate["concept2"]
                                common_neighbors = candidate["common_neighbors"]
                                
                                similarity = similarities.get((concept1, concept2), 0.0)
                                
                                if similarity >= ca_params.min_similarity:
                                    connections_to_create.append({
                                        "concept1": concept1,
                                        "concept2": concept2,
                                        "similarity": similarity,
                                        "common_neighbors": common_neighbors
                                    })
                                else:
                                    logger.debug(f"   Skipping {concept1}-{concept2}: similarity {similarity:.3f} < {ca_params.min_similarity}")
                            
                            logger.info(f"   {len(connections_to_create)} pairs passed similarity threshold")
                            
                            # Create connections with rate limiting
                            for i, conn in enumerate(connections_to_create):
                                # Check safety limits before each creation
                                should_continue, reason = self.monitor.should_continue_operations()
                                if not should_continue:
                                    logger.warning(f"⚠️ CA stopped during creation: {reason}")
                                    break
                                
                                success = await self._create_semantic_connection(
                                    conn["concept1"], conn["concept2"], 
                                    conn["similarity"], conn["common_neighbors"]
                                )
                                
                                if success:
                                    connections_created += 1
                                    total_operations += 1
                                    
                                    if connections_created % 10 == 0:  # Log progress every 10 connections
                                        logger.info(f"   Created {connections_created} connections...")
                                
                                # Rate limiting
                                await asyncio.sleep(1.0 / ca_params.max_operations_per_second)
                    
                    else:
                        # POST-DREAM and MAINTENANCE: Use complex quality analysis
                        self.monitor.current_status = CAOperationStatus.ANALYZING
                        logger.info(f"   Analyzing {len(raw_candidates)} candidates for quality...")
                        
                        # BATCH PROCESSING: Collect all pairs first
                        concept_pairs = []
                        candidate_info = []
                        
                        # Limit candidates for safety
                        analysis_limit = min(len(raw_candidates), 100)
                        if len(raw_candidates) > analysis_limit:
                            logger.warning(f"   Limiting analysis to first {analysis_limit} candidates")
                        
                        for idx, candidate_data in enumerate(raw_candidates[:analysis_limit]):
                            concept1 = candidate_data["concept1"]
                            concept2 = candidate_data["concept2"]
                            common_neighbors = candidate_data["common_neighbors"]
                            
                            concept_pairs.append((concept1, concept2))
                            candidate_info.append({
                                "concept1": concept1,
                                "concept2": concept2,
                                "common_neighbors": common_neighbors,
                                "index": idx
                            })
                        
                        # Get ALL similarities in ONE batch call
                        logger.info(f"   Calculating batch similarities for {len(concept_pairs)} pairs...")
                        similarities = await self._get_batch_similarities(concept_pairs)
                        logger.info(f"   Batch calculation complete")
                        
                        # Now analyze candidates with the batch results
                        analyzed_candidates = []
                        for info in candidate_info:
                            concept1 = info["concept1"]
                            concept2 = info["concept2"]
                            common_neighbors = info["common_neighbors"]
                            
                            # Get similarity from batch results
                            semantic_similarity = similarities.get((concept1, concept2), 0.0)
                            
                            # Skip if too low
                            if semantic_similarity < ca_params.min_similarity:
                                continue
                            
                            # Get existing connection counts
                            connections1 = await self._get_node_connection_count(concept1)
                            connections2 = await self._get_node_connection_count(concept2)
                            
                            # Analyze the candidate
                            candidate = self.quality_analyzer.analyze_connection_candidate(
                                concept1, concept2, semantic_similarity, common_neighbors,
                                connections1, connections2
                            )
                            
                            # Check if we should create this connection
                            if self.quality_analyzer.should_create_connection(
                                candidate, ca_params.min_quality_score, ca_params.max_connections_per_node
                            ):
                                analyzed_candidates.append(candidate)
                        
                        # Sort by priority (highest first)
                        analyzed_candidates.sort(key=lambda c: c.creation_priority, reverse=True)
                        
                        logger.info(f"   {len(analyzed_candidates)} candidates passed quality analysis")
                        
                        # Create connections with limits and monitoring
                        creation_limit = min(
                            len(analyzed_candidates),
                            ca_params.max_new_connections,
                            max_creation_ops
                        )
                        
                        self.monitor.current_status = CAOperationStatus.CREATING_CONNECTIONS
                        
                        for i, candidate in enumerate(analyzed_candidates[:creation_limit]):
                            # Check safety limits before each creation
                            should_continue, reason = self.monitor.should_continue_operations()
                            if not should_continue:
                                logger.warning(f"⚠️ CA stopped during creation: {reason}")
                                break
                            
                            # Create the connection
                            success = await self._create_semantic_connection(
                                candidate.concept1, candidate.concept2,
                                candidate.semantic_similarity, candidate.common_neighbors
                            )
                            
                            if success:
                                connections_created += 1
                                total_operations += 1
                                
                                if i % 10 == 0:  # Log progress every 10 connections
                                    logger.info(f"   Created {connections_created} connections...")
                            
                            # Rate limiting
                            await asyncio.sleep(1.0 / ca_params.max_operations_per_second)
                    
                    logger.info(f"   Created {connections_created} new semantic connections")
                else:
                    logger.info("   No connection candidates found")
            
        except Exception as e:
            logger.error(f"Error during semantic CA execution: {e}")
            return CAResult(
                success=False,
                connections_created=connections_created,
                connections_pruned=connections_pruned,
                total_operations=total_operations,
                phase=phase,
                session_id=session_id,
                execution_time=time.time() - start_time,
                quality_improvement=0.0,
                error_message=str(e)
            )
        
        # Finalize and return results
        return await self._finalize_ca_result(
            session_id, start_time, phase, connections_created, connections_pruned,
            total_operations, initial_quality
        )
    
    async def _finalize_ca_result(self, session_id: str, start_time: float, phase: CAPhase,
                                connections_created: int, connections_pruned: int,
                                total_operations: int, initial_quality: QualityMetrics,
                                error_message: str = None) -> CAResult:
        """Finalize CA results with quality analysis"""
        
        execution_time = time.time() - start_time
        
        # Get final graph state and quality
        final_stats = await self._get_graph_stats()
        final_quality = self.quality_analyzer.calculate_quality_metrics(final_stats)
        
        # Calculate quality improvement
        quality_improvement = final_quality.quality_score - initial_quality.quality_score
        
        # Track quality change
        quality_changes = self.quality_analyzer.track_quality_change(initial_quality, final_quality)
        
        # End monitoring session
        session_metrics = self.monitor.end_session()
        
        # Create result
        result = CAResult(
            success=error_message is None and not self.monitor.is_emergency_stopped(),
            connections_created=connections_created,
            connections_pruned=connections_pruned,
            total_operations=total_operations,
            phase=phase,
            session_id=session_id,
            execution_time=execution_time,
            quality_improvement=quality_improvement,
            error_message=error_message,
            emergency_stop=self.monitor.is_emergency_stopped()
        )
        
        # Log comprehensive results
        net_connections = connections_created - connections_pruned
        logger.info(f"✅ Semantic CA Completed: {phase.value}")
        logger.info(f"   Operations: {total_operations} ({connections_created} created, {connections_pruned} pruned)")
        logger.info(f"   Net change: {net_connections:+d} connections")
        logger.info(f"   Quality change: {quality_improvement:+.3f}")
        logger.info(f"   Duration: {execution_time:.2f}s")
        
        if quality_improvement > 0:
            logger.info(f"🎉 Graph quality improved!")
        elif quality_improvement < -0.1:
            logger.warning(f"⚠️ Graph quality declined significantly")
        
        return result
    
    def _handle_emergency_stop(self, reason: str):
        """Handle emergency stop events"""
        logger.error(f"🚨 Semantic CA Emergency Stop: {reason}")
        # Could implement additional emergency procedures here
    
    async def run_dream_ca_cycle(self, pre_dream_params: Optional[CAParameters] = None,
                               post_dream_params: Optional[CAParameters] = None) -> Tuple[CAResult, CAResult]:
        """
        Run a complete dream CA cycle: pre-dream exploration + post-dream consolidation
        """
        logger.info("🌙 Starting Dream CA Cycle")
        
        # Pre-dream CA (exploration)
        logger.info("🔍 Pre-dream exploration phase")
        pre_result = await self.apply_semantic_ca_rules(CAPhase.PRE_DREAM, pre_dream_params)
        
        if not pre_result.success:
            logger.error("❌ Pre-dream CA failed, skipping post-dream")
            return pre_result, CAResult(
                success=False, connections_created=0, connections_pruned=0,
                total_operations=0, phase=CAPhase.POST_DREAM, session_id="skipped",
                execution_time=0.0, quality_improvement=0.0,
                error_message="Pre-dream CA failed"
            )
        
        # Small delay between phases
        await asyncio.sleep(2.0)
        
        # Post-dream CA (consolidation)
        logger.info("🔄 Post-dream consolidation phase")
        post_result = await self.apply_semantic_ca_rules(CAPhase.POST_DREAM, post_dream_params)
        
        # Summary
        total_created = pre_result.connections_created + post_result.connections_created
        total_pruned = pre_result.connections_pruned + post_result.connections_pruned
        total_quality_change = pre_result.quality_improvement + post_result.quality_improvement
        
        logger.info(f"🌙 Dream CA Cycle Complete:")
        logger.info(f"   Total created: {total_created}")
        logger.info(f"   Total pruned: {total_pruned}")
        logger.info(f"   Net change: {total_created - total_pruned:+d}")
        logger.info(f"   Quality change: {total_quality_change:+.3f}")
        
        return pre_result, post_result
    
    def get_ca_status(self) -> Dict[str, Any]:
        """Get current CA system status"""
        return {
            "current_phase": self.current_phase.value,
            "monitor_status": self.monitor.get_current_metrics(),
            "quality_trend": self.quality_analyzer.get_quality_trend(),
            "emergency_stopped": self.monitor.is_emergency_stopped(),
            "parameter_effectiveness": self.adaptive_params.analyze_parameter_effectiveness()
        }
    
    def reset_emergency_state(self):
        """Reset emergency stop state (use with caution)"""
        self.monitor.reset_emergency_stop()
        logger.info("🔄 CA emergency state reset")

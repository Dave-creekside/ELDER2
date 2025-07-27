"""
Quality Analysis System for Cellular Automata
Monitors semantic coherence and connection quality
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger("ca-quality")

@dataclass
class QualityMetrics:
    """Container for CA quality metrics"""
    avg_semantic_weight: float
    min_semantic_weight: float
    max_semantic_weight: float
    connection_density: float
    cluster_coherence: float
    hub_concentration: float
    semantic_entropy: float
    quality_score: float

@dataclass
class ConnectionCandidate:
    """Represents a potential connection with quality metrics"""
    concept1: str
    concept2: str
    semantic_similarity: float
    common_neighbors: int
    quality_score: float
    creation_priority: int

class QualityAnalyzer:
    """
    Analyzes semantic quality and connection coherence for CA operations
    """
    
    def __init__(self):
        self.quality_history = []
        self.connection_patterns = defaultdict(list)
        
    def analyze_connection_candidate(self, concept1: str, concept2: str, 
                                   semantic_similarity: float, 
                                   common_neighbors: int,
                                   existing_connections1: int,
                                   existing_connections2: int) -> ConnectionCandidate:
        """
        Analyze the quality of a potential connection
        """
        # Base quality from semantic similarity
        quality_score = semantic_similarity
        
        # Boost quality based on common neighbors (structural support)
        neighbor_boost = min(common_neighbors * 0.1, 0.3)
        quality_score += neighbor_boost
        
        # Penalize hub formation (prevent single nodes from dominating)
        hub_penalty = 0.0
        if existing_connections1 > 10:
            hub_penalty += (existing_connections1 - 10) * 0.02
        if existing_connections2 > 10:
            hub_penalty += (existing_connections2 - 10) * 0.02
        quality_score -= min(hub_penalty, 0.4)
        
        # Calculate creation priority (higher = create first)
        priority = int(quality_score * 1000)
        if common_neighbors >= 3:
            priority += 100
        if semantic_similarity >= 0.8:
            priority += 200
            
        return ConnectionCandidate(
            concept1=concept1,
            concept2=concept2,
            semantic_similarity=semantic_similarity,
            common_neighbors=common_neighbors,
            quality_score=max(0.0, quality_score),
            creation_priority=priority
        )
    
    def should_create_connection(self, candidate: ConnectionCandidate, 
                               min_quality: float = 0.6,
                               max_connections_per_node: int = 15) -> bool:
        """
        Determine if a connection candidate should be created
        """
        # Quality threshold check
        if candidate.quality_score < min_quality:
            return False
            
        # Semantic similarity threshold check
        if candidate.semantic_similarity < min_quality:
            return False
            
        # Common neighbor requirement
        if candidate.common_neighbors < 2:
            return False
            
        # Hub prevention - this would need connection counts from the graph
        # For now, approve if quality checks pass
        return True
    
    def calculate_pruning_threshold(self, current_metrics: QualityMetrics,
                                   target_density: float = 0.1) -> float:
        """
        Calculate dynamic pruning threshold based on current graph quality
        """
        base_threshold = 0.3  # Baseline pruning threshold
        
        # If density is too high, be more aggressive
        if current_metrics.connection_density > target_density:
            density_factor = current_metrics.connection_density / target_density
            base_threshold += (density_factor - 1.0) * 0.2
            
        # If average quality is low, be more aggressive
        if current_metrics.avg_semantic_weight < 0.5:
            quality_factor = 0.5 / current_metrics.avg_semantic_weight
            base_threshold += (quality_factor - 1.0) * 0.1
            
        # If we have too many hubs, be more aggressive
        if current_metrics.hub_concentration > 0.3:
            hub_factor = current_metrics.hub_concentration / 0.3
            base_threshold += (hub_factor - 1.0) * 0.15
            
        return min(base_threshold, 0.8)  # Cap at 0.8 to avoid over-pruning
    
    def assess_semantic_coherence(self, connections: List[Dict[str, Any]]) -> float:
        """
        Assess the semantic coherence of a set of connections
        """
        if not connections:
            return 0.0
            
        weights = [conn.get('weight', 0.0) for conn in connections]
        
        if not weights:
            return 0.0
            
        # Calculate variance in weights (lower variance = more coherent)
        mean_weight = sum(weights) / len(weights)
        variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
        
        # Coherence is inverse of variance, normalized
        coherence = 1.0 / (1.0 + variance)
        
        # Boost coherence if average weight is high
        if mean_weight > 0.6:
            coherence *= 1.2
            
        return min(coherence, 1.0)
    
    def calculate_quality_metrics(self, graph_stats: Dict[str, Any]) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics for the current graph state
        """
        # Extract basic stats
        concept_count = graph_stats.get('concept_count', 0)
        relationship_count = graph_stats.get('semantic_relationships', 0)
        avg_weight = graph_stats.get('avg_semantic_weight', 0.0) or 0.0
        min_weight = graph_stats.get('min_semantic_weight', 0.0) or 0.0
        max_weight = graph_stats.get('max_semantic_weight', 0.0) or 0.0
        
        # Calculate connection density
        max_possible_connections = concept_count * (concept_count - 1) / 2 if concept_count > 1 else 1
        connection_density = relationship_count / max_possible_connections if max_possible_connections > 0 else 0.0
        
        # Estimate cluster coherence (simplified)
        cluster_coherence = min(avg_weight, 1.0) if avg_weight > 0 else 0.0
        
        # Estimate hub concentration (simplified)
        # In a well-distributed graph, we'd expect avg connections per node
        avg_connections_per_node = (2 * relationship_count) / concept_count if concept_count > 0 else 0.0
        hub_concentration = min(avg_connections_per_node / 10.0, 1.0)  # Normalize to 0-1
        
        # Calculate semantic entropy (diversity of weights)
        semantic_entropy = 0.0
        if max_weight > min_weight and avg_weight > 0:
            weight_range = max_weight - min_weight
            semantic_entropy = weight_range / max_weight  # Normalized diversity
            
        # Overall quality score
        quality_score = (
            avg_weight * 0.4 +  # 40% semantic quality
            cluster_coherence * 0.3 +  # 30% coherence
            (1.0 - connection_density) * 0.2 +  # 20% sparsity (avoid over-connection)
            semantic_entropy * 0.1  # 10% diversity
        )
        
        return QualityMetrics(
            avg_semantic_weight=avg_weight,
            min_semantic_weight=min_weight,
            max_semantic_weight=max_weight,
            connection_density=connection_density,
            cluster_coherence=cluster_coherence,
            hub_concentration=hub_concentration,
            semantic_entropy=semantic_entropy,
            quality_score=quality_score
        )
    
    def track_quality_change(self, before_metrics: QualityMetrics, 
                           after_metrics: QualityMetrics) -> Dict[str, Any]:
        """
        Track quality changes from a CA operation
        """
        changes = {
            'avg_semantic_weight_change': after_metrics.avg_semantic_weight - before_metrics.avg_semantic_weight,
            'connection_density_change': after_metrics.connection_density - before_metrics.connection_density,
            'quality_score_change': after_metrics.quality_score - before_metrics.quality_score,
            'cluster_coherence_change': after_metrics.cluster_coherence - before_metrics.cluster_coherence,
            'improvement': after_metrics.quality_score > before_metrics.quality_score
        }
        
        # Store in history
        self.quality_history.append({
            'timestamp': time.time(),
            'before': before_metrics,
            'after': after_metrics,
            'changes': changes
        })
        
        # Keep history manageable
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-50:]
            
        return changes
    
    def get_quality_trend(self, lookback_count: int = 10) -> str:
        """
        Analyze recent quality trends
        """
        if len(self.quality_history) < 2:
            return "insufficient_data"
            
        recent_history = self.quality_history[-lookback_count:]
        improvements = sum(1 for h in recent_history if h['changes']['improvement'])
        
        improvement_rate = improvements / len(recent_history)
        
        if improvement_rate >= 0.7:
            return "improving"
        elif improvement_rate >= 0.3:
            return "stable"
        else:
            return "declining"
    
    def recommend_ca_parameters(self, current_metrics: QualityMetrics) -> Dict[str, Any]:
        """
        Recommend CA parameters based on current graph quality
        """
        recommendations = {}
        
        # Semantic similarity threshold
        if current_metrics.avg_semantic_weight < 0.5:
            recommendations['min_similarity'] = 0.7  # Be more selective
        elif current_metrics.avg_semantic_weight > 0.7:
            recommendations['min_similarity'] = 0.6  # Can be more exploratory
        else:
            recommendations['min_similarity'] = 0.65
            
        # Common neighbors threshold
        if current_metrics.connection_density > 0.2:
            recommendations['common_neighbors_threshold'] = 3  # Require more support
        else:
            recommendations['common_neighbors_threshold'] = 2
            
        # Pruning threshold
        recommendations['prune_threshold'] = self.calculate_pruning_threshold(current_metrics)
        
        # Operation limits
        if current_metrics.connection_density > 0.15:
            recommendations['max_operations'] = 200  # Conservative
        else:
            recommendations['max_operations'] = 500  # More exploratory
            
        return recommendations

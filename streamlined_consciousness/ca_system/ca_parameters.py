"""
Adaptive Parameter System for Cellular Automata
Dynamically adjusts CA parameters based on graph state and performance
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .ca_quality import QualityMetrics, QualityAnalyzer

logger = logging.getLogger("ca-parameters")

class CAPhase(Enum):
    PRE_DREAM = "pre_dream"
    DURING_DREAM = "during_dream"  
    POST_DREAM = "post_dream"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"

@dataclass
class CAParameters:
    """Container for CA operation parameters"""
    # Core thresholds
    min_similarity: float
    common_neighbors_threshold: int
    prune_threshold: float
    max_operations: int
    
    # Connection limits
    max_connections_per_node: int
    max_new_connections: int
    
    # Quality controls
    min_quality_score: float
    hub_prevention_threshold: int
    
    # Performance controls
    max_operations_per_second: float
    operation_timeout: float
    
    # Metadata
    phase: CAPhase
    adaptation_reason: str = ""

class AdaptiveParameters:
    """
    Dynamically adapts CA parameters based on graph state and performance
    """
    
    def __init__(self):
        self.parameter_history = []
        self.quality_analyzer = QualityAnalyzer()
        self.base_parameters = self._get_base_parameters()
        
    def _get_base_parameters(self) -> Dict[CAPhase, CAParameters]:
        """
        Define base parameter sets for different CA phases
        """
        return {
            CAPhase.PRE_DREAM: CAParameters(
                min_similarity=0.25,  # Very lenient for maximum exploration
                common_neighbors_threshold=1,  # More exploratory discovery
                prune_threshold=0.2,  # Light pruning to keep exploration space open
                max_operations=3000,  # Use full exploration capacity
                max_connections_per_node=12,  # Allow more hub formation for exploration
                max_new_connections=200,  # Much higher exploration limit
                min_quality_score=0.5,  # Balanced quality threshold
                hub_prevention_threshold=15,  # More permissive for exploration
                max_operations_per_second=50.0,  # Much faster exploration for M3
                operation_timeout=90.0,  # More time for complex operations
                phase=CAPhase.PRE_DREAM,
                adaptation_reason="Exploratory pre-dream parameters for abstract connections"
            ),
            
            CAPhase.POST_DREAM: CAParameters(
                min_similarity=0.4,  # Much more permissive for abstract concepts
                common_neighbors_threshold=2,  # Lower requirement to allow more connections
                prune_threshold=0.3,  # Less aggressive pruning
                max_operations=1500,  # Allow more consolidation work
                max_connections_per_node=10,
                max_new_connections=50,  # Focus more on pruning than creating
                min_quality_score=0.6,
                hub_prevention_threshold=15,
                max_operations_per_second=30.0,  # Faster consolidation for M3
                operation_timeout=120.0,
                phase=CAPhase.POST_DREAM,
                adaptation_reason="Moderate post-dream consolidation - prune <50%"
            ),
            
            CAPhase.MAINTENANCE: CAParameters(
                min_similarity=0.65,
                common_neighbors_threshold=2,
                prune_threshold=0.35,
                max_operations=200,
                max_connections_per_node=6,
                max_new_connections=30,
                min_quality_score=0.65,
                hub_prevention_threshold=10,
                max_operations_per_second=3.0,
                operation_timeout=45.0,
                phase=CAPhase.MAINTENANCE,
                adaptation_reason="Base maintenance parameters"
            ),
            
            CAPhase.EMERGENCY: CAParameters(
                min_similarity=0.8,  # Very strict
                common_neighbors_threshold=4,  # High support requirement
                prune_threshold=0.5,  # Aggressive pruning
                max_operations=100,  # Very limited
                max_connections_per_node=5,
                max_new_connections=10,
                min_quality_score=0.75,
                hub_prevention_threshold=8,
                max_operations_per_second=2.0,
                operation_timeout=30.0,
                phase=CAPhase.EMERGENCY,
                adaptation_reason="Emergency constraint parameters"
            )
        }
    
    def adapt_parameters(self, phase: CAPhase, current_metrics: QualityMetrics,
                        session_history: List[Dict[str, Any]] = None,
                        performance_issues: List[str] = None) -> CAParameters:
        """
        Adapt parameters based on current graph state and performance
        """
        # Start with base parameters for the phase
        base_params = self.base_parameters[phase]
        adapted_params = CAParameters(
            min_similarity=base_params.min_similarity,
            common_neighbors_threshold=base_params.common_neighbors_threshold,
            prune_threshold=base_params.prune_threshold,
            max_operations=base_params.max_operations,
            max_connections_per_node=base_params.max_connections_per_node,
            max_new_connections=base_params.max_new_connections,
            min_quality_score=base_params.min_quality_score,
            hub_prevention_threshold=base_params.hub_prevention_threshold,
            max_operations_per_second=base_params.max_operations_per_second,
            operation_timeout=base_params.operation_timeout,
            phase=phase,
            adaptation_reason="Base parameters"
        )
        
        adaptations = []
        
        # Adapt based on graph density
        if current_metrics.connection_density > 0.2:
            adapted_params.min_similarity = min(adapted_params.min_similarity + 0.1, 0.9)
            adapted_params.prune_threshold = min(adapted_params.prune_threshold + 0.1, 0.7)
            adapted_params.max_operations = max(adapted_params.max_operations // 2, 50)
            adaptations.append("high_density")
        elif current_metrics.connection_density < 0.05:
            adapted_params.min_similarity = max(adapted_params.min_similarity - 0.05, 0.5)
            adapted_params.max_operations = min(adapted_params.max_operations * 1.5, 1000)
            adaptations.append("low_density")
        
        # Adapt based on semantic quality
        if current_metrics.avg_semantic_weight < 0.4:
            adapted_params.min_similarity = min(adapted_params.min_similarity + 0.15, 0.9)
            adapted_params.min_quality_score = min(adapted_params.min_quality_score + 0.1, 0.8)
            adapted_params.prune_threshold = min(adapted_params.prune_threshold + 0.15, 0.8)
            adaptations.append("low_quality")
        elif current_metrics.avg_semantic_weight > 0.7:
            adapted_params.min_similarity = max(adapted_params.min_similarity - 0.05, 0.5)
            adapted_params.max_operations = min(adapted_params.max_operations * 1.2, 1000)
            adaptations.append("high_quality")
        
        # Adapt based on hub concentration
        if current_metrics.hub_concentration > 0.4:
            adapted_params.hub_prevention_threshold = max(adapted_params.hub_prevention_threshold - 3, 5)
            adapted_params.max_connections_per_node = max(adapted_params.max_connections_per_node - 2, 3)
            adapted_params.common_neighbors_threshold = min(adapted_params.common_neighbors_threshold + 1, 5)
            adaptations.append("high_hub_concentration")
        
        # Adapt based on performance issues
        if performance_issues:
            if "rate_limit_exceeded" in performance_issues:
                adapted_params.max_operations_per_second = max(adapted_params.max_operations_per_second * 0.7, 1.0)
                adapted_params.max_operations = max(adapted_params.max_operations // 2, 50)
                adaptations.append("rate_limiting")
            
            if "connection_explosion" in performance_issues:
                adapted_params.min_similarity = min(adapted_params.min_similarity + 0.2, 0.9)
                adapted_params.max_operations = max(adapted_params.max_operations // 3, 25)
                adapted_params.max_new_connections = max(adapted_params.max_new_connections // 3, 5)
                adaptations.append("explosion_prevention")
            
            if "low_success_rate" in performance_issues:
                adapted_params.operation_timeout = min(adapted_params.operation_timeout * 1.5, 300.0)
                adapted_params.max_operations_per_second = max(adapted_params.max_operations_per_second * 0.8, 1.0)
                adaptations.append("success_rate_optimization")
        
        # Adapt based on session history
        if session_history and len(session_history) >= 3:
            recent_sessions = session_history[-3:]
            
            # Check for declining performance
            quality_trend = [s.get('avg_semantic_weight', 0.5) for s in recent_sessions]
            if len(quality_trend) >= 2 and quality_trend[-1] < quality_trend[0] - 0.1:
                adapted_params.min_similarity = min(adapted_params.min_similarity + 0.1, 0.9)
                adapted_params.prune_threshold = min(adapted_params.prune_threshold + 0.1, 0.7)
                adaptations.append("declining_quality_trend")
            
            # Check for emergency stops
            emergency_stops = sum(s.get('emergency_stops', 0) for s in recent_sessions)
            if emergency_stops > 0:
                adapted_params.max_operations = max(adapted_params.max_operations // 2, 25)
                adapted_params.max_operations_per_second = max(adapted_params.max_operations_per_second * 0.5, 1.0)
                adaptations.append("emergency_stop_prevention")
        
        # Update adaptation reason
        if adaptations:
            adapted_params.adaptation_reason = f"Adapted for: {', '.join(adaptations)}"
        
        # Store in history
        self.parameter_history.append({
            'timestamp': time.time(),
            'phase': phase,
            'original_params': base_params,
            'adapted_params': adapted_params,
            'current_metrics': current_metrics,
            'adaptations': adaptations
        })
        
        # Keep history manageable
        if len(self.parameter_history) > 100:
            self.parameter_history = self.parameter_history[-50:]
        
        logger.info(f"📊 Parameters adapted for {phase.value}: {adaptations}")
        
        return adapted_params
    
    def get_emergency_parameters(self, reason: str = "generic") -> CAParameters:
        """
        Get ultra-conservative parameters for emergency situations
        """
        emergency_params = self.base_parameters[CAPhase.EMERGENCY]
        
        # Make even more conservative based on emergency type
        if "explosion" in reason.lower():
            emergency_params.max_operations = 25
            emergency_params.min_similarity = 0.9
            emergency_params.max_new_connections = 5
        elif "rate" in reason.lower():
            emergency_params.max_operations_per_second = 1.0
            emergency_params.operation_timeout = 15.0
        elif "quality" in reason.lower():
            emergency_params.min_similarity = 0.85
            emergency_params.min_quality_score = 0.8
            emergency_params.prune_threshold = 0.6
        
        emergency_params.adaptation_reason = f"Emergency: {reason}"
        
        return emergency_params
    
    def recommend_phase_transition(self, current_phase: CAPhase, 
                                 current_metrics: QualityMetrics,
                                 session_performance: Dict[str, Any]) -> Tuple[CAPhase, str]:
        """
        Recommend when to transition between CA phases
        """
        # Emergency transition conditions
        if (current_metrics.connection_density > 0.3 or 
            current_metrics.avg_semantic_weight < 0.3 or
            session_performance.get('emergency_stops', 0) > 0):
            return CAPhase.EMERGENCY, "Graph instability detected"
        
        # Normal phase transitions
        if current_phase == CAPhase.PRE_DREAM:
            # Stay in pre-dream until we've done sufficient exploration
            ops_completed = session_performance.get('operations_successful', 0)
            if ops_completed >= 200 or current_metrics.connection_density > 0.15:
                return CAPhase.MAINTENANCE, "Pre-dream exploration complete"
        
        elif current_phase == CAPhase.POST_DREAM:
            # Stay in post-dream until consolidation is complete
            pruning_ratio = (session_performance.get('connections_pruned', 0) / 
                           max(session_performance.get('connections_created', 1), 1))
            if pruning_ratio > 2.0 and current_metrics.connection_density < 0.15:
                return CAPhase.MAINTENANCE, "Post-dream consolidation complete"
        
        elif current_phase == CAPhase.EMERGENCY:
            # Exit emergency when metrics improve
            if (current_metrics.connection_density < 0.2 and 
                current_metrics.avg_semantic_weight > 0.4 and
                session_performance.get('emergency_stops', 0) == 0):
                return CAPhase.MAINTENANCE, "Emergency conditions resolved"
        
        # Stay in current phase
        return current_phase, f"Continuing {current_phase.value}"
    
    def get_parameter_summary(self, params: CAParameters) -> Dict[str, Any]:
        """
        Get a human-readable summary of parameters
        """
        return {
            "phase": params.phase.value,
            "semantic_threshold": params.min_similarity,
            "neighbor_requirement": params.common_neighbors_threshold,
            "pruning_aggressiveness": params.prune_threshold,
            "operation_limit": params.max_operations,
            "connection_limits": {
                "per_node": params.max_connections_per_node,
                "total_new": params.max_new_connections
            },
            "quality_controls": {
                "min_score": params.min_quality_score,
                "hub_threshold": params.hub_prevention_threshold
            },
            "performance_controls": {
                "max_ops_per_sec": params.max_operations_per_second,
                "timeout": params.operation_timeout
            },
            "adaptation_reason": params.adaptation_reason
        }
    
    def analyze_parameter_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze how effective recent parameter adaptations have been
        """
        if len(self.parameter_history) < 3:
            return {"message": "Insufficient history for analysis"}
        
        recent_adaptations = self.parameter_history[-10:]
        
        # Analyze adaptation frequency
        adaptation_types = []
        for entry in recent_adaptations:
            adaptation_types.extend(entry['adaptations'])
        
        from collections import Counter
        adaptation_frequency = Counter(adaptation_types)
        
        # Analyze quality trends after adaptations
        quality_improvements = 0
        for i, entry in enumerate(recent_adaptations[1:], 1):
            prev_quality = recent_adaptations[i-1]['current_metrics'].avg_semantic_weight
            curr_quality = entry['current_metrics'].avg_semantic_weight
            if curr_quality > prev_quality:
                quality_improvements += 1
        
        improvement_rate = quality_improvements / max(len(recent_adaptations) - 1, 1)
        
        return {
            "adaptations_analyzed": len(recent_adaptations),
            "most_common_adaptations": dict(adaptation_frequency.most_common(5)),
            "quality_improvement_rate": round(improvement_rate, 3),
            "effectiveness": "high" if improvement_rate > 0.6 else "medium" if improvement_rate > 0.3 else "low",
            "recommendations": self._generate_parameter_recommendations(adaptation_frequency, improvement_rate)
        }
    
    def _generate_parameter_recommendations(self, adaptation_frequency: Dict[str, int], 
                                          improvement_rate: float) -> List[str]:
        """
        Generate recommendations for parameter tuning
        """
        recommendations = []
        
        if improvement_rate < 0.3:
            recommendations.append("Consider more conservative base parameters")
        
        if adaptation_frequency.get('high_density', 0) > 3:
            recommendations.append("Base parameters may be too permissive for connection creation")
        
        if adaptation_frequency.get('low_quality', 0) > 3:
            recommendations.append("Consider raising minimum similarity thresholds")
        
        if adaptation_frequency.get('emergency_stop_prevention', 0) > 1:
            recommendations.append("Operation limits may be too high")
        
        if improvement_rate > 0.7:
            recommendations.append("Current adaptation strategy is working well")
        
        return recommendations

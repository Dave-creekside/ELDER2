"""
Real-time Monitoring System for Cellular Automata Operations
Provides safety circuits and performance tracking
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger("ca-monitor")

class CAOperationStatus(Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    CREATING_CONNECTIONS = "creating_connections"
    PRUNING = "pruning"
    COMPLETED = "completed"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"

@dataclass
class CAOperationRecord:
    """Record of a single CA operation"""
    timestamp: float
    operation_type: str  # "creation" or "pruning"
    concept1: Optional[str] = None
    concept2: Optional[str] = None
    semantic_weight: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class CASessionMetrics:
    """Metrics for a complete CA session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    operations_attempted: int = 0
    operations_successful: int = 0
    connections_created: int = 0
    connections_pruned: int = 0
    avg_semantic_weight: float = 0.0
    max_operations_per_second: float = 0.0
    emergency_stops: int = 0
    quality_improvement: float = 0.0

class CAMonitor:
    """
    Real-time monitor for CA operations with safety circuits
    """
    
    def __init__(self, max_operations_per_second: float = 100.0,
                 max_total_operations: int = 5000,
                 quality_decline_threshold: float = -0.2):
        
        # Safety thresholds
        self.max_operations_per_second = max_operations_per_second
        self.max_total_operations = max_total_operations
        self.quality_decline_threshold = quality_decline_threshold
        
        # State tracking
        self.current_status = CAOperationStatus.IDLE
        self.current_session: Optional[CASessionMetrics] = None
        self.operation_history = deque(maxlen=1000)
        self.session_history = []
        
        # Rate limiting
        self.operation_times = deque(maxlen=100)
        self.emergency_stop_flag = False
        self.emergency_stop_callbacks: List[Callable] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance tracking
        self.total_operations = 0
        self.total_sessions = 0
        self.avg_session_duration = 0.0
        
    def start_session(self, session_type: str = "dream_ca") -> str:
        """
        Start a new CA monitoring session
        """
        with self.lock:
            session_id = f"{session_type}_{int(time.time())}_{self.total_sessions}"
            
            self.current_session = CASessionMetrics(
                session_id=session_id,
                start_time=time.time()
            )
            
            self.current_status = CAOperationStatus.ANALYZING
            self.emergency_stop_flag = False
            self.total_sessions += 1
            
            logger.info(f"🔍 CA Session Started: {session_id}")
            return session_id
    
    def end_session(self) -> Optional[CASessionMetrics]:
        """
        End the current CA session and return metrics
        """
        with self.lock:
            if not self.current_session:
                return None
                
            self.current_session.end_time = time.time()
            self.current_status = CAOperationStatus.COMPLETED
            
            # Calculate session duration
            duration = self.current_session.end_time - self.current_session.start_time
            
            # Update rolling average
            if self.avg_session_duration == 0:
                self.avg_session_duration = duration
            else:
                self.avg_session_duration = (self.avg_session_duration * 0.8) + (duration * 0.2)
            
            # Store in history
            session_copy = self.current_session
            self.session_history.append(session_copy)
            
            # Keep history manageable
            if len(self.session_history) > 50:
                self.session_history = self.session_history[-25:]
            
            logger.info(f"✅ CA Session Completed: {session_copy.session_id} "
                       f"({session_copy.operations_successful}/{session_copy.operations_attempted} ops, "
                       f"{duration:.2f}s)")
            
            self.current_session = None
            return session_copy
    
    def record_operation(self, operation_type: str, concept1: str = None, 
                        concept2: str = None, semantic_weight: float = None,
                        success: bool = True, error_message: str = None) -> bool:
        """
        Record a CA operation and check safety limits
        Returns False if emergency stop is triggered
        """
        current_time = time.time()
        
        with self.lock:
            # Record the operation
            operation = CAOperationRecord(
                timestamp=current_time,
                operation_type=operation_type,
                concept1=concept1,
                concept2=concept2,
                semantic_weight=semantic_weight,
                success=success,
                error_message=error_message
            )
            
            self.operation_history.append(operation)
            self.operation_times.append(current_time)
            self.total_operations += 1
            
            # Update session metrics
            if self.current_session:
                self.current_session.operations_attempted += 1
                if success:
                    self.current_session.operations_successful += 1
                    
                if operation_type == "creation":
                    self.current_session.connections_created += 1
                elif operation_type == "pruning":
                    self.current_session.connections_pruned += 1
                    
                if semantic_weight:
                    # Update rolling average semantic weight
                    if self.current_session.avg_semantic_weight == 0:
                        self.current_session.avg_semantic_weight = semantic_weight
                    else:
                        self.current_session.avg_semantic_weight = (
                            self.current_session.avg_semantic_weight * 0.9 + 
                            semantic_weight * 0.1
                        )
            
            # Check safety limits
            emergency_triggered = self._check_safety_limits()
            
            if emergency_triggered:
                self._trigger_emergency_stop("Safety limits exceeded")
                return False
                
            return True
    
    def _check_safety_limits(self) -> bool:
        """
        Check if any safety limits have been exceeded
        """
        current_time = time.time()
        
        # Check operations per second rate
        recent_ops = [t for t in self.operation_times if current_time - t <= 1.0]
        current_ops_per_second = len(recent_ops)
        
        if current_ops_per_second > self.max_operations_per_second:
            logger.warning(f"⚠️ Operation rate limit exceeded: {current_ops_per_second}/s > {self.max_operations_per_second}/s")
            return True
        
        # Check total operations in session
        if self.current_session and self.current_session.operations_attempted >= self.max_total_operations:
            logger.warning(f"⚠️ Total operation limit exceeded: {self.current_session.operations_attempted} > {self.max_total_operations}")
            return True
        
        # Check for consistent failures
        recent_operations = list(self.operation_history)[-20:]  # Last 20 operations
        if len(recent_operations) >= 10:
            failure_rate = sum(1 for op in recent_operations if not op.success) / len(recent_operations)
            if failure_rate > 0.5:
                logger.warning(f"⚠️ High failure rate detected: {failure_rate:.1%}")
                return True
        
        return False
    
    def _trigger_emergency_stop(self, reason: str):
        """
        Trigger emergency stop procedures
        """
        self.emergency_stop_flag = True
        self.current_status = CAOperationStatus.EMERGENCY_STOP
        
        if self.current_session:
            self.current_session.emergency_stops += 1
        
        logger.error(f"🚨 EMERGENCY STOP: {reason}")
        
        # Execute emergency callbacks
        for callback in self.emergency_stop_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Error in emergency stop callback: {e}")
    
    def add_emergency_stop_callback(self, callback: Callable[[str], None]):
        """
        Add a callback to be executed on emergency stop
        """
        self.emergency_stop_callbacks.append(callback)
    
    def is_emergency_stopped(self) -> bool:
        """
        Check if emergency stop is active
        """
        return self.emergency_stop_flag
    
    def reset_emergency_stop(self):
        """
        Reset emergency stop (use carefully!)
        """
        with self.lock:
            self.emergency_stop_flag = False
            if self.current_status == CAOperationStatus.EMERGENCY_STOP:
                self.current_status = CAOperationStatus.IDLE
        
        logger.info("🔄 Emergency stop reset")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current monitoring metrics
        """
        with self.lock:
            current_time = time.time()
            
            # Calculate current operations per second
            recent_ops = [t for t in self.operation_times if current_time - t <= 5.0]
            current_ops_per_second = len(recent_ops) / 5.0
            
            # Recent success rate
            recent_operations = list(self.operation_history)[-50:]
            success_rate = (sum(1 for op in recent_operations if op.success) / len(recent_operations)) if recent_operations else 1.0
            
            metrics = {
                "status": self.current_status.value,
                "emergency_stopped": self.emergency_stop_flag,
                "current_ops_per_second": round(current_ops_per_second, 2),
                "success_rate": round(success_rate, 3),
                "total_operations": self.total_operations,
                "total_sessions": self.total_sessions,
                "avg_session_duration": round(self.avg_session_duration, 2)
            }
            
            if self.current_session:
                session_duration = current_time - self.current_session.start_time
                metrics.update({
                    "current_session": {
                        "id": self.current_session.session_id,
                        "duration": round(session_duration, 2),
                        "operations_attempted": self.current_session.operations_attempted,
                        "operations_successful": self.current_session.operations_successful,
                        "connections_created": self.current_session.connections_created,
                        "connections_pruned": self.current_session.connections_pruned,
                        "avg_semantic_weight": round(self.current_session.avg_semantic_weight, 3)
                    }
                })
            
            return metrics
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """
        with self.lock:
            if not self.session_history:
                return {"message": "No session history available"}
            
            recent_sessions = self.session_history[-10:]  # Last 10 sessions
            
            # Calculate aggregated metrics
            total_operations = sum(s.operations_successful for s in recent_sessions)
            total_connections_created = sum(s.connections_created for s in recent_sessions)
            total_connections_pruned = sum(s.connections_pruned for s in recent_sessions)
            
            avg_session_success_rate = sum(
                s.operations_successful / s.operations_attempted if s.operations_attempted > 0 else 0
                for s in recent_sessions
            ) / len(recent_sessions)
            
            avg_semantic_quality = sum(s.avg_semantic_weight for s in recent_sessions) / len(recent_sessions)
            
            emergency_stops = sum(s.emergency_stops for s in recent_sessions)
            
            return {
                "sessions_analyzed": len(recent_sessions),
                "total_successful_operations": total_operations,
                "total_connections_created": total_connections_created,
                "total_connections_pruned": total_connections_pruned,
                "avg_session_success_rate": round(avg_session_success_rate, 3),
                "avg_semantic_quality": round(avg_semantic_quality, 3),
                "emergency_stops": emergency_stops,
                "avg_session_duration": round(self.avg_session_duration, 2),
                "performance_trend": self._calculate_performance_trend()
            }
    
    def _calculate_performance_trend(self) -> str:
        """
        Calculate overall performance trend
        """
        if len(self.session_history) < 3:
            return "insufficient_data"
        
        recent = self.session_history[-5:]
        older = self.session_history[-10:-5] if len(self.session_history) >= 10 else self.session_history[:-5]
        
        if not older:
            return "insufficient_data"
        
        recent_quality = sum(s.avg_semantic_weight for s in recent) / len(recent)
        older_quality = sum(s.avg_semantic_weight for s in older) / len(older)
        
        recent_success = sum(s.operations_successful / s.operations_attempted if s.operations_attempted > 0 else 0 for s in recent) / len(recent)
        older_success = sum(s.operations_successful / s.operations_attempted if s.operations_attempted > 0 else 0 for s in older) / len(older)
        
        quality_improvement = recent_quality - older_quality
        success_improvement = recent_success - older_success
        
        if quality_improvement > 0.05 and success_improvement > 0.05:
            return "improving"
        elif quality_improvement < -0.05 or success_improvement < -0.05:
            return "declining"
        else:
            return "stable"
    
    def should_continue_operations(self) -> tuple[bool, str]:
        """
        Determine if CA operations should continue
        Returns (should_continue, reason)
        """
        if self.emergency_stop_flag:
            return False, "Emergency stop active"
        
        if not self.current_session:
            return False, "No active session"
        
        current_time = time.time()
        session_duration = current_time - self.current_session.start_time
        
        # Check maximum session duration (safety limit)
        if session_duration > 300:  # 5 minutes max
            return False, "Session duration limit exceeded"
        
        # Check operation limits
        if self.current_session.operations_attempted >= self.max_total_operations:
            return False, "Operation limit reached"
        
        # Check rate limits
        recent_ops = [t for t in self.operation_times if current_time - t <= 1.0]
        if len(recent_ops) >= self.max_operations_per_second:
            return False, "Rate limit reached"
        
        return True, "Operations can continue"

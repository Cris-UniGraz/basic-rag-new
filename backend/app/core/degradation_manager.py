import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Callable, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from loguru import logger

from app.core.config import settings
from app.core.async_metadata_processor import async_metadata_processor


class DegradationLevel(Enum):
    """Levels of service degradation."""
    FULL = "full"                    # All features available
    ENHANCED = "enhanced"            # Full features with optimizations
    STANDARD = "standard"            # Standard features, some optimizations disabled
    REDUCED = "reduced"              # Reduced features, basic functionality
    MINIMAL = "minimal"              # Minimal functionality, emergency mode
    EMERGENCY = "emergency"          # Only critical operations


class ComponentStatus(Enum):
    """Status of system components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"


class DegradationTrigger(Enum):
    """Triggers for degradation."""
    COMPONENT_FAILURE = auto()
    PERFORMANCE_DEGRADATION = auto()
    RESOURCE_EXHAUSTION = auto()
    CIRCUIT_BREAKER_OPEN = auto()
    MANUAL_OVERRIDE = auto()
    HEALTH_CHECK_FAILURE = auto()
    LOAD_THRESHOLD_EXCEEDED = auto()


@dataclass
class ComponentHealth:
    """Health information for a system component."""
    name: str
    status: ComponentStatus
    last_check: datetime
    error_rate: float = 0.0
    response_time: float = 0.0
    availability: float = 100.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationEvent:
    """Event that triggered degradation."""
    timestamp: datetime
    trigger: DegradationTrigger
    component: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    level_before: DegradationLevel = DegradationLevel.FULL
    level_after: DegradationLevel = DegradationLevel.FULL


class DegradationStrategy(ABC):
    """Abstract base class for degradation strategies."""
    
    @abstractmethod
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the degradation strategy."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities in this mode."""
        pass
    
    @abstractmethod
    def get_restrictions(self) -> List[str]:
        """Get list of restrictions in this mode."""
        pass


class FullServiceStrategy(DegradationStrategy):
    """Full service with all features available."""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply full service strategy."""
        return {
            "retriever_types": ["base", "parent", "multi_query", "hyde", "bm25"],
            "reranking_enabled": True,
            "advanced_search": True,
            "caching_enabled": True,
            "parallel_processing": True,
            "ensemble_retrievers": True,
            "query_optimization": True,
            "semantic_search": True,
            "metadata_filtering": True,
            "response_enhancement": True
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "retriever_count": 5,
            "reranking": True,
            "parallel_retrieval": True,
            "advanced_query_processing": True,
            "full_metadata_support": True,
            "caching": True
        }
    
    def get_restrictions(self) -> List[str]:
        return []


class EnhancedServiceStrategy(DegradationStrategy):
    """Enhanced service with performance optimizations (intermediate mode)."""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply enhanced service strategy."""
        return {
            "retriever_types": ["base", "parent", "multi_query", "hyde"],
            "reranking_enabled": True,
            "advanced_search": True,
            "caching_enabled": True,
            "parallel_processing": True,
            "ensemble_retrievers": True,
            "query_optimization": True,
            "semantic_search": True,
            "metadata_filtering": True,
            "response_enhancement": False  # Disabled for performance
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "retriever_count": 4,
            "reranking": True,
            "parallel_retrieval": True,
            "advanced_query_processing": True,
            "full_metadata_support": True,
            "caching": True
        }
    
    def get_restrictions(self) -> List[str]:
        return ["Response enhancement disabled"]


class StandardServiceStrategy(DegradationStrategy):
    """Standard service with some features disabled."""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply standard service strategy."""
        return {
            "retriever_types": ["base", "parent", "multi_query"],
            "reranking_enabled": True,
            "advanced_search": False,
            "caching_enabled": True,
            "parallel_processing": True,
            "ensemble_retrievers": False,
            "query_optimization": True,
            "semantic_search": True,
            "metadata_filtering": True,
            "response_enhancement": False
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "retriever_count": 3,
            "reranking": True,
            "parallel_retrieval": True,
            "advanced_query_processing": False,
            "full_metadata_support": True,
            "caching": True
        }
    
    def get_restrictions(self) -> List[str]:
        return [
            "Advanced search disabled",
            "Ensemble retrievers disabled",
            "Response enhancement disabled"
        ]


class ReducedServiceStrategy(DegradationStrategy):
    """Reduced service with basic functionality."""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reduced service strategy."""
        return {
            "retriever_types": ["base", "parent"],
            "reranking_enabled": False,
            "advanced_search": False,
            "caching_enabled": True,
            "parallel_processing": False,
            "ensemble_retrievers": False,
            "query_optimization": False,
            "semantic_search": True,
            "metadata_filtering": False,
            "response_enhancement": False
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "retriever_count": 2,
            "reranking": False,
            "parallel_retrieval": False,
            "advanced_query_processing": False,
            "full_metadata_support": False,
            "caching": True
        }
    
    def get_restrictions(self) -> List[str]:
        return [
            "Reranking disabled",
            "Advanced search disabled",
            "Parallel processing disabled",
            "Query optimization disabled",
            "Metadata filtering disabled"
        ]


class MinimalServiceStrategy(DegradationStrategy):
    """Minimal service with only basic retrieval."""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply minimal service strategy."""
        return {
            "retriever_types": ["base"],
            "reranking_enabled": False,
            "advanced_search": False,
            "caching_enabled": False,
            "parallel_processing": False,
            "ensemble_retrievers": False,
            "query_optimization": False,
            "semantic_search": True,
            "metadata_filtering": False,
            "response_enhancement": False
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "retriever_count": 1,
            "reranking": False,
            "parallel_retrieval": False,
            "advanced_query_processing": False,
            "full_metadata_support": False,
            "caching": False
        }
    
    def get_restrictions(self) -> List[str]:
        return [
            "Only basic vector retrieval available",
            "All advanced features disabled",
            "Caching disabled",
            "Single retriever only"
        ]


class EmergencyServiceStrategy(DegradationStrategy):
    """Emergency service with cached responses only."""
    
    async def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply emergency service strategy."""
        return {
            "retriever_types": [],
            "reranking_enabled": False,
            "advanced_search": False,
            "caching_enabled": True,
            "parallel_processing": False,
            "ensemble_retrievers": False,
            "query_optimization": False,
            "semantic_search": False,
            "metadata_filtering": False,
            "response_enhancement": False,
            "cache_only": True
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "retriever_count": 0,
            "reranking": False,
            "parallel_retrieval": False,
            "advanced_query_processing": False,
            "full_metadata_support": False,
            "caching": True,
            "cache_only": True
        }
    
    def get_restrictions(self) -> List[str]:
        return [
            "Cache-only responses",
            "No live retrieval",
            "All real-time features disabled"
        ]


class DegradationManager:
    """
    Manages graceful degradation of service functionality using strategy pattern.
    
    Features:
    - Multiple degradation levels with different capabilities
    - Automatic degradation based on component health
    - Manual degradation control
    - Recovery detection and automatic upgrade
    - Strategy pattern for different service modes
    - Comprehensive monitoring and logging
    
    The strategy pattern allows different degradation strategies to be applied
    dynamically based on system health and performance metrics.
    """
    
    def __init__(self):
        """Initialize the degradation manager."""
        self.current_level = DegradationLevel.FULL
        self.strategies = {
            DegradationLevel.FULL: FullServiceStrategy(),
            DegradationLevel.ENHANCED: EnhancedServiceStrategy(),
            DegradationLevel.STANDARD: StandardServiceStrategy(),
            DegradationLevel.REDUCED: ReducedServiceStrategy(),
            DegradationLevel.MINIMAL: MinimalServiceStrategy(),
            DegradationLevel.EMERGENCY: EmergencyServiceStrategy()
        }
        
        # Component health tracking
        self.component_health: Dict[str, ComponentHealth] = {}
        self.degradation_history: List[DegradationEvent] = []
        self.max_history = 1000
        
        # State management
        self._state_lock = asyncio.Lock()
        self._manual_override = False
        self._override_level: Optional[DegradationLevel] = None
        self._override_timestamp: Optional[datetime] = None
        
        # Background monitoring
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._is_monitoring = False
        
        # Configuration
        self.auto_recovery_enabled = True
        self.recovery_check_interval = 30.0  # seconds
        self.degradation_thresholds = {
            DegradationLevel.ENHANCED: {
                "max_error_rate": 0.05,
                "max_response_time": 10.0,
                "min_availability": 98.0
            },
            DegradationLevel.STANDARD: {
                "max_error_rate": 0.10,
                "max_response_time": 15.0,
                "min_availability": 95.0
            },
            DegradationLevel.REDUCED: {
                "max_error_rate": 0.20,
                "max_response_time": 25.0,
                "min_availability": 90.0
            },
            DegradationLevel.MINIMAL: {
                "max_error_rate": 0.35,
                "max_response_time": 40.0,
                "min_availability": 80.0
            },
            DegradationLevel.EMERGENCY: {
                "max_error_rate": 0.50,
                "max_response_time": 60.0,
                "min_availability": 50.0
            }
        }
        
        logger.info("DegradationManager initialized")
    
    async def start_monitoring(self):
        """Start background monitoring for automatic degradation."""
        if self._is_monitoring:
            logger.warning("Degradation monitoring is already running")
            return
        
        self._is_monitoring = True
        self._shutdown_event.clear()
        
        # Start recovery monitoring task
        recovery_task = asyncio.create_task(self._recovery_monitoring_loop())
        self._background_tasks.add(recovery_task)
        recovery_task.add_done_callback(self._background_tasks.discard)
        
        logger.info("Degradation monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        if not self._is_monitoring:
            return
        
        logger.info("Stopping degradation monitoring...")
        
        self._is_monitoring = False
        self._shutdown_event.set()
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("Degradation monitoring stopped")
    
    async def update_component_health(
        self,
        component_name: str,
        status: ComponentStatus,
        error_rate: float = 0.0,
        response_time: float = 0.0,
        availability: float = 100.0,
        details: Optional[Dict[str, Any]] = None
    ):
        """Update health information for a component."""
        async with self._state_lock:
            self.component_health[component_name] = ComponentHealth(
                name=component_name,
                status=status,
                last_check=datetime.now(),
                error_rate=error_rate,
                response_time=response_time,
                availability=availability,
                details=details or {}
            )
        
        # Check if degradation is needed
        if not self._manual_override:
            await self._evaluate_degradation_need()
    
    async def _evaluate_degradation_need(self):
        """Evaluate if degradation is needed based on component health."""
        if not self.component_health:
            return
        
        # Calculate overall system health metrics
        total_components = len(self.component_health)
        healthy_components = sum(
            1 for health in self.component_health.values()
            if health.status == ComponentStatus.HEALTHY
        )
        
        avg_error_rate = sum(
            health.error_rate for health in self.component_health.values()
        ) / total_components
        
        avg_response_time = sum(
            health.response_time for health in self.component_health.values()
        ) / total_components
        
        avg_availability = sum(
            health.availability for health in self.component_health.values()
        ) / total_components
        
        # Determine appropriate degradation level
        target_level = self._determine_degradation_level(
            avg_error_rate, avg_response_time, avg_availability
        )
        
        if target_level != self.current_level:
            await self._initiate_degradation(
                target_level,
                DegradationTrigger.COMPONENT_FAILURE,
                details={
                    "avg_error_rate": avg_error_rate,
                    "avg_response_time": avg_response_time,
                    "avg_availability": avg_availability,
                    "healthy_components": f"{healthy_components}/{total_components}"
                }
            )
    
    def _determine_degradation_level(
        self,
        error_rate: float,
        response_time: float,
        availability: float
    ) -> DegradationLevel:
        """Determine appropriate degradation level based on metrics."""
        # Check from most degraded to least degraded
        for level in [
            DegradationLevel.EMERGENCY,
            DegradationLevel.MINIMAL,
            DegradationLevel.REDUCED,
            DegradationLevel.STANDARD,
            DegradationLevel.ENHANCED
        ]:
            thresholds = self.degradation_thresholds.get(level, {})
            
            if (error_rate <= thresholds.get("max_error_rate", 1.0) and
                response_time <= thresholds.get("max_response_time", 1000.0) and
                availability >= thresholds.get("min_availability", 0.0)):
                return level
        
        return DegradationLevel.FULL
    
    async def _initiate_degradation(
        self,
        target_level: DegradationLevel,
        trigger: DegradationTrigger,
        component: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initiate degradation to a specific level."""
        async with self._state_lock:
            if target_level == self.current_level:
                return
            
            previous_level = self.current_level
            
            # Create degradation event
            event = DegradationEvent(
                timestamp=datetime.now(),
                trigger=trigger,
                component=component,
                details=details or {},
                level_before=previous_level,
                level_after=target_level
            )
            
            # Update current level
            self.current_level = target_level
            
            # Add to history
            self.degradation_history.append(event)
            if len(self.degradation_history) > self.max_history:
                self.degradation_history.pop(0)
            
            # Log degradation
            log_level = "CRITICAL" if target_level == DegradationLevel.EMERGENCY else "WARNING"
            logger.log(
                log_level,
                f"Service degradation: {previous_level.value} → {target_level.value} "
                f"(trigger: {trigger.name})"
            )
            
            # Log async
            async_metadata_processor.log_async(
                log_level,
                "Service degradation initiated",
                {
                    "previous_level": previous_level.value,
                    "new_level": target_level.value,
                    "trigger": trigger.name,
                    "component": component,
                    "details": details
                },
                priority=3 if target_level == DegradationLevel.EMERGENCY else 2
            )
    
    async def manual_degrade(
        self,
        level: DegradationLevel,
        reason: str = "Manual override"
    ):
        """Manually set degradation level."""
        async with self._state_lock:
            self._manual_override = True
            self._override_level = level
            self._override_timestamp = datetime.now()
        
        await self._initiate_degradation(
            level,
            DegradationTrigger.MANUAL_OVERRIDE,
            details={"reason": reason}
        )
        
        logger.warning(f"Manual degradation to {level.value}: {reason}")
    
    async def clear_manual_override(self):
        """Clear manual override and return to automatic degradation."""
        async with self._state_lock:
            self._manual_override = False
            self._override_level = None
            self._override_timestamp = None
        
        logger.info("Manual override cleared, returning to automatic degradation")
        
        # Re-evaluate degradation need
        await self._evaluate_degradation_need()
    
    async def get_current_configuration(self) -> Dict[str, Any]:
        """Get current service configuration based on degradation level."""
        strategy = self.strategies.get(self.current_level)
        if not strategy:
            logger.error(f"No strategy found for level: {self.current_level}")
            return {}
        
        config = await strategy.apply({})
        
        # Add metadata
        config.update({
            "degradation_level": self.current_level.value,
            "capabilities": strategy.get_capabilities(),
            "restrictions": strategy.get_restrictions(),
            "manual_override": self._manual_override,
            "timestamp": datetime.now().isoformat()
        })
        
        return config
    
    async def _recovery_monitoring_loop(self):
        """Background loop for monitoring recovery conditions."""
        logger.info("Starting recovery monitoring loop")
        
        while not self._shutdown_event.is_set():
            try:
                if not self._manual_override and self.auto_recovery_enabled:
                    await self._check_recovery_conditions()
                
                await asyncio.sleep(self.recovery_check_interval)
                
            except Exception as e:
                logger.error(f"Error in recovery monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_recovery_conditions(self):
        """Check if conditions allow for service recovery."""
        if self.current_level == DegradationLevel.FULL:
            return
        
        # Check if we can upgrade to a better level
        if not self.component_health:
            return
        
        total_components = len(self.component_health)
        avg_error_rate = sum(
            health.error_rate for health in self.component_health.values()
        ) / total_components
        
        avg_response_time = sum(
            health.response_time for health in self.component_health.values()
        ) / total_components
        
        avg_availability = sum(
            health.availability for health in self.component_health.values()
        ) / total_components
        
        # Check if we can upgrade
        potential_level = self._determine_degradation_level(
            avg_error_rate, avg_response_time, avg_availability
        )
        
        # Only upgrade if conditions are stable (have been good for a while)
        if self._is_upgrade_safe(potential_level):
            await self._initiate_degradation(
                potential_level,
                DegradationTrigger.COMPONENT_FAILURE,  # Recovery
                details={
                    "recovery": True,
                    "avg_error_rate": avg_error_rate,
                    "avg_response_time": avg_response_time,
                    "avg_availability": avg_availability
                }
            )
    
    def _is_upgrade_safe(self, potential_level: DegradationLevel) -> bool:
        """Check if it's safe to upgrade to a better level."""
        if potential_level.value <= self.current_level.value:
            return False
        
        # Check recent history to ensure stability
        recent_events = [
            event for event in self.degradation_history[-10:]
            if (datetime.now() - event.timestamp).seconds < 300  # Last 5 minutes
        ]
        
        # Don't upgrade if there have been recent degradations
        if len(recent_events) > 2:
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current degradation manager status."""
        strategy = self.strategies.get(self.current_level)
        
        return {
            "current_level": self.current_level.value,
            "manual_override": self._manual_override,
            "override_level": self._override_level.value if self._override_level else None,
            "override_timestamp": self._override_timestamp.isoformat() if self._override_timestamp else None,
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "monitoring_active": self._is_monitoring,
            "capabilities": strategy.get_capabilities() if strategy else {},
            "restrictions": strategy.get_restrictions() if strategy else [],
            "component_health": {
                name: {
                    "status": health.status.value,
                    "error_rate": health.error_rate,
                    "response_time": health.response_time,
                    "availability": health.availability,
                    "last_check": health.last_check.isoformat()
                }
                for name, health in self.component_health.items()
            },
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "trigger": event.trigger.name,
                    "level_before": event.level_before.value,
                    "level_after": event.level_after.value,
                    "component": event.component,
                    "details": event.details
                }
                for event in self.degradation_history[-10:]
            ]
        }
    
    async def cleanup(self):
        """Clean up degradation manager resources."""
        await self.stop_monitoring()
        
        async with self._state_lock:
            self.component_health.clear()
            self.degradation_history.clear()
        
        logger.info("DegradationManager cleanup completed")


# Global degradation manager
degradation_manager = DegradationManager()


# Utility functions for easy integration
async def get_current_service_config() -> Dict[str, Any]:
    """Get current service configuration based on degradation level."""
    return await degradation_manager.get_current_configuration()


async def update_component_health_from_circuit_breaker(
    component_name: str,
    circuit_breaker_status: Dict[str, Any]
):
    """Update component health based on circuit breaker status."""
    state = circuit_breaker_status.get("state", "unknown")
    metrics = circuit_breaker_status.get("metrics", {})
    
    if state == "closed":
        status = ComponentStatus.HEALTHY
    elif state == "half_open":
        status = ComponentStatus.DEGRADED
    else:  # open
        status = ComponentStatus.FAILING
    
    await degradation_manager.update_component_health(
        component_name=component_name,
        status=status,
        error_rate=metrics.get("failure_rate", 0.0),
        response_time=metrics.get("avg_response_time", 0.0),
        availability=100.0 - (metrics.get("failure_rate", 0.0) * 100),
        details=circuit_breaker_status
    )


async def check_feature_availability(feature: str) -> bool:
    """Check if a specific feature is available in current degradation level."""
    config = await get_current_service_config()
    return config.get(feature, False)


async def get_available_retrievers() -> List[str]:
    """Get list of available retrievers in current degradation level."""
    config = await get_current_service_config()
    return config.get("retriever_types", [])
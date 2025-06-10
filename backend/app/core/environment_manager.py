"""
Environment configuration manager for different deployment environments.

This module provides utilities to manage environment-specific configurations,
validate settings, and apply optimizations based on the deployment environment.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger

from app.core.config import settings


class Environment(Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EnvironmentProfile:
    """Environment-specific configuration profile."""
    name: str
    environment: Environment
    description: str
    config_overrides: Dict[str, Any]
    required_vars: List[str]
    optional_vars: List[str]
    resource_limits: Dict[str, Any]
    security_settings: Dict[str, Any]
    performance_settings: Dict[str, Any]


class EnvironmentManager:
    """
    Manages environment-specific configurations and validations.
    
    Features:
    - Environment detection and validation
    - Configuration profile management
    - Resource limit enforcement
    - Security settings validation
    - Performance optimization recommendations
    """
    
    def __init__(self):
        """Initialize the environment manager."""
        self.current_environment = self._detect_environment()
        self.profiles = self._load_environment_profiles()
        self.current_profile = self.profiles.get(self.current_environment)
        
        logger.info(f"Environment manager initialized for: {self.current_environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Detect the current deployment environment."""
        # Check environment variable first
        env_var = os.getenv("ENVIRONMENT", "").lower()
        
        if env_var in ["production", "prod"]:
            return Environment.PRODUCTION
        elif env_var in ["staging", "stage"]:
            return Environment.STAGING
        elif env_var in ["development", "dev", "local"]:
            return Environment.DEVELOPMENT
        
        # Check settings flags
        if settings.PRODUCTION_MODE:
            return Environment.PRODUCTION
        elif settings.STAGING_MODE:
            return Environment.STAGING
        else:
            return Environment.DEVELOPMENT
    
    def _load_environment_profiles(self) -> Dict[Environment, EnvironmentProfile]:
        """Load environment-specific configuration profiles."""
        profiles = {}
        
        # Development Profile
        profiles[Environment.DEVELOPMENT] = EnvironmentProfile(
            name="Development",
            environment=Environment.DEVELOPMENT,
            description="Local development environment with relaxed settings",
            config_overrides={
                "LOG_LEVEL": "DEBUG",
                "DEBUG_MODE": True,
                "SHOW_INTERNAL_MESSAGES": True,
                "HEALTH_CHECK_INTERVAL_SECONDS": 60,
                "MAX_CONCURRENT_TASKS": 5,
                "ENABLE_CACHE": True,
                "CACHE_TTL": 900,  # 15 minutes
            },
            required_vars=[
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_COHERE_API_KEY",
                "MONGODB_CONNECTION_STRING",
            ],
            optional_vars=[
                "REDIS_URL",
                "PROMETHEUS_ENABLED",
                "GRAFANA_ENABLED",
            ],
            resource_limits={
                "max_memory_mb": 2048,
                "max_cpu_percent": 95.0,
                "max_connections": 10,
            },
            security_settings={
                "cors_enabled": True,
                "rate_limiting": False,
                "api_key_required": False,
                "log_sensitive_data": True,
            },
            performance_settings={
                "connection_pooling": "minimal",
                "cache_strategy": "basic",
                "background_tasks": "light",
            }
        )
        
        # Staging Profile
        profiles[Environment.STAGING] = EnvironmentProfile(
            name="Staging",
            environment=Environment.STAGING,
            description="Staging environment with production-like settings",
            config_overrides={
                "LOG_LEVEL": "INFO",
                "DEBUG_MODE": False,
                "SHOW_INTERNAL_MESSAGES": False,
                "HEALTH_CHECK_INTERVAL_SECONDS": 30,
                "MAX_CONCURRENT_TASKS": 20,
                "CACHE_TTL": 3600,  # 1 hour
            },
            required_vars=[
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_COHERE_API_KEY",
                "MONGODB_CONNECTION_STRING",
                "REDIS_URL",
            ],
            optional_vars=[
                "PROMETHEUS_ENABLED",
                "GRAFANA_ENABLED",
                "ALERTING_WEBHOOK_URL",
            ],
            resource_limits={
                "max_memory_mb": 3072,
                "max_cpu_percent": 85.0,
                "max_connections": 25,
            },
            security_settings={
                "cors_enabled": True,
                "rate_limiting": False,
                "api_key_required": False,
                "log_sensitive_data": False,
            },
            performance_settings={
                "connection_pooling": "moderate",
                "cache_strategy": "advanced",
                "background_tasks": "moderate",
            }
        )
        
        # Production Profile
        profiles[Environment.PRODUCTION] = EnvironmentProfile(
            name="Production",
            environment=Environment.PRODUCTION,
            description="Production environment with optimized settings",
            config_overrides={
                "LOG_LEVEL": "INFO",
                "DEBUG_MODE": False,
                "SHOW_INTERNAL_MESSAGES": False,
                "HEALTH_CHECK_INTERVAL_SECONDS": 15,
                "MAX_CONCURRENT_TASKS": 100,
                "CACHE_TTL": 7200,  # 2 hours
                "PRODUCTION_MODE": True,
            },
            required_vars=[
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_COHERE_API_KEY",
                "MONGODB_CONNECTION_STRING",
                "REDIS_URL",
            ],
            optional_vars=[
                "PROMETHEUS_ENABLED",
                "GRAFANA_ENABLED",
                "ALERTING_WEBHOOK_URL",
                "ALERTING_EMAIL_ENABLED",
                "ALERTING_SLACK_ENABLED",
            ],
            resource_limits={
                "max_memory_mb": 4096,
                "max_cpu_percent": 80.0,
                "max_connections": 50,
            },
            security_settings={
                "cors_enabled": False,
                "rate_limiting": True,
                "api_key_required": True,
                "log_sensitive_data": False,
            },
            performance_settings={
                "connection_pooling": "optimized",
                "cache_strategy": "multi_level",
                "background_tasks": "full",
            }
        )
        
        return profiles
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate current environment configuration."""
        validation_results = {
            "environment": self.current_environment.value,
            "profile": self.current_profile.name if self.current_profile else "Unknown",
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        if not self.current_profile:
            validation_results["valid"] = False
            validation_results["errors"].append(f"No profile found for environment: {self.current_environment.value}")
            return validation_results
        
        # Validate required environment variables
        missing_required = []
        for var in self.current_profile.required_vars:
            if not os.getenv(var):
                missing_required.append(var)
        
        if missing_required:
            validation_results["valid"] = False
            validation_results["errors"].extend([
                f"Missing required environment variable: {var}" for var in missing_required
            ])
        
        # Check optional variables and provide warnings
        missing_optional = []
        for var in self.current_profile.optional_vars:
            if not os.getenv(var):
                missing_optional.append(var)
        
        if missing_optional:
            validation_results["warnings"].extend([
                f"Optional environment variable not set: {var}" for var in missing_optional
            ])
        
        # Validate resource limits
        self._validate_resource_limits(validation_results)
        
        # Validate security settings
        self._validate_security_settings(validation_results)
        
        # Generate performance recommendations
        self._generate_performance_recommendations(validation_results)
        
        return validation_results
    
    def _validate_resource_limits(self, results: Dict[str, Any]):
        """Validate resource limit configurations."""
        limits = self.current_profile.resource_limits
        
        # Check memory settings
        if hasattr(settings, 'PRODUCTION_MAX_MEMORY_MB'):
            if settings.PRODUCTION_MAX_MEMORY_MB > limits["max_memory_mb"]:
                results["warnings"].append(
                    f"Memory limit ({settings.PRODUCTION_MAX_MEMORY_MB}MB) exceeds recommended "
                    f"maximum for {self.current_environment.value} ({limits['max_memory_mb']}MB)"
                )
        
        # Check connection limits
        total_connections = (
            getattr(settings, 'MILVUS_MAX_CONNECTIONS', 0) +
            getattr(settings, 'MONGO_MAX_CONNECTIONS', 0) +
            getattr(settings, 'AZURE_OPENAI_MAX_CONNECTIONS', 0)
        )
        
        if total_connections > limits["max_connections"]:
            results["warnings"].append(
                f"Total max connections ({total_connections}) exceeds recommended "
                f"limit for {self.current_environment.value} ({limits['max_connections']})"
            )
    
    def _validate_security_settings(self, results: Dict[str, Any]):
        """Validate security configuration."""
        security = self.current_profile.security_settings
        
        # Check CORS settings
        if self.current_environment == Environment.PRODUCTION:
            if settings.ENABLE_CORS and not security["cors_enabled"]:
                results["warnings"].append(
                    "CORS is enabled in production - consider restricting origins"
                )
        
        # Check rate limiting
        if self.current_environment == Environment.PRODUCTION:
            if not getattr(settings, 'PRODUCTION_RATE_LIMITING_ENABLED', False):
                results["warnings"].append(
                    "Rate limiting is disabled in production - consider enabling for security"
                )
        
        # Check sensitive data logging
        if self.current_environment == Environment.PRODUCTION:
            if getattr(settings, 'PRODUCTION_LOG_SENSITIVE_DATA', True):
                results["errors"].append(
                    "Sensitive data logging is enabled in production - this is a security risk"
                )
    
    def _generate_performance_recommendations(self, results: Dict[str, Any]):
        """Generate environment-specific performance recommendations."""
        perf_settings = self.current_profile.performance_settings
        
        recommendations = []
        
        if self.current_environment == Environment.DEVELOPMENT:
            recommendations.extend([
                "Enable caching for faster development iterations",
                "Use minimal connection pooling to conserve resources",
                "Consider disabling background tasks for simpler debugging"
            ])
        
        elif self.current_environment == Environment.STAGING:
            recommendations.extend([
                "Enable production-like caching for realistic testing",
                "Use moderate connection pooling to simulate production load",
                "Enable health checks to test monitoring systems"
            ])
        
        elif self.current_environment == Environment.PRODUCTION:
            recommendations.extend([
                "Enable multi-level caching for optimal performance",
                "Use optimized connection pooling with auto-scaling",
                "Enable all background tasks for automated maintenance",
                "Configure monitoring and alerting for production visibility"
            ])
        
        results["recommendations"] = recommendations
    
    def apply_environment_optimizations(self):
        """Apply environment-specific optimizations."""
        if not self.current_profile:
            logger.warning("No profile available - skipping environment optimizations")
            return
        
        logger.info(f"Applying optimizations for {self.current_profile.name} environment")
        
        # Apply configuration overrides
        for key, value in self.current_profile.config_overrides.items():
            if hasattr(settings, key):
                old_value = getattr(settings, key)
                setattr(settings, key, value)
                logger.debug(f"Applied override: {key} = {value} (was: {old_value})")
        
        # Apply performance settings
        self._apply_performance_settings()
        
        # Apply security settings
        self._apply_security_settings()
        
        logger.info("Environment optimizations applied successfully")
    
    def _apply_performance_settings(self):
        """Apply performance-specific settings."""
        perf_strategy = self.current_profile.performance_settings.get("connection_pooling")
        
        if perf_strategy == "minimal":
            # Minimal settings for development
            if hasattr(settings, 'MILVUS_MAX_CONNECTIONS'):
                settings.MILVUS_MAX_CONNECTIONS = min(settings.MILVUS_MAX_CONNECTIONS, 3)
            if hasattr(settings, 'MONGO_MAX_CONNECTIONS'):
                settings.MONGO_MAX_CONNECTIONS = min(settings.MONGO_MAX_CONNECTIONS, 2)
                
        elif perf_strategy == "optimized":
            # Apply production connection pool multipliers
            if hasattr(settings, 'PRODUCTION_MIN_CONNECTIONS_MULTIPLIER'):
                # These multipliers are already applied in the config methods
                pass
    
    def _apply_security_settings(self):
        """Apply security-specific settings."""
        security = self.current_profile.security_settings
        
        # Apply CORS settings
        if "cors_enabled" in security:
            settings.ENABLE_CORS = security["cors_enabled"]
        
        # Apply rate limiting
        if "rate_limiting" in security and hasattr(settings, 'PRODUCTION_RATE_LIMITING_ENABLED'):
            settings.PRODUCTION_RATE_LIMITING_ENABLED = security["rate_limiting"]
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        return {
            "environment": self.current_environment.value,
            "profile": {
                "name": self.current_profile.name,
                "description": self.current_profile.description,
                "performance_strategy": self.current_profile.performance_settings,
                "security_level": "high" if self.current_environment == Environment.PRODUCTION else "medium" if self.current_environment == Environment.STAGING else "low",
            } if self.current_profile else None,
            "configuration": {
                "connection_pools": settings.get_connection_pool_config(),
                "health_checks": settings.get_health_check_config(),
                "performance": settings.get_performance_config(),
                "security": settings.get_security_config(),
                "observability": settings.get_observability_config(),
            },
            "resource_limits": settings.get_resource_limits(),
            "validation": self.validate_environment()
        }
    
    def export_configuration(self, format: str = "json") -> str:
        """Export current configuration in specified format."""
        config_data = self.get_environment_info()
        
        if format.lower() == "json":
            return json.dumps(config_data, indent=2, default=str)
        elif format.lower() == "env":
            # Export as environment variables
            env_vars = []
            for key, value in config_data["configuration"].items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        env_vars.append(f"{key.upper()}_{sub_key.upper()}={sub_value}")
                else:
                    env_vars.append(f"{key.upper()}={value}")
            return "\n".join(env_vars)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def check_readiness(self) -> Dict[str, Any]:
        """Check if the environment is ready for deployment."""
        validation = self.validate_environment()
        
        readiness = {
            "ready": validation["valid"] and len(validation["errors"]) == 0,
            "environment": self.current_environment.value,
            "checks": {
                "configuration": validation["valid"],
                "required_vars": len([e for e in validation["errors"] if "Missing required" in e]) == 0,
                "security": len([e for e in validation["errors"] if "security" in e.lower()]) == 0,
                "resources": len([w for w in validation["warnings"] if "limit" in w.lower()]) == 0,
            },
            "issues": validation["errors"],
            "warnings": validation["warnings"],
            "recommendations": validation["recommendations"]
        }
        
        return readiness


# Global environment manager instance
environment_manager = EnvironmentManager()


# Utility functions for easy access
def get_current_environment() -> Environment:
    """Get the current deployment environment."""
    return environment_manager.current_environment


def validate_current_environment() -> Dict[str, Any]:
    """Validate the current environment configuration."""
    return environment_manager.validate_environment()


def apply_environment_optimizations():
    """Apply optimizations for the current environment."""
    environment_manager.apply_environment_optimizations()


def check_deployment_readiness() -> Dict[str, Any]:
    """Check if the current environment is ready for deployment."""
    return environment_manager.check_readiness()


def get_environment_summary() -> str:
    """Get a summary of the current environment configuration."""
    info = environment_manager.get_environment_info()
    validation = info["validation"]
    
    summary = f"""
Environment: {info['environment'].upper()}
Profile: {info['profile']['name'] if info['profile'] else 'Unknown'}
Status: {'✅ READY' if validation['valid'] else '❌ NOT READY'}

Configuration:
- Connection Pools: {info['configuration']['connection_pools']['milvus_min']}-{info['configuration']['connection_pools']['milvus_max']} (Milvus)
- Performance: {info['configuration']['performance']['max_concurrent_requests']} max requests
- Security: {'Enabled' if info['configuration']['security']['security_enabled'] else 'Disabled'}
- Observability: {'Enabled' if info['configuration']['observability']['enabled'] else 'Disabled'}

Issues: {len(validation['errors'])} errors, {len(validation['warnings'])} warnings
"""
    
    if validation['errors']:
        summary += f"\nErrors:\n" + "\n".join(f"- {error}" for error in validation['errors'])
    
    if validation['warnings']:
        summary += f"\nWarnings:\n" + "\n".join(f"- {warning}" for warning in validation['warnings'])
    
    return summary.strip()
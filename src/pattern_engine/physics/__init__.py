"""
Physics Module for Pattern Engine
=================================

Physics-based validation and gray zone detection.
"""

from .rules import (
    # Enums and dataclasses
    Domain,
    PhysicsRule,
    ValidationResult,
    
    # Main validator
    PhysicsValidator,
    
    # Pre-configured rules
    FLOOD_RULES,
    MANUFACTURING_RULES,
    ENERGY_RULES,
    
    # Convenience constructors
    create_flood_validator,
    create_manufacturing_validator,
    create_energy_validator,
)

__all__ = [
    "Domain",
    "PhysicsRule",
    "ValidationResult",
    "PhysicsValidator",
    "FLOOD_RULES",
    "MANUFACTURING_RULES",
    "ENERGY_RULES",
    "create_flood_validator",
    "create_manufacturing_validator",
    "create_energy_validator",
]

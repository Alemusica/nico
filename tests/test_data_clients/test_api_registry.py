"""
🔌 Test API Registry
====================

Tests for the data client API registry pattern.
The registry provides:
- Centralized client discovery
- Lazy client instantiation
- Credential management
- Rate limiting coordination
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Callable
from enum import Enum
from unittest.mock import MagicMock, patch, AsyncMock


# ============================================================================
# API Registry Implementation (to be moved to src/data/ later)
# ============================================================================

class DataSource(Enum):
    """Available data sources."""
    CMEMS = "cmems"
    ERA5 = "era5"
    SENTINEL = "sentinel"
    GRACE = "grace"
    ARGO = "argo"
    GPM = "gpm"
    CYGNSS = "cygnss"
    TIDE_GAUGE = "tide_gauge"
    AIRCRAFT = "aircraft"
    CLIMATE_INDICES = "climate_indices"
    SEMANTIC_SCHOLAR = "semantic_scholar"


@dataclass
class ClientConfig:
    """Configuration for a data client."""
    name: str
    source: DataSource
    description: str
    requires_auth: bool = False
    auth_env_vars: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60
    default_timeout: int = 30
    retry_attempts: int = 3
    cache_ttl_hours: int = 24


@dataclass
class ClientStatus:
    """Status of a registered client."""
    name: str
    available: bool
    authenticated: bool = False
    last_request: Optional[str] = None
    error_message: Optional[str] = None


class APIRegistry:
    """
    Central registry for data acquisition clients.
    
    Provides:
    - Client registration and discovery
    - Lazy instantiation
    - Credential validation
    - Health checking
    
    Usage:
        registry = APIRegistry()
        
        # Register clients
        registry.register("cmems", CMEMSClient, CMEMSConfig)
        
        # Get client
        cmems = registry.get_client("cmems")
        
        # Check status
        status = registry.status()
    """
    
    def __init__(self):
        self._clients: Dict[str, Type] = {}
        self._configs: Dict[str, ClientConfig] = {}
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register(
        self,
        name: str,
        client_class: Type = None,
        config: ClientConfig = None,
        factory: Callable = None,
    ) -> None:
        """
        Register a data client.
        
        Args:
            name: Unique client name
            client_class: Client class (instantiated lazily)
            config: Client configuration
            factory: Optional factory function for custom instantiation
        """
        if name in self._clients:
            raise ValueError(f"Client '{name}' already registered")
        
        self._clients[name] = client_class
        self._configs[name] = config or ClientConfig(
            name=name,
            source=DataSource.CMEMS,  # Default
            description=f"{name} data client",
        )
        
        if factory:
            self._factories[name] = factory
    
    def unregister(self, name: str) -> bool:
        """Remove a client from registry."""
        if name not in self._clients:
            return False
        
        del self._clients[name]
        del self._configs[name]
        
        if name in self._instances:
            del self._instances[name]
        
        if name in self._factories:
            del self._factories[name]
        
        return True
    
    def get_client(self, name: str, **kwargs) -> Any:
        """
        Get or create client instance.
        
        Args:
            name: Client name
            **kwargs: Additional arguments for client initialization
            
        Returns:
            Client instance
        """
        if name not in self._clients:
            raise KeyError(f"Unknown client: {name}")
        
        # Return cached instance if exists
        if name in self._instances and not kwargs:
            return self._instances[name]
        
        # Create new instance
        if name in self._factories:
            instance = self._factories[name](**kwargs)
        else:
            client_class = self._clients[name]
            instance = client_class(**kwargs) if client_class else None
        
        # Cache if no custom kwargs
        if not kwargs:
            self._instances[name] = instance
        
        return instance
    
    def list_clients(self) -> List[str]:
        """List all registered client names."""
        return list(self._clients.keys())
    
    def get_config(self, name: str) -> Optional[ClientConfig]:
        """Get client configuration."""
        return self._configs.get(name)
    
    def status(self) -> Dict[str, ClientStatus]:
        """Get status of all registered clients."""
        statuses = {}
        
        for name in self._clients:
            config = self._configs.get(name)
            
            # Check if client is available (class exists)
            available = self._clients[name] is not None
            
            # Check authentication if required
            authenticated = True
            if config and config.requires_auth:
                import os
                authenticated = all(
                    os.getenv(var) for var in config.auth_env_vars
                )
            
            statuses[name] = ClientStatus(
                name=name,
                available=available,
                authenticated=authenticated,
            )
        
        return statuses
    
    def health_check(self) -> Dict[str, bool]:
        """Quick health check of all clients."""
        return {
            name: status.available and status.authenticated
            for name, status in self.status().items()
        }
    
    def __contains__(self, name: str) -> bool:
        return name in self._clients
    
    def __len__(self) -> int:
        return len(self._clients)


# Global registry instance
_global_registry: Optional[APIRegistry] = None


def get_registry() -> APIRegistry:
    """Get the global API registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = APIRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None


# ============================================================================
# Tests
# ============================================================================

class TestDataSource:
    """Test DataSource enum."""
    
    def test_all_sources_defined(self):
        """All expected data sources should be defined."""
        expected = [
            "cmems", "era5", "sentinel", "grace", "argo",
            "gpm", "cygnss", "tide_gauge", "aircraft",
            "climate_indices", "semantic_scholar"
        ]
        
        actual = [ds.value for ds in DataSource]
        
        for source in expected:
            assert source in actual, f"Missing source: {source}"
    
    def test_source_values_lowercase(self):
        """Source values should be lowercase."""
        for source in DataSource:
            assert source.value == source.value.lower()


class TestClientConfig:
    """Test ClientConfig dataclass."""
    
    def test_create_minimal_config(self):
        """Should create config with required fields."""
        config = ClientConfig(
            name="test",
            source=DataSource.CMEMS,
            description="Test client",
        )
        
        assert config.name == "test"
        assert config.source == DataSource.CMEMS
        assert config.requires_auth is False
    
    def test_create_authenticated_config(self):
        """Should create config with auth requirements."""
        config = ClientConfig(
            name="era5",
            source=DataSource.ERA5,
            description="ERA5 client",
            requires_auth=True,
            auth_env_vars=["CDS_API_KEY"],
            rate_limit_per_minute=20,
        )
        
        assert config.requires_auth is True
        assert "CDS_API_KEY" in config.auth_env_vars
        assert config.rate_limit_per_minute == 20
    
    def test_default_values(self):
        """Should have sensible defaults."""
        config = ClientConfig(
            name="test",
            source=DataSource.GPM,
            description="Test",
        )
        
        assert config.rate_limit_per_minute == 60
        assert config.default_timeout == 30
        assert config.retry_attempts == 3
        assert config.cache_ttl_hours == 24


class TestClientStatus:
    """Test ClientStatus dataclass."""
    
    def test_create_status(self):
        """Should create status."""
        status = ClientStatus(
            name="test",
            available=True,
            authenticated=True,
        )
        
        assert status.available is True
        assert status.authenticated is True
    
    def test_unavailable_status(self):
        """Should handle unavailable client."""
        status = ClientStatus(
            name="missing",
            available=False,
            error_message="Module not installed",
        )
        
        assert status.available is False
        assert "not installed" in status.error_message


class TestAPIRegistry:
    """Test APIRegistry class."""
    
    @pytest.fixture(autouse=True)
    def reset_registry_fixture(self):
        """Reset global registry before each test."""
        reset_registry()
        yield
        reset_registry()
    
    @pytest.fixture
    def registry(self):
        """Fresh registry for testing."""
        return APIRegistry()
    
    def test_register_client(self, registry):
        """Should register a client."""
        class MockClient:
            pass
        
        config = ClientConfig(
            name="mock",
            source=DataSource.CMEMS,
            description="Mock client",
        )
        
        registry.register("mock", MockClient, config)
        
        assert "mock" in registry
        assert len(registry) == 1
    
    def test_register_duplicate_raises(self, registry):
        """Should raise on duplicate registration."""
        registry.register("test", None)
        
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", None)
    
    def test_list_clients(self, registry):
        """Should list registered clients."""
        registry.register("client1", None)
        registry.register("client2", None)
        
        clients = registry.list_clients()
        
        assert "client1" in clients
        assert "client2" in clients
        assert len(clients) == 2
    
    def test_get_client_creates_instance(self, registry):
        """Should create client instance."""
        class MockClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        
        registry.register("mock", MockClient)
        
        client = registry.get_client("mock")
        
        assert isinstance(client, MockClient)
    
    def test_get_client_caches_instance(self, registry):
        """Should cache client instance."""
        class MockClient:
            pass
        
        registry.register("mock", MockClient)
        
        client1 = registry.get_client("mock")
        client2 = registry.get_client("mock")
        
        assert client1 is client2
    
    def test_get_client_with_kwargs_no_cache(self, registry):
        """Should not cache when kwargs provided."""
        class MockClient:
            def __init__(self, **kwargs):
                self.value = kwargs.get("value", 0)
        
        registry.register("mock", MockClient)
        
        client1 = registry.get_client("mock", value=1)
        client2 = registry.get_client("mock", value=2)
        
        assert client1.value == 1
        assert client2.value == 2
        assert client1 is not client2
    
    def test_get_unknown_client_raises(self, registry):
        """Should raise for unknown client."""
        with pytest.raises(KeyError, match="Unknown client"):
            registry.get_client("nonexistent")
    
    def test_unregister_client(self, registry):
        """Should unregister client."""
        registry.register("temp", None)
        
        assert "temp" in registry
        
        result = registry.unregister("temp")
        
        assert result is True
        assert "temp" not in registry
    
    def test_unregister_unknown_returns_false(self, registry):
        """Should return False for unknown client."""
        result = registry.unregister("nonexistent")
        
        assert result is False
    
    def test_get_config(self, registry):
        """Should return client config."""
        config = ClientConfig(
            name="test",
            source=DataSource.ARGO,
            description="Test config",
            rate_limit_per_minute=10,
        )
        
        registry.register("test", None, config)
        
        retrieved = registry.get_config("test")
        
        assert retrieved.rate_limit_per_minute == 10
        assert retrieved.source == DataSource.ARGO
    
    def test_status_all_clients(self, registry):
        """Should return status of all clients."""
        class Client1:
            pass
        
        registry.register("client1", Client1)
        registry.register("client2", None)  # No class
        
        statuses = registry.status()
        
        assert "client1" in statuses
        assert "client2" in statuses
        assert statuses["client1"].available is True
        assert statuses["client2"].available is False
    
    def test_status_with_auth_check(self, registry):
        """Should check authentication status."""
        class AuthClient:
            pass
        
        config = ClientConfig(
            name="auth_client",
            source=DataSource.ERA5,
            description="Auth required",
            requires_auth=True,
            auth_env_vars=["NONEXISTENT_VAR_12345"],
        )
        
        registry.register("auth_client", AuthClient, config)
        
        statuses = registry.status()
        
        # Should be unauthenticated (env var doesn't exist)
        assert statuses["auth_client"].authenticated is False
    
    def test_health_check(self, registry):
        """Should return health check results."""
        class HealthyClient:
            pass
        
        registry.register("healthy", HealthyClient)
        registry.register("unhealthy", None)
        
        health = registry.health_check()
        
        assert health["healthy"] is True
        assert health["unhealthy"] is False
    
    def test_factory_function(self, registry):
        """Should use factory for custom instantiation."""
        class CustomClient:
            def __init__(self, special_arg):
                self.special = special_arg
        
        def factory(**kwargs):
            return CustomClient(special_arg="factory_created")
        
        registry.register("custom", CustomClient, factory=factory)
        
        client = registry.get_client("custom")
        
        assert client.special == "factory_created"
    
    def test_contains_operator(self, registry):
        """Should support 'in' operator."""
        registry.register("test", None)
        
        assert "test" in registry
        assert "missing" not in registry
    
    def test_len_operator(self, registry):
        """Should support len()."""
        assert len(registry) == 0
        
        registry.register("a", None)
        registry.register("b", None)
        
        assert len(registry) == 2


class TestGlobalRegistry:
    """Test global registry functions."""
    
    @pytest.fixture(autouse=True)
    def reset(self):
        reset_registry()
        yield
        reset_registry()
    
    def test_get_registry_creates_singleton(self):
        """Should create singleton registry."""
        reg1 = get_registry()
        reg2 = get_registry()
        
        assert reg1 is reg2
    
    def test_reset_registry(self):
        """Should reset global registry."""
        reg1 = get_registry()
        reg1.register("test", None)
        
        reset_registry()
        
        reg2 = get_registry()
        
        assert "test" not in reg2
        assert reg1 is not reg2


class TestRegistryIntegration:
    """Integration tests for registry pattern."""
    
    @pytest.fixture(autouse=True)
    def reset(self):
        reset_registry()
        yield
        reset_registry()
    
    def test_register_multiple_sources(self):
        """Should handle multiple data sources."""
        registry = get_registry()
        
        # Register various client types
        sources = [
            ("cmems", DataSource.CMEMS),
            ("era5", DataSource.ERA5),
            ("sentinel", DataSource.SENTINEL),
            ("argo", DataSource.ARGO),
        ]
        
        for name, source in sources:
            config = ClientConfig(
                name=name,
                source=source,
                description=f"{name} client",
            )
            registry.register(name, MagicMock, config)
        
        assert len(registry) == 4
        
        # All should be healthy (MagicMock is truthy)
        health = registry.health_check()
        for name, _ in sources:
            assert health[name] is True
    
    def test_lazy_instantiation_performance(self):
        """Clients should not be instantiated until requested."""
        registry = APIRegistry()
        
        instantiation_count = 0
        
        class ExpensiveClient:
            def __init__(self):
                nonlocal instantiation_count
                instantiation_count += 1
        
        # Register 10 clients
        for i in range(10):
            registry.register(f"client_{i}", ExpensiveClient)
        
        assert instantiation_count == 0  # No instantiation yet
        
        # Request one client
        registry.get_client("client_0")
        
        assert instantiation_count == 1  # Only one instantiated


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

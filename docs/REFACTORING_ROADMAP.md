# 🚀 REFACTORING ROADMAP

> **Piano di refactoring completo per portare la codebase da 6.5/10 a 8.5/10**
> 
> **Timeline**: 7 settimane | **Effort**: 130-180 ore | **Team**: 1-2 devs

---

## 🎯 EXECUTIVE SUMMARY

### Valutazione Attuale: **6.5/10**

**✅ Punti di forza**:
- Logging strutturato production-ready (investigation_logger.py)
- Validation pipeline con severity levels (CRITICAL/WARNING/INFO)
- Architettura backend/frontend separata
- Documentazione presente (CHANGELOG, README, docstrings)

**🔴 Problemi critici da risolvere**:
1. **God Object**: `api/main.py` con 2530 linee (antipattern)
2. **Testing**: Coverage < 10% (rischio altissimo)
3. **Responsabilità**: `investigation_agent.py` 1344 linee
4. **Config**: Hardcoded, no environment variables
5. **DI**: Dependency injection assente
6. **Async**: Inconsistenze sync/async
7. **Errors**: Error handling non standardizzato

---

## 📊 METRICHE OBIETTIVO

| Metrica | Before | After | Improvement |
|---------|--------|-------|-------------|
| **Lines in main.py** | 2530 | < 150 | -94% |
| **Test coverage** | 10% | > 80% | +700% |
| **Agent complexity** | 1344 lines | < 400 | -70% |
| **Hard deps** | Many | 0 | ✅ Mockable |
| **Investigation time** | Unknown | < 10s | Measured |
| **Valutazione finale** | 6.5/10 | **8.5/10** | +2.0 |

---

## 🗓️ SPRINT PLANNING

### 📆 SPRINT 1: Fondamenta (1-2 settimane, 40-60h)

#### Task 1.1: Split api/main.py → routers modulari 🔴
**Priority**: CRITICAL | **Effort**: 12-16h

**Obiettivo**: Ridurre `api/main.py` da 2530 linee a < 150 linee

**Nuova struttura**:
```
api/
  ├── main.py                      # < 150 linee (setup + middleware)
  ├── routers/
  │   ├── __init__.py
  │   ├── investigation_router.py  # WebSocket, briefing, investigate
  │   ├── knowledge_router.py      # Papers, events, patterns, stats
  │   ├── data_router.py           # Upload, files, cache, preview
  │   └── pipeline_router.py       # Pipeline stages (refine, correlate)
```

**Checklist**:
- [ ] Creare `api/routers/` directory
- [ ] Estrarre investigation endpoints → `investigation_router.py`
- [ ] Estrarre knowledge endpoints → `knowledge_router.py`
- [ ] Estrarre data endpoints → `data_router.py`
- [ ] Estrarre pipeline endpoints → `pipeline_router.py`
- [ ] Aggiornare imports in `main.py`
- [ ] Verificare con `wc -l api/main.py` < 150
- [ ] Run `radon cc api/` per verificare complexity < 10

**Comando**:
```bash
git checkout -b refactor/router-split
mkdir -p api/routers
touch api/routers/{__init__,investigation,knowledge,data,pipeline}_router.py
```

---

#### Task 1.2: Setup testing framework 🔴
**Priority**: CRITICAL | **Effort**: 8-12h

**Obiettivo**: Creare infrastruttura test production-ready

**Struttura**:
```
tests/
  ├── __init__.py
  ├── conftest.py                  # Fixtures condivise
  ├── unit/                        # 75% dei test
  │   ├── test_investigation_logger.py
  │   ├── test_investigation_validators.py
  │   └── test_resolver.py
  ├── integration/                 # 20% dei test
  │   ├── test_investigation_agent.py
  │   └── test_knowledge_service.py
  └── e2e/                         # 5% dei test
      └── test_investigation_endpoints.py
```

**Setup**:
```bash
# Install dependencies
pip install pytest pytest-asyncio pytest-cov httpx hypothesis pytest-mock

# Create pytest.ini
cat > pytest.ini << 'EOF'
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
addopts = 
    --verbose
    --cov=src
    --cov=api
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=70
EOF
```

**Checklist**:
- [ ] Install test dependencies
- [ ] Create `pytest.ini`
- [ ] Create `tests/conftest.py` con fixtures
- [ ] Setup directory structure
- [ ] Verificare con `pytest --collect-only`

---

#### Task 1.3: Primi 20 unit tests critici 🔴
**Priority**: CRITICAL | **Effort**: 12-16h

**Obiettivo**: Coverage > 70% per componenti core

**Test prioritari**:

1. **test_investigation_logger.py** (6 test)
   - `test_start_step()`
   - `test_complete_step()`
   - `test_fail_step()`
   - `test_get_summary()`
   - `test_log_validation()`
   - `test_log_health_check()`

2. **test_investigation_validators.py** (8 test)
   - `test_paper_validator_critical()`
   - `test_paper_validator_warning()`
   - `test_paper_sanitizer()`
   - `test_duplicate_detector_by_doi()`
   - `test_duplicate_detector_by_title()`
   - `test_validate_papers_batch()`
   - `test_validation_levels()`
   - `test_sanitizer_edge_cases()`

3. **test_resolver.py** (4 test)
   - `test_variable_resolver_slcci()`
   - `test_variable_resolver_cmems()`
   - `test_get_coordinates()`
   - `test_auto_detect_format()`

4. **test_investigation_agent.py** (2 test - integration)
   - `test_investigate_flow_mock()`
   - `test_paper_collection_validation()`

**Run**:
```bash
pytest tests/unit/ -v --cov
pytest tests/integration/ -v
```

---

#### Task 1.4: Pydantic Settings per config 🟡
**Priority**: HIGH | **Effort**: 6-8h

**Obiettivo**: Zero hardcoded config, tutto via environment variables

**File: `api/config.py`**:
```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # App
    app_name: str = "NICO Causal Discovery"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: List[str] = ["http://localhost:5173"]
    
    # SurrealDB
    surreal_url: str = "ws://localhost:8001/rpc"
    surreal_namespace: str = "oceanography"
    surreal_database: str = "knowledge"
    surreal_user: str = "root"
    surreal_pass: str  # Required from .env
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str  # Required from .env
    
    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "llama2"
    llm_base_url: str = "http://localhost:11434"
    
    # Data Manager
    data_cache_dir: str = "./data/cache"
    data_max_cache_size_gb: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = "NICO_"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

**File: `.env.example`**:
```bash
# NICO Configuration
# Copy to .env and fill in secrets

# SurrealDB (Required)
NICO_SURREAL_PASS=your_surreal_password

# Neo4j (Required)
NICO_NEO4J_PASSWORD=your_neo4j_password

# Optional overrides
# NICO_DEBUG=true
# NICO_CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

**Checklist**:
- [ ] Create `api/config.py`
- [ ] Create `.env.example`
- [ ] Add `.env` to `.gitignore`
- [ ] Replace hardcoded configs in codebase
- [ ] Test with `python -c "from api.config import settings; print(settings.surreal_url)"`

---

### 📆 SPRINT 2: Architettura (2-3 settimane, 60-80h)

#### Task 2.1: Service layer per Investigation Agent 🟡
**Priority**: HIGH | **Effort**: 16-20h

**Obiettivo**: Ridurre `investigation_agent.py` da 1344 → < 400 linee

**Nuova architettura**:
```
src/services/
  ├── __init__.py
  ├── data_collection_service.py   # Satellite, ERA5, climate indices
  ├── paper_search_service.py      # Semantic Scholar integration
  ├── validation_service.py        # Paper validation logic
  └── storage_service.py           # Knowledge base storage

src/agent/
  └── investigation_agent.py       # Orchestrator leggero
```

**InvestigationAgent refactored**:
```python
class InvestigationAgent:
    def __init__(
        self,
        data_service: DataCollectionService,
        paper_service: PaperSearchService,
        validation_service: ValidationService,
        storage_service: StorageService,
        logger: InvestigationLogger = None,
    ):
        self.data = data_service
        self.papers = paper_service
        self.validation = validation_service
        self.storage = storage_service
        self._logger = logger
    
    async def investigate(self, query: str):
        """Main investigation flow - now just orchestration"""
        # Parse query
        parsed = await self._parse_query(query)
        
        # Collect data (delegated to service)
        data = await self.data.collect_all(
            location=parsed.location,
            time_range=parsed.time_range,
        )
        
        # Search papers (delegated)
        papers = await self.papers.search(
            query=query,
            location=parsed.location,
        )
        
        # Validate (delegated)
        validated = await self.validation.validate_papers(papers)
        
        # Store (delegated)
        await self.storage.store_papers(validated)
        
        return InvestigationResult(...)
```

**Checklist**:
- [ ] Create `DataCollectionService` con metodi satellite/era5/indices
- [ ] Create `PaperSearchService` con semantic scholar client
- [ ] Create `ValidationService` usando investigation_validators.py
- [ ] Create `StorageService` wrappando knowledge_service
- [ ] Refactor `InvestigationAgent` as orchestrator
- [ ] Verify `wc -l src/agent/investigation_agent.py` < 400

---

#### Task 2.2: Dependency injection container 🟡
**Priority**: HIGH | **Effort**: 12-16h

**Obiettivo**: Zero hard dependencies, tutto mockable per testing

**File: `api/container.py`**:
```python
from dependency_injector import containers, providers
from api.config import settings
from api.services.surrealdb_knowledge import SurrealDBKnowledgeService
from api.services.neo4j_knowledge import Neo4jKnowledgeService
from src.services.data_collection_service import DataCollectionService
from src.services.paper_search_service import PaperSearchService
from src.services.validation_service import ValidationService
from src.services.storage_service import StorageService
from src.agent.investigation_agent import InvestigationAgent

class Container(containers.DeclarativeContainer):
    """Dependency injection container"""
    
    # Config
    config = providers.Configuration()
    config.from_pydantic(settings)
    
    # Knowledge services
    surreal_knowledge = providers.Singleton(
        SurrealDBKnowledgeService,
        url=config.surreal_url,
        namespace=config.surreal_namespace,
        database=config.surreal_database,
        username=config.surreal_user,
        password=config.surreal_pass,
    )
    
    neo4j_knowledge = providers.Singleton(
        Neo4jKnowledgeService,
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
    )
    
    # Services
    data_collection = providers.Factory(
        DataCollectionService,
        cache_dir=config.data_cache_dir,
    )
    
    paper_search = providers.Factory(
        PaperSearchService,
    )
    
    validation = providers.Factory(
        ValidationService,
    )
    
    storage = providers.Factory(
        StorageService,
        knowledge_service=surreal_knowledge,
    )
    
    # Investigation agent
    investigation_agent = providers.Factory(
        InvestigationAgent,
        data_service=data_collection,
        paper_service=paper_search,
        validation_service=validation,
        storage_service=storage,
    )
```

**Usage in FastAPI**:
```python
# api/main.py
from api.container import Container

app = FastAPI()
app.container = Container()

@app.websocket("/ws/investigate")
async def investigate(websocket: WebSocket):
    agent = app.container.investigation_agent()
    # Now fully injected and mockable!
```

**Testing**:
```python
# tests/conftest.py
@pytest.fixture
def mock_container():
    container = Container()
    container.surreal_knowledge.override(MockKnowledgeService())
    container.paper_search.override(MockPaperService())
    return container
```

**Checklist**:
- [ ] Install `pip install dependency-injector`
- [ ] Create `api/container.py`
- [ ] Wire up all services
- [ ] Update `api/main.py` to use container
- [ ] Create test fixtures with overrides
- [ ] Verify all endpoints work

---

#### Task 2.3: Custom exceptions + error handling 🟡
**Priority**: MEDIUM | **Effort**: 8-12h

**File: `api/exceptions.py`**:
```python
from typing import Optional, Dict, Any

class NicoException(Exception):
    """Base exception for all NICO errors"""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status_code

class DataCollectionError(NicoException):
    """Data collection from external sources failed"""
    def __init__(self, message: str, source: str, **kwargs):
        super().__init__(message, details={"source": source}, **kwargs)

class ValidationError(NicoException):
    """Data validation failed"""
    def __init__(self, message: str, validator: str, **kwargs):
        super().__init__(
            message,
            details={"validator": validator},
            status_code=422,
            **kwargs
        )

class StorageError(NicoException):
    """Storage operation failed"""
    def __init__(self, message: str, backend: str, **kwargs):
        super().__init__(message, details={"backend": backend}, **kwargs)

class ExternalAPIError(NicoException):
    """External API call failed"""
    def __init__(self, message: str, api: str, **kwargs):
        super().__init__(message, details={"api": api}, **kwargs)
```

**Global handler in `api/main.py`**:
```python
from api.exceptions import NicoException
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(NicoException)
async def nico_exception_handler(request: Request, exc: NicoException):
    """Handle all NICO exceptions uniformly"""
    logger.error(
        f"{exc.__class__.__name__}: {exc.message}",
        extra={
            "details": exc.details,
            "path": request.url.path,
        }
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
        }
    )
```

**Checklist**:
- [ ] Create exception hierarchy
- [ ] Add global handler
- [ ] Replace all `HTTPException(500)` with custom exceptions
- [ ] Add error logging
- [ ] Test error responses

---

#### Task 2.4: Async/await consistency 🟡
**Priority**: MEDIUM | **Effort**: 12-16h

**Obiettivo**: Pipeline 100% async, no sync wrappers

**Problemi da fixare**:
```python
# ❌ BEFORE: Mixed sync/async (doesn't work!)
def _collect_papers(self):
    papers = self._search_papers()  # sync
    await service.store(papers)     # async - ERROR!

# ✅ AFTER: Full async
async def _collect_papers(self):
    papers = await self._search_papers()
    validated = await self._validate_papers(papers)
    await service.store(validated)
```

**Checklist**:
- [ ] Audit codebase: `grep -r "def.*await" src/ api/`
- [ ] Convert all methods con `await` to `async def`
- [ ] Rimuovere `asyncio.run()` calls
- [ ] Verify async context managers per DB
- [ ] Use `AsyncIterator` per streaming
- [ ] Test con `pytest-asyncio`

---

### 📆 SPRINT 3: Quality (1-2 settimane, 30-40h)

#### Task 3.1: Test coverage >80% 🟢
**Priority**: MEDIUM | **Effort**: 12-16h

**Obiettivo**: Production-ready test suite

**Coverage targets**:
- `src/core/`: > 85%
- `src/services/`: > 80%
- `api/routers/`: > 75%
- `src/agent/`: > 70%

**Tipi di test**:
1. **Unit tests** (parametrized)
2. **Integration tests** (con mock DB)
3. **Property-based tests** (hypothesis)
4. **Edge case tests** (network failures, timeouts)

**Run**:
```bash
pytest --cov --cov-report=html
open htmlcov/index.html
```

---

#### Task 3.2: OpenAPI documentation 🟢
**Priority**: MEDIUM | **Effort**: 6-8h

**Obiettivo**: Swagger UI perfetto su `/docs`

**Example endpoint**:
```python
@router.post(
    "/investigate/briefing",
    response_model=InvestigationBriefing,
    tags=["investigation"],
    summary="Create investigation briefing",
    description="""
    Generate briefing with data source estimates before investigation.
    Returns estimated data size, time range, and download time.
    """,
    responses={
        200: {"description": "Briefing created successfully"},
        422: {"description": "Invalid query format"},
        500: {"description": "Internal server error"},
    },
)
async def create_briefing(
    request: BriefingRequest = Body(
        ...,
        example={
            "query": "Lago Maggiore floods October 2000",
            "temporal_resolution": "daily",
            "spatial_resolution": "0.25",
        }
    ),
):
    """Create investigation briefing with data estimates"""
```

---

#### Task 3.3: Performance profiling 🟢
**Priority**: LOW | **Effort**: 8-12h

**Tools**:
```bash
# CPU profiling
pip install py-spy
py-spy record -o profile.svg -- uvicorn api.main:app

# Memory profiling
pip install memory-profiler
mprof run uvicorn api.main:app
mprof plot

# Load testing
pip install locust
locust -f tests/load/investigation_load.py
```

---

#### Task 3.4: Security audit 🟢
**Priority**: MEDIUM | **Effort**: 4-6h

**Tools**:
```bash
# Static analysis
pip install bandit safety
bandit -r src/ api/
safety check

# Dependency vulnerabilities
pip-audit
```

**Checklist**:
- [x] Input validation (Pydantic)
- [ ] Rate limiting (slowapi)
- [x] SQL injection (using ORM)
- [ ] Secrets in .env
- [ ] CORS restrictive

---

## 🎁 BONUS TASKS

### Pre-commit hooks 🔵
```bash
pip install pre-commit
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
EOF

pre-commit install
```

---

### Docker Compose 🔵
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NICO_SURREAL_URL=ws://surrealdb:8000/rpc
      - NICO_NEO4J_URI=bolt://neo4j:7687
    depends_on:
      - surrealdb
      - neo4j
    volumes:
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    depends_on:
      - backend

  surrealdb:
    image: surrealdb/surrealdb:latest
    ports:
      - "8001:8000"
    command: start --user root --pass root memory

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
```

---

## 📈 DEFINIZIONE DI DONE

### Sprint 1 ✅
- [ ] `api/main.py` < 150 linee
- [ ] 4 router files creati
- [ ] pytest setup completo
- [ ] 20+ unit tests passing
- [ ] Coverage > 70% core components
- [ ] Pydantic settings implementato
- [ ] `.env.example` committato

### Sprint 2 ✅
- [ ] 4 service files creati
- [ ] `investigation_agent.py` < 400 linee
- [ ] DI container funzionante
- [ ] Custom exceptions in uso
- [ ] Zero HTTPException(500)
- [ ] Pipeline 100% async
- [ ] All tests passing

### Sprint 3 ✅
- [ ] Coverage > 80%
- [ ] `/docs` endpoint perfetto
- [ ] Performance baseline < 10s
- [ ] Zero critical security issues
- [ ] CI/CD pipeline setup (bonus)
- [ ] Docker compose working (bonus)

---

## 🛠️ TOOLS INSTALLATION

```bash
# Create requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.12.0
httpx==0.25.2
hypothesis==6.92.1

# Code quality
black==23.12.0
isort==5.13.2
mypy==1.7.1
pylint==3.0.3
radon==6.0.1
bandit==1.7.5
safety==2.3.5

# DI & Config
dependency-injector==4.41.0
pydantic-settings==2.1.0

# Performance
py-spy==0.3.14
memory-profiler==0.61.0
locust==2.20.0

# DevOps
pre-commit==3.6.0
EOF

pip install -r requirements-dev.txt
```

---

## 📅 TIMELINE

```
Week 1-2: Sprint 1 (Fondamenta)
├─ Day 1-2: Router split
├─ Day 3-4: Test setup + primi test
└─ Day 5: Config management

Week 3-5: Sprint 2 (Architettura)
├─ Day 6-9: Service layer
├─ Day 10-12: Dependency injection
├─ Day 13-14: Exceptions
└─ Day 15-16: Async consistency

Week 6-7: Sprint 3 (Quality)
├─ Day 17-19: Test coverage
├─ Day 20: OpenAPI docs
├─ Day 21-22: Performance
└─ Day 23: Security audit

TOTAL: 7 weeks, 130-180 hours
```

---

## 🎯 COME INIZIARE OGGI

```bash
# 1. Create refactor branch
git checkout -b refactor/router-split

# 2. Create new directories
mkdir -p api/routers tests/{unit,integration,e2e}

# 3. Create router files
touch api/routers/{__init__,investigation,knowledge,data,pipeline}_router.py

# 4. Create test files
touch tests/unit/test_{logger,validators,resolver}.py
touch tests/integration/test_investigation_agent.py
touch tests/e2e/test_api_endpoints.py
touch tests/conftest.py

# 5. Create config
touch api/config.py .env.example

# 6. Start refactoring!
code api/routers/investigation_router.py
```

---

## ✅ SUCCESS CRITERIA

**Prima (Now)**:
- api/main.py: 2530 linee
- Test coverage: 10%
- Valutazione: 6.5/10

**Dopo Sprint 1**:
- api/main.py: < 150 linee ✅
- Test coverage: > 70% ✅
- Valutazione: 7.5/10

**Dopo Sprint 2**:
- investigation_agent.py: < 400 linee ✅
- Zero hard dependencies ✅
- Valutazione: 8.0/10

**Dopo Sprint 3**:
- Test coverage: > 80% ✅
- Investigation time: < 10s ✅
- **Valutazione finale: 8.5/10** 🎉

---

**Ready to ship production-grade code?** 🚀

Let's start with Sprint 1, Task 1.1!

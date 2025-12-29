---
description: Aggiungi nuovo endpoint API FastAPI
---

# Nuovo Endpoint API

Crea un nuovo endpoint FastAPI seguendo la struttura del progetto NICO.

## Requisiti

1. **Router**: Crea in `api/routers/` se nuovo dominio, altrimenti aggiungi a router esistente
2. **Service**: Logica business in `api/services/`
3. **Models**: Pydantic models in `api/models.py`
4. **Tests**: Test in `tests/test_api/`

## Template Router

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/DOMAIN", tags=["DOMAIN"])

class RequestModel(BaseModel):
    field: str

class ResponseModel(BaseModel):
    id: str
    data: dict

@router.get("/", response_model=list[ResponseModel])
async def list_items():
    """List all items."""
    pass

@router.get("/{item_id}", response_model=ResponseModel)
async def get_item(item_id: str):
    """Get single item."""
    pass

@router.post("/", response_model=ResponseModel)
async def create_item(request: RequestModel):
    """Create new item."""
    pass
```

## Checklist

- [ ] Type hints su tutti i parametri
- [ ] Docstring su ogni endpoint
- [ ] Error handling con HTTPException
- [ ] Registra router in `api/main.py`

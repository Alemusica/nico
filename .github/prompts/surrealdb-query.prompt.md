---
description: Query SurrealDB per knowledge graph
---

# SurrealDB Query

Crea query per SurrealDB nel progetto NICO.

## Connessione

```python
from surrealdb import Surreal

async def get_db():
    db = Surreal("ws://localhost:8001/rpc")
    await db.connect()
    await db.use("causal", "knowledge")
    return db
```

## Schema Esistente

```
events:event - { id, name, date, location, type, severity }
papers:paper - { id, title, authors, year, doi, abstract }
patterns:pattern - { source, target, lag, strength, mechanism }
```

## Query Comuni

### Select con filtro
```python
result = await db.query(
    "SELECT * FROM events WHERE type = $type AND date > $date",
    {"type": "flood", "date": "2000-01-01"}
)
```

### Relazioni
```python
result = await db.query("""
    SELECT *, ->causes->events AS effects 
    FROM events 
    WHERE id = $id
""", {"id": "events:flood_2000"})
```

### Aggregazioni
```python
result = await db.query("""
    SELECT type, count() AS total 
    FROM events 
    GROUP BY type
""")
```

### Insert
```python
result = await db.query("""
    CREATE events SET 
        name = $name,
        date = $date,
        type = $type
""", {"name": "Storm", "date": "2025-01-01", "type": "storm"})
```

## Note

- SurrealQL ≠ SQL standard
- Usare `$param` per parametri
- Risultati sono liste di dict

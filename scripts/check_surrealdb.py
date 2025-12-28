#!/usr/bin/env python3
"""Check SurrealDB contents."""
import json
from surrealdb import Surreal

db = Surreal("ws://localhost:8001/rpc")
db.use("causal", "knowledge")

print("📊 DATABASE CONTENTS:")
for table in ["paper", "event", "pattern", "climate_index"]:
    result = db.query(f"SELECT count() FROM {table} GROUP ALL")
    count = result[0]["count"] if result else 0
    print(f"   {table}: {count}")

# Show first paper
result = db.query("SELECT title, year FROM paper LIMIT 3")
print("\n📄 Sample papers:")
for p in result:
    print(f"   - {p.get('title', 'N/A')[:60]}... ({p.get('year', 'N/A')})")

db.close()

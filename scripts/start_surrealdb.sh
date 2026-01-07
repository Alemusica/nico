#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# START SURREALDB - CTW Early Warning Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════
#
# Questo script avvia SurrealDB con i database per CTW:
#
# NAMESPACES:
#   surge_shazam/
#   ├── data_catalog     - API registry, data sources
#   ├── knowledge        - Papers, events, patterns
#   └── causal_graph     - Discovered causal chains
#
# USAGE:
#   ./scripts/start_surrealdb.sh         # Avvia in foreground
#   ./scripts/start_surrealdb.sh &       # Avvia in background
#   ./scripts/start_surrealdb.sh stop    # Ferma il database
#   ./scripts/start_surrealdb.sh status  # Verifica stato
#
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DB_PATH="$HOME/.surrealdb/data/surge_shazam"
PORT=8000

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  🌊 CTW - Causal Thunder Watch"
    echo "  📦 SurrealDB Knowledge Base"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

# Status command
if [ "$1" = "status" ]; then
    echo -e "${YELLOW}Checking SurrealDB status...${NC}"
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ SurrealDB is running on port $PORT${NC}"
        
        # Show stats
        echo ""
        echo "Namespace: surge_shazam"
        curl -s -X POST "http://localhost:$PORT/sql" \
            -H "Accept: application/json" \
            -H "Authorization: Basic cm9vdDpyb290" \
            -H "surreal-ns: surge_shazam" \
            -H "surreal-db: data_catalog" \
            -d "SELECT count() FROM data_source GROUP ALL;" 2>/dev/null | \
            python3 -c "import sys, json; d=json.load(sys.stdin); print(f'  data_sources: {d[0].get(\"result\", [{}])[0].get(\"count\", 0) if d else 0}')" 2>/dev/null || echo "  (query failed)"
    else
        echo -e "${RED}✗ SurrealDB is not running${NC}"
    fi
    exit 0
fi

# Stop command
if [ "$1" = "stop" ]; then
    echo -e "${YELLOW}Stopping SurrealDB...${NC}"
    pkill -f "surreal start" 2>/dev/null
    sleep 1
    if lsof -i :$PORT > /dev/null 2>&1; then
        echo -e "${RED}✗ Failed to stop (port still in use)${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ SurrealDB stopped${NC}"
    exit 0
fi

# Check if already running
if lsof -i :$PORT > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ SurrealDB already running on port $PORT${NC}"
    echo "  Use './scripts/start_surrealdb.sh stop' to stop it first"
    echo "  Or './scripts/start_surrealdb.sh status' to check"
    exit 1
fi

# Check database path exists
if [ ! -d "$DB_PATH" ]; then
    echo -e "${YELLOW}Creating database directory: $DB_PATH${NC}"
    mkdir -p "$DB_PATH"
fi

print_banner

echo -e "  Database: ${YELLOW}$DB_PATH${NC}"
echo -e "  URL:      ${YELLOW}http://localhost:$PORT${NC}"
echo -e "  Auth:     ${YELLOW}root:root${NC}"
echo ""

echo -e "${GREEN}Namespaces:${NC}"
echo "  surge_shazam/"
echo "  ├── data_catalog   - API registry"
echo "  ├── knowledge      - Papers, events"
echo "  └── causal_graph   - Discovered chains"
echo ""

echo -e "${GREEN}Query examples:${NC}"
echo '  # List all data sources'
echo '  surreal sql --endpoint http://127.0.0.1:8000 -u root -p root \'
echo '    --ns surge_shazam --db data_catalog <<< "SELECT * FROM data_source"'
echo ''
echo '  # Get real-time sources'
echo '  surreal sql --endpoint http://127.0.0.1:8000 -u root -p root \'
echo '    --ns surge_shazam --db data_catalog <<< "SELECT * FROM data_source WHERE latency_hours < 24"'
echo ""

echo -e "${GREEN}Seeding:${NC}"
echo "  python scripts/seed_api_registry.py     # Import API registry"
echo "  python scripts/seed_knowledge_graph.py  # Import papers/events"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo ""

# Start SurrealDB with RocksDB storage (persistent)
echo -e "${GREEN}Starting SurrealDB...${NC}"
exec surreal start \
    --user root \
    --pass root \
    --bind 0.0.0.0:$PORT \
    "rocksdb:$DB_PATH"

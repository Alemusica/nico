#!/bin/bash
################################################################################
# Production-Grade Startup Script
# Handles: process cleanup, dependency checks, graceful startup, health checks
################################################################################

set -e  # Exit on error

PROJECT_DIR="/Users/alessioivoycazzaniga/nico"
VENV_DIR="$PROJECT_DIR/.venv"
BACKEND_PORT=8000
FRONTEND_PORT=5173
LOG_DIR="$PROJECT_DIR/logs"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

# Create logs directory
mkdir -p "$LOG_DIR"

################################################################################
# 1. CLEANUP ZOMBIE PROCESSES
################################################################################
log_info "Cleaning up existing processes..."

# Kill uvicorn
if lsof -ti:$BACKEND_PORT > /dev/null 2>&1; then
    log_warning "Port $BACKEND_PORT occupied, killing process..."
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Kill vite
if lsof -ti:$FRONTEND_PORT > /dev/null 2>&1; then
    log_warning "Port $FRONTEND_PORT occupied, killing process..."
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Kill by process name (fallback)
pkill -9 -f "uvicorn api.main" 2>/dev/null || true
pkill -9 -f "vite" 2>/dev/null || true
sleep 1

log_success "Process cleanup complete"

################################################################################
# 2. DEPENDENCY CHECKS
################################################################################
log_info "Checking dependencies..."

# Check Python venv
if [ ! -d "$VENV_DIR" ]; then
    log_error "Virtual environment not found: $VENV_DIR"
    exit 1
fi

# Check Node modules
if [ ! -d "$PROJECT_DIR/frontend/node_modules" ]; then
    log_warning "Node modules not found, installing..."
    cd "$PROJECT_DIR/frontend"
    npm install
    cd "$PROJECT_DIR"
fi

log_success "Dependencies OK"

################################################################################
# 3. START BACKEND
################################################################################
log_info "Starting backend server..."

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

# Rotate logs if too large (>50MB)
if [ -f "$BACKEND_LOG" ]; then
    size=$(stat -f%z "$BACKEND_LOG" 2>/dev/null || stat -c%s "$BACKEND_LOG" 2>/dev/null || echo "0")
    if [ "$size" -gt 52428800 ]; then
        mv "$BACKEND_LOG" "$BACKEND_LOG.old"
        log_info "Rotated backend log (size: ${size} bytes)"
    fi
fi

# Start uvicorn with proper logging
nohup uvicorn api.main:app \
    --host 0.0.0.0 \
    --port $BACKEND_PORT \
    --reload \
    --log-level info \
    --access-log \
    > "$BACKEND_LOG" 2>&1 &

BACKEND_PID=$!
log_info "Backend starting (PID: $BACKEND_PID)..."

# Wait for backend health
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:$BACKEND_PORT/api/v1/health > /dev/null 2>&1; then
        log_success "Backend healthy at http://localhost:$BACKEND_PORT"
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    log_error "Backend failed to start after ${max_attempts}s"
    log_error "Check logs: tail -f $BACKEND_LOG"
    exit 1
fi

################################################################################
# 4. START FRONTEND
################################################################################
log_info "Starting frontend server..."

cd "$PROJECT_DIR/frontend"

# Rotate logs if too large
if [ -f "$FRONTEND_LOG" ]; then
    size=$(stat -f%z "$FRONTEND_LOG" 2>/dev/null || stat -c%s "$FRONTEND_LOG" 2>/dev/null || echo "0")
    if [ "$size" -gt 52428800 ]; then
        mv "$FRONTEND_LOG" "$FRONTEND_LOG.old"
        log_info "Rotated frontend log (size: ${size} bytes)"
    fi
fi

# Start Vite
nohup npm run dev > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
log_info "Frontend starting (PID: $FRONTEND_PID)..."

# Wait for frontend
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
        log_success "Frontend ready at http://localhost:$FRONTEND_PORT"
        break
    fi
    attempt=$((attempt + 1))
    sleep 1
done

if [ $attempt -eq $max_attempts ]; then
    log_warning "Frontend may not be ready (timeout after ${max_attempts}s)"
fi

################################################################################
# 5. SUMMARY
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_success "System started successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  🔹 Backend:   http://localhost:$BACKEND_PORT"
echo "  🔹 Frontend:  http://localhost:$FRONTEND_PORT"
echo "  🔹 API Docs:  http://localhost:$BACKEND_PORT/docs"
echo ""
echo "  📋 Backend PID:  $BACKEND_PID"
echo "  📋 Frontend PID: $FRONTEND_PID"
echo ""
echo "  📜 Logs:"
echo "     tail -f $BACKEND_LOG"
echo "     tail -f $FRONTEND_LOG"
echo ""
echo "  🛑 Stop: pkill -9 -f 'uvicorn|vite'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

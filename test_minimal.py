"""Test esecuzione - trova la funzione che rompe l'app."""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import streamlit as st

st.title("🧪 Test Esecuzione")

# Import tutto
from src.core.logging_config import setup_streamlit_logging, get_logger
from app.state import init_session_state
from app.styles import apply_custom_css
from app.components.sidebar import render_sidebar, AppConfig
from app.components.tabs import render_tabs, _render_empty_tabs
from app.components.globe import render_globe_landing

st.success("✅ Import OK")

# Test 1: Logging
try:
    logger = setup_streamlit_logging(level="DEBUG")
    st.success("✅ 1. setup_streamlit_logging OK")
except Exception as e:
    st.error(f"❌ 1. setup_streamlit_logging: {e}")

# Test 2: CSS
try:
    apply_custom_css()
    st.success("✅ 2. apply_custom_css OK")
except Exception as e:
    st.error(f"❌ 2. apply_custom_css: {e}")

# Test 3: Session state
try:
    init_session_state()
    st.success("✅ 3. init_session_state OK")
except Exception as e:
    st.error(f"❌ 3. init_session_state: {e}")

# Test 4: Sidebar
try:
    config = render_sidebar()
    st.success(f"✅ 4. render_sidebar OK - config: {type(config)}")
except Exception as e:
    st.error(f"❌ 4. render_sidebar: {e}")
    import traceback
    st.code(traceback.format_exc())
    config = AppConfig()

# Test 5: Empty tabs (questo è quello che viene chiamato quando non ci sono dati)
try:
    _render_empty_tabs(config)
    st.success("✅ 5. _render_empty_tabs OK")
except Exception as e:
    st.error(f"❌ 5. _render_empty_tabs: {e}")
    import traceback
    st.code(traceback.format_exc())

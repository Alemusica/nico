#!/usr/bin/env python
"""Quick import test for export tabs."""
import sys
sys.path.insert(0, '.')
from app.components.tabs import render_tabs, _render_cmems_l4_export_tab, _render_unified_export_tab
print('✅ All export tab functions import successfully')

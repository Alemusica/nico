"""Minimal Streamlit test."""
import streamlit as st

st.set_page_config(page_title="Test", layout="wide")

st.title("🧪 Minimal Test")

# Test selectbox with key
val = st.sidebar.selectbox(
    "Test Select",
    ["A", "B", "C"],
    key="test_select_unique_123"
)

st.write(f"Selected: {val}")

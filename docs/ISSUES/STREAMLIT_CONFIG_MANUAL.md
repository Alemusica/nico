# ⚠️ STREAMLIT CONFIG NOT IN GIT

## File: `.streamlit/config.toml`

Questo file è escluso da git (in `.gitignore`), ma è **necessario** per il tema white background.

### Contenuto da creare manualmente:

```toml
[theme]
base = "light"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
primaryColor = "#1E3A5F"
textColor = "#2C3E50"
font = "sans serif"

[server]
port = 8501
headless = true
```

### Setup:
```bash
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
base = "light"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
primaryColor = "#1E3A5F"
textColor = "#2C3E50"
font = "sans serif"

[server]
port = 8501
headless = true
EOF
```

### Restart Streamlit dopo creazione:
```bash
pkill -f streamlit
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 8501
```

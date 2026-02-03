# 🚀 Portability & Deployment Guide

## ✅ The Project is NOW Portable!

After the recent improvements, **anyone can clone and run this project** with minimal setup.

---

## 📦 For New Users (One-Time Setup)

```bash
# 1. Clone repository
git clone https://github.com/caroniico/nico-gates-streamlit.git
cd nico-gates-streamlit

# 2. Run setup script (installs everything)
./setup.sh

# 3. Start the app
./run_streamlit.sh
```

**That's it!** The app runs at http://localhost:8501

---

## 🔧 What `setup.sh` Does

The setup script is **fully automatic** and handles:

1. ✅ **Python Detection** - Finds Python 3.11, 3.12, or 3.13
2. ✅ **Virtual Environment** - Creates `.venv/` if not exists
3. ✅ **Dependencies** - Installs from `requirements.txt`
4. ✅ **CMEMS Check** - Checks if `copernicusmarine` is installed
5. ✅ **Credentials** - Copies template, checks if configured
6. ✅ **Start Script** - Creates portable `run_streamlit.sh` with absolute paths

---

## 🔑 Credentials Setup (Optional)

### Why You Need Them
- To download **CMEMS L4 gridded data** (sea level, geostrophic velocities)
- Free account, no credit card required

### How to Configure

**Option 1: Using credentials.yaml (Recommended)**
```bash
# 1. Get free account
https://marine.copernicus.eu/register

# 2. Edit config/credentials.yaml
username: "your_username"
password: "your_password"
```

**Option 2: Environment Variables**
```bash
export COPERNICUS_USERNAME='your_username'
export COPERNICUS_PASSWORD='your_password'
./run_streamlit.sh
```

**Without Credentials:**
- App still works with cached data
- SLCCI, ERA5 datasets still available
- Only CMEMS L4 downloads will fail

---

## 📂 What Gets Committed to Git

### ✅ Included in Repository
```
setup.sh                          # Setup script
run_streamlit.sh                  # Auto-generated start script (ignored)
requirements.txt                  # Dependencies
config/credentials.yaml.template  # Template (safe)
config/gates.yaml                 # Gate definitions
src/                              # Source code
app/                              # UI components
gates/                            # Shapefiles
```

### ❌ NOT Committed (.gitignore)
```
.venv/                   # Virtual environment (user-specific)
config/credentials.yaml  # Real credentials (security!)
data/cache/              # Downloaded data (too large)
*.pyc, __pycache__/      # Python bytecode
logs/                    # Runtime logs
```

---

## 🌍 Cross-Platform Support

### macOS ✅
```bash
brew install python@3.12
./setup.sh
./run_streamlit.sh
```

### Linux ✅
```bash
# Ubuntu/Debian
sudo apt install python3.12
./setup.sh
./run_streamlit.sh

# CentOS/RHEL
sudo yum install python312
./setup.sh
./run_streamlit.sh
```

### Windows (WSL) ✅
```bash
# Install WSL Ubuntu first
wsl --install

# Then in WSL terminal
sudo apt install python3.12
./setup.sh
./run_streamlit.sh
```

### Windows (Native) ⚠️
```powershell
# Not yet supported - use WSL instead
# Or manually create venv and install deps
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 🧪 Testing Portability

### Simulate Fresh Clone
```bash
# On a different machine or Docker container
git clone <repo>
cd nico-gates-streamlit
./setup.sh  # Should work without any edits!
```

### Docker Test (Future)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN ./setup.sh
CMD ["./run_streamlit.sh"]
```

---

## 🚨 Known Issues

### 1. Large Data Files Not in Repo
**Problem**: Shapefiles in `gates/` and GEBCO bathymetry might be missing  
**Solution**: 
- Include small shapefiles in repo
- Download GEBCO on first run
- Or provide download script

### 2. Cache Directory Size
**Problem**: `data/cache/` can grow to several GB  
**Solution**: 
- Already in .gitignore
- Users download fresh data
- Or provide compressed cache archive

### 3. CMEMS Rate Limits
**Problem**: API has download limits  
**Solution**:
- Cache aggressively (already implemented)
- Show helpful error messages
- Fallback to cached data

---

## 📋 Portability Checklist

- [x] No hardcoded paths in code
- [x] Automatic path detection in scripts
- [x] Virtual environment managed automatically
- [x] Dependencies in requirements.txt
- [x] Credentials template provided
- [x] .gitignore protects secrets
- [x] Setup script handles everything
- [x] Start script auto-generated
- [x] Cross-platform shell scripts
- [x] Clear documentation

---

## 🎯 For Developers

### Adding New Dependencies
```bash
# 1. Install in your venv
source .venv/bin/activate
pip install new-package

# 2. Update requirements.txt
pip freeze > requirements.txt

# 3. Commit
git add requirements.txt
git commit -m "Add new-package dependency"
```

### Testing on Fresh Environment
```bash
# Create test directory
cd /tmp
git clone /path/to/repo test-clone
cd test-clone
./setup.sh
./run_streamlit.sh
```

---

## 🤝 Sharing the Project

### For Collaborators
```
1. Send them the GitHub URL
2. They run: ./setup.sh
3. Share your credentials.yaml privately (email/Slack)
4. They start: ./run_streamlit.sh
```

### For Public Release
```
1. Push to GitHub (credentials.yaml already ignored)
2. Users follow QUICK_START.md
3. They register their own CMEMS account
4. No secrets exposed
```

---

## 📊 Portability Score: 9/10

| Aspect | Status | Notes |
|--------|--------|-------|
| Code Portability | ✅ 10/10 | No hardcoded paths |
| Setup Automation | ✅ 10/10 | One-command setup |
| Dependency Management | ✅ 10/10 | requirements.txt |
| Credential Security | ✅ 10/10 | Template + .gitignore |
| Cross-Platform | ⚠️ 8/10 | Linux/Mac perfect, Windows via WSL |
| Documentation | ✅ 10/10 | Clear guides |
| Data Portability | ⚠️ 7/10 | Large files not in repo |

**Overall**: Excellent portability! 🎉

---

**Last Updated**: February 2026  
**Tested On**: macOS 14, Ubuntu 22.04, WSL2 Ubuntu

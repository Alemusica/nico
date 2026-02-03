# 🌊 NICO Streamlit Gates - Quick Start Guide

**One-command setup for analyzing Arctic straits with satellite altimetry**

---

## 📦 Installation (First Time Only)

```bash
# 1. Clone the repository
git clone https://github.com/caroniico/nico-gates-streamlit.git
cd nico-gates-streamlit

# 2. Run setup (installs everything automatically)
./setup.sh
```

**That's it!** The script will:
- ✅ Detect Python 3.11+
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Check CMEMS credentials
- ✅ Create portable start script

---

## 🚀 Running the App

```bash
# Start Streamlit
./run_streamlit.sh
```

Open your browser: **http://localhost:8501**

---

## 🔑 CMEMS Credentials (Optional but Recommended)

To download **CMEMS L4 gridded data** (sea level, geostrophic velocities):

### Option 1: Using credentials.yaml (Recommended)

1. Get free account: https://marine.copernicus.eu/register
2. Edit `config/credentials.yaml`:

```yaml
ceda:
  username: "your_copernicus_username"
  password: "your_copernicus_password"
```

### Option 2: Environment Variables

```bash
export COPERNICUS_USERNAME='your_username'
export COPERNICUS_PASSWORD='your_password'
./run_streamlit.sh
```

**Without credentials:** You can still use the app with cached data and other datasets (SLCCI, ERA5).

---

## 📊 What Can You Do?

### Available Arctic Straits
- 🇬🇱 **Fram Strait** (West/East) - Arctic-Atlantic gateway
- 🇬🇱 **Davis Strait** (West/East) - Baffin Bay exchange
- 🇩🇰 **Denmark Strait** - Iceland-Greenland overflow
- 🇳🇴 **Barents Opening** - Atlantic inflow
- 🇷🇺 **Bering Strait** - Pacific-Arctic connection
- 🇨🇦 **Nares Strait** - Canadian Archipelago

### Available Datasets
- 📡 **SLCCI** - ESA Sea Level CCI (along-track, 1993-2015)
- 🌊 **CMEMS L4** - Gridded sea level + velocities (1993-present)
- 🌍 **ERA5** - Wind/pressure reanalysis
- 🗺️ **GEBCO** - Bathymetry for volume transport

### Key Features
- 🧮 **Volume Transport** - Calculate ocean flux through straits
- 📈 **Dynamic Ocean Topography** - Sea level slopes
- 🌀 **Geostrophic Velocity** - Ocean currents from pressure gradients
- 📊 **Monthly Climatology** - Long-term patterns
- 📤 **Export** - High-quality PNG graphs + CSV data

---

## 🛠️ Requirements

- **Python**: 3.11, 3.12, or 3.13
- **OS**: macOS, Linux, Windows (WSL)
- **RAM**: 8GB minimum (16GB recommended for large datasets)
- **Disk**: 5GB for cache data

---

## 📂 Project Structure

```
nico-gates-streamlit/
├── setup.sh              # 🔧 One-command setup
├── run_streamlit.sh      # 🚀 Start the app (auto-generated)
├── streamlit_app.py      # 🎨 Main UI
├── config/
│   ├── gates.yaml        # Gate definitions
│   ├── credentials.yaml  # API credentials
│   └── datasets.yaml     # Dataset configs
├── src/
│   └── services/         # Data loading services
├── app/
│   └── components/       # UI components
├── data/                 # 💾 Cache directory
└── gates/                # Shapefiles
```

---

## 🐛 Troubleshooting

### "Python 3.11+ not found"
```bash
# macOS
brew install python@3.12

# Ubuntu/Debian
sudo apt install python3.12

# CentOS/RHEL
sudo yum install python312
```

### "CMEMS downloads fail"
1. Check credentials in `config/credentials.yaml`
2. Or set environment variables: `COPERNICUS_USERNAME`, `COPERNICUS_PASSWORD`
3. Test with: `copernicusmarine login --username YOUR_USER --password YOUR_PASS`

### "Import errors"
```bash
# Reinstall dependencies
source .venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

### "Port 8501 already in use"
```bash
# Find and kill existing Streamlit process
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run streamlit_app.py --server.port 8502
```

---

## 📚 Documentation

- **Full Docs**: See `docs/` directory
- **API Reference**: `docs/API_USAGE.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Feature List**: `docs/FEATURE_INVENTORY.md`

---

## 🤝 Contributing

See `docs/CONTRIBUTING.md` for development setup.

---

## 📧 Support

- **Issues**: https://github.com/caroniico/nico-gates-streamlit/issues
- **Docs**: Check `docs/` folder

---

## 🎯 Quick Test

After setup, test with:

```bash
./run_streamlit.sh
```

1. Select **"Fram Strait WEST"** from dropdown
2. Choose dataset **"CMEMS L4"**
3. Date range: **2020-01-01 to 2020-12-31**
4. Variables: **adt** (sea level)
5. Click **"📥 Load Data"**
6. Explore tabs: **Spatial Map**, **Bathymetry**, **DOT**, **Geostrophic Velocity**

---

**Last Updated**: February 2026
**Branch**: `feature/gates-streamlit`

# Contributing Guide

Thank you for your interest in contributing to the SLCCI Altimetry project!

## 🚀 Getting Started

### Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd nico

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black isort mypy
```

### Running the App

```bash
streamlit run streamlit_app.py
```

## 📝 Code Style

### Python Style

We follow **PEP 8** with these tools:
- **Black** for formatting (line length: 88)
- **isort** for import sorting
- **mypy** for type checking

```bash
# Format code
black src/ app/
isort src/ app/

# Type check
mypy src/ app/
```

### Docstrings

Use Google-style docstrings:

```python
def compute_slope(lon: np.ndarray, dot: np.ndarray, mean_latitude: float) -> SlopeResult:
    """
    Compute DOT slope using longitude binning and linear regression.
    
    Parameters
    ----------
    lon : np.ndarray
        Longitude values in degrees
    dot : np.ndarray
        DOT values in meters
    mean_latitude : float
        Mean latitude for unit conversion
        
    Returns
    -------
    SlopeResult
        Slope analysis results with slope_mm_per_m, r_squared, etc.
        
    Examples
    --------
    >>> result = compute_slope(lon, dot, 65.0)
    >>> print(result.slope_mm_per_m)
    0.0023
    """
```

### Type Hints

Always use type hints:

```python
# Good
def load_cycle(filepath: str | Path, decode_times: bool = False) -> xr.Dataset | None:
    ...

# Bad
def load_cycle(filepath, decode_times=False):
    ...
```

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed structure.

### Adding a New Feature

1. **Analysis Feature**:
   - Add to `src/analysis/`
   - Export from `__init__.py`
   - Add unit tests

2. **Visualization**:
   - Add to `src/visualization/plotly_charts.py` or `matplotlib_charts.py`
   - Use consistent styling

3. **Dashboard Tab**:
   - Create `app/components/new_tab.py`
   - Register in `app/components/tabs.py`

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/unit/test_slope.py -v
```

### Test Structure

```python
# tests/unit/test_slope.py
import numpy as np
import pytest
from src.analysis.slope import bin_by_longitude, compute_slope

class TestBinByLongitude:
    def test_basic_binning(self):
        lon = np.array([0.0, 0.005, 0.01, 0.015, 0.02])
        values = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        
        centers, means, stds, counts = bin_by_longitude(lon, values, 0.01)
        
        assert len(centers) == 2
        assert counts[0] == 2
        
    def test_empty_input(self):
        centers, means, _, _ = bin_by_longitude(np.array([]), np.array([]), 0.01)
        assert len(centers) == 0
```

## 📦 Commits

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Tests
- `chore`: Maintenance

**Examples**:
```
feat(analysis): add monthly slope computation

Add compute_monthly_slopes() function that calculates DOT slopes
grouped by month. Uses existing bin_by_longitude for consistency.

Closes #42
```

```
fix(loader): handle missing TimeDay variable

Some cycles don't have TimeDay variable. Now gracefully returns
None for date instead of crashing.
```

## 🔀 Pull Requests

1. **Create feature branch**:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make changes** following code style

3. **Add tests** for new functionality

4. **Run checks**:
   ```bash
   black src/ app/
   pytest tests/
   mypy src/
   ```

5. **Commit** with descriptive message

6. **Push and create PR**:
   ```bash
   git push origin feat/my-feature
   ```

### PR Checklist

- [ ] Code follows style guide
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation updated
- [ ] CHANGELOG updated (for significant changes)

## 📖 Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for structural changes
- Add docstrings to all public functions
- Update type hints

## 🐛 Reporting Issues

Include:
1. **Description** of the problem
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment** (Python version, OS, package versions)

## 💡 Feature Requests

Include:
1. **Use case** - Why do you need this?
2. **Proposed solution** - How should it work?
3. **Alternatives** - What else have you tried?

---

## Questions?

Feel free to open an issue for questions or discussions!

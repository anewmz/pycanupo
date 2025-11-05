# Installation Guide

## Python Dependencies

Install the Python dependencies from PyPI:

```bash
pip install -r requirements.txt
```

## CloudComPy Installation

CloudComPy is not available on PyPI and must be installed separately. It's a Python wrapper for CloudCompare.

### Option 1: Pre-built Binaries (Recommended)

1. **Download the binary** for your Python version from:
   https://www.simulation.openfields.fr/index.php/cloudcompy-downloads/

2. **Extract the archive** to a directory (e.g., `C:\CloudComPy310`)

3. **Add to Python path** (one of these methods):
   - Add the `CloudCompare` subdirectory to your `PYTHONPATH` environment variable
   - Or add this line to `main.py` (before other imports):
     ```python
     import sys
     sys.path.append(r"C:\path\to\CloudComPy310\CloudCompare")
     ```

4. **Set environment variable** (optional, for debug traces):
   ```bash
   set _CCTRACE_=ON
   ```

### Option 2: Conda Installation

If you use conda, you can create an environment with CloudComPy:

```bash
conda create -n cloudcompy python=3.11
conda activate cloudcompy
conda install -c conda-forge cloudcompy
```

### Verify Installation

Test that CloudComPy is installed correctly:

```python
import cloudComPy as cc
cc.initCC()
print("CloudComPy initialized successfully!")
```


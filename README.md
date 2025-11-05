# PyCANUPO

A modular, pure-Python implementation of the CANUPO point cloud classifier using **CloudComPy**, **scikit-learn**, and **NumPy**.

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

**Quick start:**
```bash
pip install -r requirements.txt
# Then install CloudComPy from binaries (see INSTALL.md)
```

## Structure

| File | Responsibility |
|------|----------------|
| `main.py` | CLI entry point and orchestration |
| `config.py` | Inline configuration for training and prediction paths |
| `io_utils.py` | I/O utilities for CloudComPy clouds |
| `features.py` | Multi-scale geometric feature extraction |
| `model.py` | Training and prediction logic |
| `metrics.py` | Accuracy and evaluation metrics |
| `compare.py` | Comparison between qCANUPO and Python model |

## Usage

### Command Line Interface

```bash
python main.py train --class1 data/class0/*.txt --class2 data/class1/*.txt -o model.pkl
python main.py predict --model model.pkl --cloud scene.txt -o classified.bin
```

### Inline Configuration

Edit `config.py` to set your training and prediction paths, then set `RUN_INLINE = True` to run without CLI arguments.

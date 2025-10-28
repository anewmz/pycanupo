# PyCANUPO

A modular, pure-Python re-implementation of the CANUPO point cloud classifier using **CloudComPy**, **scikit-learn**, and **NumPy**.

## Structure

| File | Responsibility |
|------|----------------|
| `main.py` | CLI entry point and orchestration |
| `io_utils.py` | I/O utilities for CloudComPy clouds |
| `features.py` | Multi-scale geometric feature extraction |
| `model.py` | Training and prediction logic |
| `metrics.py` | Accuracy and evaluation metrics |
| `compare.py` | Comparison between qCANUPO and Python model |

## Usage

```bash
python main.py train --class1 data/class0/*.txt --class2 data/class1/*.txt -o model.pkl
python main.py predict --model model.pkl --cloud scene.txt -o classified.bin
```

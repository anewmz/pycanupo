"""
Main entry point for the PyCANUPO project.
Handles command-line interface and inline configuration.
"""

import os
import sys
import argparse
from pathlib import Path
from types import SimpleNamespace

# CloudComPy setup (configure if needed)
# sys.path.append(r"C:\Users\alsay\CloudComPy310\CloudCompare")
# os.environ["_CCTRACE_"] = "ON"  # only if you want C++ debug traces

from model import cmd_train, cmd_predict
from compare import cmd_compare

# Import inline configuration
try:
    from config import (
        CLASS0_FILES, CLASS1_FILES, SCENES, SCENE_OUTS,
        PKL_OUT, PYPRM_OUT, LABEL0, LABEL1,
        RADII_ARG, KNN_FOR_AUTO, LEVELS_FOR_AUTO,
        MODEL_TYPE, CONF_THRESHOLD, RUN_INLINE
    )
except ImportError:
    # Fallback if config.py doesn't exist - use CLI mode
    RUN_INLINE = False
    CLASS0_FILES = CLASS1_FILES = SCENES = SCENE_OUTS = []
    PKL_OUT = PYPRM_OUT = ""
    LABEL0 = LABEL1 = ""
    RADII_ARG = "0.05,0.10,0.20"
    KNN_FOR_AUTO = 16
    LEVELS_FOR_AUTO = 4
    MODEL_TYPE = "logreg"
    CONF_THRESHOLD = None


def build_parser():
    """Builds the CLI parser and registers subcommands."""
    parser = argparse.ArgumentParser(description="Pure-Python CANUPO-style classifier (CloudComPy).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- train
    pt = sub.add_parser("train", help="train a binary classifier from labeled clouds")
    pt.add_argument("--class1", nargs="+", default=[], help="files for class 0")
    pt.add_argument("--class2", nargs="+", default=[], help="files for class 1")
    pt.add_argument("--label0", default="class0", help="name for class 0")
    pt.add_argument("--label1", default="class1", help="name for class 1")
    pt.add_argument("--radii", default="0.05,0.10,0.20", help="comma-separated radii (same units as cloud) or 'auto'")
    pt.add_argument("--knn", type=int, default=16, help="k for auto-radii estimation (if radii='auto')")
    pt.add_argument("--levels", type=int, default=4, help="number of levels for auto-radii (if radii='auto')")
    pt.add_argument("--model", choices=["svm", "logreg"], default="svm")
    pt.add_argument("--C", type=float, default=10.0, help="SVM C (if svm)")
    pt.add_argument("--gamma", default="scale", help="SVM gamma (if svm)")
    pt.add_argument("--kernel", default="rbf", help="SVM kernel (if svm)")
    pt.add_argument("-o", "--out", required=True, help="output .pkl")
    pt.add_argument("--pyprm", default=None, help="output .pyprm path (auto-generated from --out if logreg and not specified)")
    pt.set_defaults(func=cmd_train)

    # ---- predict
    pp = sub.add_parser("predict", help="classify a new cloud")
    pp.add_argument("--model", required=True, help="trained model .pkl")
    pp.add_argument("--cloud", required=True, help="input cloud (PLY, LAS/LAZ, BIN, ASC, etc.)")
    pp.add_argument("--threshold", type=float, default=None, help="optional confidence threshold for 'reject'")
    pp.add_argument("-o", "--out", required=True, help="output path (e.g., classified.bin)")
    pp.set_defaults(func=cmd_predict)

    # ---- compare
    pc = sub.add_parser("compare", help="Run qCANUPO (.prm) and your model on the same cloud, save both SFs, and print metrics")
    pc.add_argument("--prm_path", required=True, help="qCANUPO .prm file")
    pc.add_argument("--model", required=True, help=".pyprm or .pkl for your Python model")
    pc.add_argument("--cloud", required=True, help="input cloud")
    pc.add_argument("--threshold", type=float, default=None, help="reject our model if confidence<threshold (sets -1)")
    pc.add_argument("-o", "--out", required=True, help="output path (e.g., compare.bin)")
    pc.set_defaults(func=cmd_compare)

    return parser


def main():
    """Entry point that runs inline config or CLI depending on setup."""
    if RUN_INLINE:
        # 1) TRAIN (call cmd_train without CLI)
        # NOTE: in your code, --class1 maps to label 0, --class2 maps to label 1
        train_args = SimpleNamespace(
            class1=CLASS0_FILES,
            class2=CLASS1_FILES,
            label0=LABEL0,
            label1=LABEL1,
            radii=RADII_ARG,           # "auto" or "0.03,0.06,..."
            knn=KNN_FOR_AUTO,
            levels=LEVELS_FOR_AUTO,
            model=MODEL_TYPE,          # "logreg" or "svm"
            C=10.0,
            gamma="scale",
            kernel="rbf",
            out=PKL_OUT,
            pyprm=PYPRM_OUT if MODEL_TYPE == "logreg" else None,
        )
        cmd_train(train_args)

        # 2) PREDICT on each scene
        assert len(SCENES) == len(SCENE_OUTS), "SCENES and SCENE_OUTS must have the same length"
        for scene_path, out_path in zip(SCENES, SCENE_OUTS):
            model_path = PYPRM_OUT if (MODEL_TYPE == "logreg" and Path(PYPRM_OUT).is_file()) else PKL_OUT
            predict_args = SimpleNamespace(
                model=model_path,
                cloud=scene_path,
                threshold=CONF_THRESHOLD,
                out=out_path,
            )
            cmd_predict(predict_args)

    else:
        # Keep the original CLI behavior (train/predict/compare subcommands)
        parser = build_parser()
        args = parser.parse_args()
        args.func(args)


if __name__ == "__main__":
    main()

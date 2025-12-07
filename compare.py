"""
Compares results between qCANUPO and PyCANUPO on the same cloud.
"""

import json
import numpy as np
from pathlib import Path
from joblib import load
import cloudComPy as cc

from io_utils import read_cloud, cloud_to_numpy, add_or_replace_scalar_field, get_scalar_array, majority_smooth_labels
from features import multiscale_features
from metrics import _safe_metrics


def cmd_compare(args):
    """
    Compare qCANUPO (.prm) vs our Python model (.pyprm or .pkl) on the SAME cloud.
    Produces one output entity with 4 SFs:
      - CANUPO.class, CANUPO.confidence
      - PYCANUPO.class, PYCANUPO.confidence
    And prints metrics (treating CANUPO as reference).
    """
    # --- Load cloud (once) ---
    cloud = read_cloud(args.cloud)

    # --- 1) Run qCANUPO classification with .prm ---
    if not cc.isPluginCanupo():
        raise RuntimeError("CANUPO plugin not available in this CloudComPy build.")
    import cloudComPy.Canupo  # ensure plugin symbols are loaded

    ok = cc.Canupo.Classify(cloud, args.prm_path)
    if not ok:
        raise RuntimeError("qCANUPO classification failed")

    # Make sure SF names exist
    sf_dic = cloud.getScalarFieldDic()
    if "CANUPO.class" not in sf_dic or "CANUPO.confidence" not in sf_dic:
        raise RuntimeError("Expected qCANUPO scalar fields missing")

    # --- 2) Run OUR model on the same cloud ---
    # Prepare features
    P = cloud_to_numpy(cloud)

    # Load model (pyprm or pkl)
    portable_mode = args.model.lower().endswith(".pyprm")
    if portable_mode:
        model = json.loads(Path(args.model).read_text())
        radii = model["radii"]
        Z = multiscale_features(P, radii)
        mean  = np.array(model["scaler"]["mean"], dtype=np.float64)
        scale = np.array(model["scaler"]["scale"], dtype=np.float64)
        info  = model["clf"]
        if info["type"] != "logreg":
            raise RuntimeError("This .pyprm is not a logistic-regression export.")
        W = np.array(info["coef"], dtype=np.float64)
        b = float(info["intercept"][0])
        Xs = (Z - mean) / scale
        z = Xs.dot(W.T).ravel() + b
        p1 = 1.0/(1.0+np.exp(-z))
        probs_py = np.column_stack([1.0 - p1, p1])
    else:
        pack = load(args.model)
        radii = pack["radii"]
        Z = multiscale_features(P, radii)
        probs_py = pack["sk_pipeline"].predict_proba(Z)

    y_py = probs_py.argmax(axis=1)
    c_py = probs_py.max(axis=1)

    # optional reject by threshold
    if args.threshold is not None:
        thr = float(args.threshold)
        y_py = np.where(c_py >= thr, y_py, -1)

    if getattr(args, "smooth", False):
        print(f"[info] Applying majority smoothing to Python model results (radius={getattr(args,'smooth_radius',None)} k={getattr(args,'smooth_k',None)})")
        y_py = majority_smooth_labels(P, y_py, radius=getattr(args, "smooth_radius", None), k=getattr(args, "smooth_k", None), min_neighbors=getattr(args, "smooth_min_neighbors", 3))

    # --- 3) Save OUR fields on the same entity ---
    idx_cls_py = add_or_replace_scalar_field(cloud, "PYCANUPO.class", y_py.astype(float))
    _          = add_or_replace_scalar_field(cloud, "PYCANUPO.confidence", c_py.astype(float))

    # --- 4) Export one file containing both sets of SFs ---
    out_path = Path(args.out)
    ok = cc.SaveEntities([cloud], str(out_path))
    if not ok:
        raise RuntimeError(f"Failed to save {out_path}")

    # --- 5) Metrics (treat CANUPO as reference) ---
    y_ref = get_scalar_array(cloud, "CANUPO.class").astype(int)
    # Ignore any points where CANUPO marked invalid (<0)
    valid = y_ref >= 0
    y_ref = y_ref[valid]
    y_cmp = y_py[valid]

    metrics = _safe_metrics(y_ref, y_cmp)

    print("\n=== qCANUPO vs Python Model (CANUPO as reference) ===")
    print(f"Used points: {metrics['used_pts']} / {valid.sum()} (excludes rejected by threshold)")
    print("Confusion matrix (rows=ref, cols=pred):")
    print(metrics["cm"])
    print(f"Accuracy: {metrics['acc']:.4f} | Cohen's κ: {metrics['kappa']:.4f}")
    print(f"Precision per class: {metrics['prec']}")
    print(f"Recall    per class: {metrics['rec']}")
    print(f"F1        per class: {metrics['f1']}")
    print(f"IoU       per class: {metrics['iou']}")
    print(f"[OK] Saved combined output → {out_path}")

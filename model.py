"""
Training and prediction logic for PyCANUPO models.
"""

import json
import numpy as np
from pathlib import Path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from io_utils import read_cloud, cloud_to_numpy, add_or_replace_scalar_field, ensure_parent_dir
from features import multiscale_features, estimate_radii


def cmd_train(args):
    """
    Train using two (or more) labeled inputs.
    For simplicity, we accept:
      --class1 file1.las file2.ply ...
      --class2 file3.las ...
    All points in each file inherit that class label (binary).
    """

    # --- radii: explicit list or 'auto'
    if isinstance(args.radii, str) and args.radii.lower() == "auto":
        # Use the first available training file to estimate spacing
        src_list = args.class1 if args.class1 else args.class2
        if not src_list:
            raise RuntimeError("Auto-radii requested but no training files were provided.")
        _cloud0 = read_cloud(src_list[0])
        _X0 = cloud_to_numpy(_cloud0)
        # args.knn and args.levels must exist (CLI parser or inline caller sets them)
        radii = estimate_radii(_X0, k=getattr(args, "knn", 16), n_levels=getattr(args, "levels", 4))
        print(f"[auto-radii] picked: {radii}")
    else:
        radii = [float(x) for x in str(args.radii).split(",")]

    # containers for training data
    cls_names = []
    X_list, y_list = [], []

    def ingest(files, label):
        for p in files:
            cloud = read_cloud(p)
            arr = cloud_to_numpy(cloud)
            Z = multiscale_features(arr, radii)
            X_list.append(Z)
            y_list.append(np.full(Z.shape[0], label, dtype=int))
            cls_names.append(Path(p).name)

    if args.class1:
        ingest(args.class1, 0)
    if args.class2:
        ingest(args.class2, 1)

    if not X_list:
        raise RuntimeError("No training data provided.")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)


    clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf)
    ])

    pipe.fit(X, y)

    out = {
        "radii": radii,
        "model_type": "logreg",
        "sk_pipeline": pipe,
        "meta": {
            "class_map": {0: args.label0, 1: args.label1},
            "sources": cls_names,
        }
    }
    ensure_parent_dir(args.out)
    dump(out, args.out)
    print(f"[OK] Trained model saved to: {args.out}")
    print(f"     Radii: {radii}")
    print(f"     Classes: 0→{args.label0}, 1→{args.label1}")

    # Export a portable JSON (.pyprm) for logistic regression
    try:
        scaler = pipe.named_steps["scaler"]
        lr     = pipe.named_steps["clf"]

        model_json = {
            "version": 2,
            "radii": radii,
            "scaler": {
                "mean":  scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist()
            },
            "clf": {
                "type": "logreg",
                "coef":  lr.coef_.tolist(),
                "intercept": lr.intercept_.tolist()
            },
            "features_per_scale": 6,
            "class_map": {"0": args.label0, "1": args.label1}
        }

        # Decide target path
        target_pyprm = getattr(args, "pyprm", None)
        if not target_pyprm:
            target_pyprm = str(Path(args.out).with_suffix(".pyprm"))

        # Write
        Path(target_pyprm).parent.mkdir(parents=True, exist_ok=True)
        Path(target_pyprm).write_text(json.dumps(model_json, indent=2))

        print(f"[OK] Portable model saved to: {target_pyprm}")
    except Exception as e:
        print(f"[error] .pyprm export failed: {e}")


def cmd_predict(args):
    """Predict classes on a new cloud and write scalar fields back.

    Supports:
      - .pkl  (joblib pipeline with scaler+clf)
      - .pyprm (portable JSON: radii + scaler stats + logreg weights)
    """
    import cloudComPy as cc  # type: ignore
    
    # --- detect model format ---
    model_path = str(args.model)
    portable_mode = model_path.lower().endswith(".pyprm")

    if portable_mode:
        # Load portable JSON (logreg only)
        model = json.loads(Path(model_path).read_text())
        radii = model["radii"]
        scaler_mean  = np.array(model["scaler"]["mean"],  dtype=np.float64)
        scaler_scale = np.array(model["scaler"]["scale"], dtype=np.float64)
        clf_info = model["clf"]
        if clf_info["type"] != "logreg":
            raise RuntimeError("This .pyprm is not a logistic-regression export. Use the .pkl instead.")
        W = np.array(clf_info["coef"], dtype=np.float64)  # (1, F)
        b = float(clf_info["intercept"][0])
        class_map = {0: "class0", 1: "class1"}
        if "class_map" in model:
            class_map = {int(k): v for k, v in model["class_map"].items()}
    else:
        # Load joblib pipeline (.pkl)
        pack = load(model_path)
        radii = pack["radii"]
        pipe  = pack["sk_pipeline"]
        class_map = pack["meta"]["class_map"]

    # --- load cloud and compute features ---
    cloud = read_cloud(args.cloud)
    P = cloud_to_numpy(cloud)
    Z = multiscale_features(P, radii)

    # --- predict probabilities ---
    if portable_mode:
        Xs = (Z - scaler_mean) / scaler_scale
        z  = Xs.dot(W.T).ravel() + b
        p1 = 1.0 / (1.0 + np.exp(-z))
        probs = np.column_stack([1.0 - p1, p1])
    else:
        probs = pipe.predict_proba(Z)

    yhat = probs.argmax(axis=1)
    conf = probs.max(axis=1)



    # --- write scalar fields ---
    idx_class = add_or_replace_scalar_field(cloud, "PYCANUPO.class", yhat.astype(float))
    _         = add_or_replace_scalar_field(cloud, "PYCANUPO.confidence", conf.astype(float))

    # --- set displayed SF safely ---
    try:
        sf_dic = cloud.getScalarFieldDic()
        sf_idx = idx_class if isinstance(idx_class, int) and idx_class >= 0 else sf_dic.get("PYCANUPO.class", None)
        if isinstance(sf_idx, int) and sf_idx >= 0:
            cloud.setCurrentDisplayedScalarField(sf_idx)
            cloud.showSF(True)
        else:
            print("[warn] could not resolve scalar field index to display")
    except Exception as e:
        print(f"[warn] setCurrentDisplayedScalarField failed: {e}")

    # --- save output ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cc.SaveEntities([cloud], str(out_path))
    if not ok:
        raise RuntimeError(f"Failed to save output: {out_path}")

    print(f"[OK] Predicted classes written. Saved: {out_path}")
    print(f"     class_map: {json.dumps(class_map)}")


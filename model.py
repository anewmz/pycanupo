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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import expon, randint

from io_utils import read_cloud, cloud_to_numpy, add_or_replace_scalar_field, ensure_parent_dir, majority_smooth_labels
from features import multiscale_features, estimate_radii
from metrics import _safe_metrics


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

    # Model: choose based on args.model
    if args.model == "svm":
        base = SVC(C=args.C, gamma=args.gamma, kernel=args.kernel, probability=False, class_weight="balanced")
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    elif args.model == "rf":
        clf = RandomForestClassifier(n_estimators=getattr(args, "n_estimators", 200),
                                     max_depth=getattr(args, "max_depth", None),
                                     class_weight="balanced",
                                     n_jobs=-1)
    else:
        # logistic regression
        clf = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=None)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf)
    ])

    # Optionally run a randomized search hyperparameter tuning step
    if getattr(args, "tune", False):
        print("[info] Running randomized hyperparameter search (this may take a while)")
        # choose parameter spaces depending on model choice
        if args.model == "svm":
            # tune SVC C and gamma (and kernel if desired)
            param_dist = {
                "clf__base_estimator__C": expon(scale=10),
                "clf__base_estimator__gamma": expon(scale=0.1),
            }
        elif args.model == "rf":
            param_dist = {
                "clf__n_estimators": randint(50, 500),
                "clf__max_depth": randint(3, 50),
            }
        else:
            # logistic regression
            param_dist = {"clf__C": expon(scale=10)}

        rnd = RandomizedSearchCV(pipe, param_dist, n_iter=getattr(args, "n_iter", 30),
                                 scoring=getattr(args, "scoring", "f1_macro"),
                                 cv=getattr(args, "cv", 5),
                                 n_jobs=getattr(args, "n_jobs", -1),
                                 random_state=0,
                                 verbose=1)
        rnd.fit(X, y)
        pipe = rnd.best_estimator_
        print(f"[OK] Best params (random search): {rnd.best_params_}")
    else:
        pipe.fit(X, y)

    # ========= Compute and print training accuracy =========
    y_pred = pipe.predict(X)
    metrics = _safe_metrics(y, y_pred)

    print("\n[train] metrics on training data")
    print("Confusion matrix (rows=true, cols=pred):")
    print(metrics["cm"])
    print(f"Accuracy: {metrics['acc']:.4f}")
    print(f"Precision per class: {metrics['prec']}")
    print(f"Recall    per class: {metrics['rec']}")
    print(f"F1        per class: {metrics['f1']}")
    print(f"IoU       per class: {metrics['iou']}")
    print(f"Cohen's kappa: {metrics['kappa']:.4f}")
    # ============================================================

    # Also save metrics to a text file next to the PKL model
    metrics_path = str(Path(args.out).with_stem(Path(args.out).stem + "_metrics"))
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"Training accuracy: {float(metrics['acc']):.6f}\n")
            f.write("Confusion matrix (rows=true, cols=pred):\n")
            f.write(repr(metrics["cm"]) + "\n")
            f.write("Precision per class:\n")
            f.write(repr(metrics["prec"]) + "\n")
            f.write("Recall per class:\n")
            f.write(repr(metrics["rec"]) + "\n")
            f.write("F1 per class:\n")
            f.write(repr(metrics["f1"]) + "\n")
            f.write("IoU per class:\n")
            f.write(repr(metrics["iou"]) + "\n")
            f.write(f"Cohen's kappa: {float(metrics['kappa']):.6f}\n")
            print(f"[info] Training metrics saved to: {metrics_path}")
    except Exception as e:
        print(f"[warn] Could not write metrics file: {e}")

    out = {
        "radii": radii,
        "model_type": args.model,
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

    # --- ALWAYS try to export a portable JSON (.pyprm) when using logreg ---
    # If args.pyprm is missing/None, default to <args.out>.pyprm in the same folder.
    if args.model == "logreg":
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
                # changed to match eigen_features() above => 9 per scale
                "features_per_scale": 9,
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
    else:
        print("[info] Skipping .pyprm (model is not 'logreg').")


def cmd_predict(args):
    """Predict classes on a new cloud and write scalar fields back.

    Supports:
      - .pkl  (joblib pipeline with scaler+clf)
      - .pyprm (portable JSON: radii + scaler stats + logreg weights)
    """
    try:
        import cloudComPy as cc  # type: ignore
    except ImportError:
        raise RuntimeError("CloudComPy not found. Install it with: pip install cloudcompy")
    
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

    # threshold → reject to -1
    if args.threshold is not None:
        thr = float(args.threshold)
        yhat = np.where(conf >= thr, yhat, -1)

    # optional spatial smoothing (majority vote) to reduce salt-and-pepper noise
    if getattr(args, "smooth", False):
        print(f"[info] Applying majority smoothing (radius={getattr(args, 'smooth_radius', None)} k={getattr(args,'smooth_k', None)})")
        yhat = majority_smooth_labels(P, yhat, radius=getattr(args, "smooth_radius", None), k=getattr(args, "smooth_k", None), min_neighbors=getattr(args, "smooth_min_neighbors", 3))

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
    if args.threshold is not None:
        n_reject = int((yhat == -1).sum())
        print(f"     Rejected (confidence<{args.threshold}): {n_reject} / {len(yhat)}")

    # --- optional portable JSON export for logreg models (from .pkl) ---
    # In cmd_predict, args.model is a file path, not "logreg"/"svm".
    # We only try to export a .pyprm if:
    #   - we are NOT in portable_mode (i.e. we loaded a .pkl pipeline), and
    #   - the underlying classifier is actually LogisticRegression.
    if not portable_mode:
        try:
            clf = pipe.named_steps.get("clf", None)
            if isinstance(clf, LogisticRegression):
                scaler = pipe.named_steps["scaler"]
                lr     = clf
                model_json = {
                    "version": 2,
                    "radii": radii,
                    "scaler": {
                        "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist()
                    },
                    "clf": {
                        "type": "logreg",
                        "coef": lr.coef_.tolist(),
                        "intercept": lr.intercept_.tolist()
                    },
                    "features_per_scale": 9,
                    "class_map": {
                        "0": class_map.get(0, "class0"),
                        "1": class_map.get(1, "class1")
                    }
                }
                target_pyprm = getattr(args, "pyprm", None) or str(Path(args.out).with_suffix(".pyprm"))
                Path(target_pyprm).parent.mkdir(parents=True, exist_ok=True)
                Path(target_pyprm).write_text(json.dumps(model_json, indent=2))
                print(f"[OK] Portable model saved to: {target_pyprm}")
            else:
                print("[info] Skipping .pyprm export: classifier is not LogisticRegression.")
        except Exception as e:
            print(f"[error] .pyprm export in predict() failed: {e}")
    else:
        # We are already using a portable .pyprm model, nothing to export here.
        print("[info] Skipping .pyprm export in predict(): model is already portable (.pyprm).")


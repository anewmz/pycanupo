"""
Inline configuration for PyCANUPO.
Edit this file to configure training and prediction paths when using inline mode.
Set RUN_INLINE = True to use this configuration instead of CLI.
"""

# =================== INLINE CONFIG (edit me) ===================
# If you already have local files, just fill the LOCAL PATHS below and
# leave the URL entries as None.

# -- class 0 (e.g., rock/ground)
CLASS0_FILES = [
    r"C:\pointcloud\data\train\class0\rock_A.txt",
    r"C:\pointcloud\data\train\class0\rock_B.txt",
]
CLASS0_URLS = [
    None,  # e.g. "https://example.com/rock_A.txt"
    None,  # e.g. "https://example.com/rock_B.txt"
]

# -- class 1 (e.g., vegetation)
CLASS1_FILES = [
    r"C:\pointcloud\data\train\class1\veg_A.txt",
    r"C:\pointcloud\data\train\class1\veg_B.txt",
]
CLASS1_URLS = [
    None,
    None,
]

# A scene (or several) to classify after training
SCENES = [
    r"C:\pointcloud\data\scenes\scene_01.txt",
]
SCENE_URLS = [
    None,
]

# Model outputs
PKL_OUT  = r"C:\pointcloud\models\rock_vs_veg.pkl"
PYPRM_OUT = r"C:\pointcloud\models\rock_vs_veg.pyprm"

# Classified outputs (same length/order as SCENES)
SCENE_OUTS = [
    r"C:\pointcloud\out\scene_01_py.bin",
]

# Labels and training knobs
LABEL0 = "rock"
LABEL1 = "veg"

# Radii: "auto" or comma list like "0.03,0.06,0.12,0.24"
RADII_ARG = "auto"
KNN_FOR_AUTO   = 16
LEVELS_FOR_AUTO = 4

# Model: "logreg" (portable .pyprm) or "svm" (.pkl only)
MODEL_TYPE = "logreg"

# Prediction threshold (None, or e.g., 0.6–0.8)
CONF_THRESHOLD = 0.7

# Turn this ON to run inline without using CLI
RUN_INLINE = False
# ================================================================


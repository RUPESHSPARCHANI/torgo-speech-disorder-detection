"""
==============================================================================
config.py — Central Configuration for TORGO ML Pipeline
==============================================================================
All constants, paths, and shared settings used across pipeline phases.
"""

import sys
import os
import warnings
from pathlib import Path

# Fix Windows console encoding for Unicode output
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\DELL\Downloads\NIT SIKKIM\torgo")
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES_CSV = OUTPUT_DIR / "extracted_features.csv"

# ─── Audio Parameters ────────────────────────────────────────────────────────
SAMPLE_RATE = 16000        # TORGO is recorded at 16 kHz
N_MFCC = 13               # Number of MFCC coefficients
MAX_DURATION = 5.0         # Max duration in seconds to consider

# ─── ML Parameters ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
N_ESTIMATORS = 100         # Random Forest trees

# ─── Dataset Mapping ─────────────────────────────────────────────────────────
# Use only arrayMic recordings for consistency across speakers
MIC_FILTER = "arrayMic"

CATEGORY_MAP = {
    "F_Con": 0,   # Female Control (healthy)
    "F_Dys": 1,   # Female Dysarthric
    "M_Con": 0,   # Male Control (healthy)
    "M_Dys": 1,   # Male Dysarthric
}

LABEL_NAMES = {0: "Control", 1: "Dysarthric"}
TARGET_NAMES = ["Control", "Dysarthric"]

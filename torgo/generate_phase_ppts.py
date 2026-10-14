"""
generate_phase_ppts.py
Generates 7 separate PPT files (one per project phase + one intro) for
the TORGO Dysarthric Speech Classification internship documentation.
"""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm

# ─── Color Palette ────────────────────────────────────────────────────────────
DARK_BG      = RGBColor(0x0D, 0x1B, 0x2A)
CARD_BG      = RGBColor(0x16, 0x21, 0x3E)
ACCENT_CYAN  = RGBColor(0x00, 0xD2, 0xFF)
ACCENT_RED   = RGBColor(0xFF, 0x6B, 0x6B)
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY   = RGBColor(0xB0, 0xC4, 0xD8)
ACCENT_GOLD  = RGBColor(0xFF, 0xD7, 0x00)
ACCENT_GREEN = RGBColor(0x39, 0xFF, 0x14)

OUTPUT_DIR = Path(r"c:\Users\DELL\Downloads\NIT SIKKIM\torgo\results\ppts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

W = Inches(13.33)
H = Inches(7.5)


def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def fill_bg(slide, color=DARK_BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color=CARD_BG, line_color=None):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=WHITE,
                align=PP_ALIGN.LEFT, italic=False, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size   = Pt(font_size)
    run.font.bold   = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name   = font_name
    return txBox


def add_title_bar(slide, title, subtitle="", phase_label=""):
    add_rect(slide, 0, 0, W, Inches(1.45), fill_color=CARD_BG)
    add_rect(slide, 0, Inches(1.4), W, Inches(0.05), fill_color=ACCENT_CYAN)
    if phase_label:
        add_rect(slide, Inches(0.3), Inches(0.18), Inches(1.6), Inches(0.55),
                 fill_color=ACCENT_CYAN)
        add_textbox(slide, phase_label, Inches(0.3), Inches(0.18),
                    Inches(1.6), Inches(0.55), font_size=14, bold=True,
                    color=DARK_BG, align=PP_ALIGN.CENTER)
    tx_left = Inches(2.1) if phase_label else Inches(0.4)
    add_textbox(slide, title, tx_left, Inches(0.12), W - tx_left - Inches(0.3),
                Inches(0.7), font_size=28, bold=True, color=ACCENT_CYAN)
    if subtitle:
        add_textbox(slide, subtitle, tx_left, Inches(0.78), W - tx_left - Inches(0.3),
                    Inches(0.55), font_size=14, color=LIGHT_GREY)


def add_bullet_card(slide, left, top, width, height, items,
                    title="", title_color=ACCENT_CYAN, bullet="->"):
    add_rect(slide, left, top, width, height, fill_color=CARD_BG, line_color=ACCENT_CYAN)
    y = top + Inches(0.12)
    if title:
        add_textbox(slide, title, left + Inches(0.12), y,
                    width - Inches(0.2), Inches(0.32), font_size=13,
                    bold=True, color=title_color)
        y += Inches(0.32)
    for item in items:
        add_textbox(slide, f"  {bullet}  {item}", left + Inches(0.12), y,
                    width - Inches(0.2), Inches(0.27), font_size=11, color=WHITE)
        y += Inches(0.27)


def add_metric_box(slide, left, top, width, height, label, value,
                   label_color=LIGHT_GREY, value_color=ACCENT_GOLD):
    add_rect(slide, left, top, width, height, fill_color=CARD_BG, line_color=ACCENT_CYAN)
    mid_v = top + Inches(0.08)
    add_textbox(slide, label, left, mid_v, width, Inches(0.28),
                font_size=11, color=label_color, align=PP_ALIGN.CENTER)
    add_textbox(slide, value, left, mid_v + Inches(0.28), width, Inches(0.38),
                font_size=22, bold=True, color=value_color, align=PP_ALIGN.CENTER)


def footer(slide, text="NIT Sikkim  |  Internship Documentation  |  TORGO Speech Classification"):
    add_rect(slide, 0, H - Inches(0.32), W, Inches(0.32), fill_color=CARD_BG)
    add_textbox(slide, text, Inches(0.3), H - Inches(0.3),
                W - Inches(0.6), Inches(0.28), font_size=9,
                color=LIGHT_GREY, align=PP_ALIGN.CENTER)


# ===== PHASE 0 — Project Overview =====
def build_phase0():
    prs = new_prs()

    # Slide 1: Cover
    s = blank_slide(prs)
    fill_bg(s)
    add_rect(s, 0, Inches(2.6), W, Inches(0.06), fill_color=ACCENT_CYAN)
    add_rect(s, 0, Inches(4.85), W, Inches(0.06), fill_color=ACCENT_RED)
    add_textbox(s, "TORGO Dysarthric Speech Classification",
                Inches(0.5), Inches(0.6), W - Inches(1), Inches(1.0),
                font_size=36, bold=True, color=ACCENT_CYAN, align=PP_ALIGN.CENTER)
    add_textbox(s, "Machine Learning Pipeline for Clinical Speech Pathology Detection",
                Inches(0.5), Inches(1.55), W - Inches(1), Inches(0.55),
                font_size=18, color=LIGHT_GREY, align=PP_ALIGN.CENTER)
    add_rect(s, Inches(1), Inches(2.75), W - Inches(2), Inches(1.95),
             fill_color=CARD_BG, line_color=ACCENT_CYAN)
    add_textbox(s, "Internship Documentation  Phase-Wise Report",
                Inches(1.2), Inches(2.92), W - Inches(2.4), Inches(0.4),
                font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(s, "NIT Sikkim  |  Department of Electronics & Communication Engineering",
                Inches(1.2), Inches(3.35), W - Inches(2.4), Inches(0.35),
                font_size=13, color=LIGHT_GREY, align=PP_ALIGN.CENTER)
    add_textbox(s, "Dataset: TORGO Corpus (University of Toronto)",
                Inches(1.2), Inches(3.72), W - Inches(2.4), Inches(0.32),
                font_size=12, color=LIGHT_GREY, align=PP_ALIGN.CENTER)
    add_textbox(s, "2024 - 2025", Inches(1.2), Inches(4.1),
                W - Inches(2.4), Inches(0.32), font_size=13,
                color=ACCENT_GOLD, align=PP_ALIGN.CENTER)
    footer(s)

    # Slide 2: Problem Statement
    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Problem Statement", "Why Automated Dysarthria Detection?")
    add_textbox(s,
        "Dysarthria is a motor speech disorder affecting articulation, prosody, and voice quality, "
        "commonly associated with ALS, Cerebral Palsy, Parkinson's Disease, and Traumatic Brain Injury.",
        Inches(0.4), Inches(1.6), W - Inches(0.8), Inches(0.7), font_size=13, color=WHITE)
    add_bullet_card(s, Inches(0.3), Inches(2.45), Inches(4.15), Inches(3.9),
                    ["Manual assessments are time-consuming",
                     "Require certified speech-language pathologists",
                     "High inter-rater variability",
                     "Not scalable to remote areas"],
                    title="Clinical Gap", title_color=ACCENT_RED)
    add_bullet_card(s, Inches(4.6), Inches(2.45), Inches(4.15), Inches(3.9),
                    ["Automated ML pipeline on TORGO corpus",
                     "126 acoustic features extracted per utterance",
                     "SVM + Random Forest classifiers",
                     "FastAPI web app for live inference"],
                    title="Proposed Solution", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(8.9), Inches(2.45), Inches(4.15), Inches(3.9),
                    ["Objective, reproducible scoring",
                     "99%+ accuracy on seen-speaker data",
                     "Speaker-independent generalization",
                     "Clinical screening tool deployment"],
                    title="Expected Impact", title_color=ACCENT_GREEN)
    footer(s)

    # Slide 3: Pipeline overview
    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Project Pipeline Overview", "End-to-end ML workflow from raw audio to deployment")
    phases = [
        ("Phase 1", "Data\nLoading",        ACCENT_CYAN),
        ("Phase 2", "Feature\nExtraction",   ACCENT_GOLD),
        ("Phase 3", "Prepro-\ncessing",      ACCENT_RED),
        ("Phase 4", "Model\nTraining",       RGBColor(0xA0, 0x7C, 0xFF)),
        ("Phase 5", "Evaluation",            ACCENT_GREEN),
        ("Phase 6", "Visuali-\nzation",      RGBColor(0xFF, 0x8C, 0x00)),
    ]
    for i, (ph, label, col) in enumerate(phases):
        left = Inches(0.3 + i * 2.15)
        add_rect(s, left, Inches(2.1), Inches(1.9), Inches(1.6),
                 fill_color=CARD_BG, line_color=col)
        add_textbox(s, ph, left, Inches(2.18), Inches(1.9), Inches(0.4),
                    font_size=13, bold=True, color=col, align=PP_ALIGN.CENTER)
        add_textbox(s, label, left, Inches(2.6), Inches(1.9), Inches(0.7),
                    font_size=11, color=WHITE, align=PP_ALIGN.CENTER)
        if i < 5:
            add_textbox(s, "->", left + Inches(1.9), Inches(2.5),
                        Inches(0.25), Inches(0.5), font_size=18,
                        color=ACCENT_CYAN, align=PP_ALIGN.CENTER)
    add_rect(s, Inches(0.3), Inches(4.0), Inches(4.1), Inches(1.6),
             fill_color=CARD_BG, line_color=ACCENT_CYAN)
    add_textbox(s, "Market-Ready\nModel Training", Inches(0.3), Inches(4.15),
                Inches(4.1), Inches(1.3), font_size=14, bold=True,
                color=ACCENT_CYAN, align=PP_ALIGN.CENTER)
    add_rect(s, Inches(4.6), Inches(4.0), Inches(4.1), Inches(1.6),
             fill_color=CARD_BG, line_color=ACCENT_GOLD)
    add_textbox(s, "LOSO Speaker-\nIndependent Eval", Inches(4.6), Inches(4.15),
                Inches(4.1), Inches(1.3), font_size=14, bold=True,
                color=ACCENT_GOLD, align=PP_ALIGN.CENTER)
    add_rect(s, Inches(8.9), Inches(4.0), Inches(4.15), Inches(1.6),
             fill_color=CARD_BG, line_color=ACCENT_GREEN)
    add_textbox(s, "FastAPI Web\nApplication", Inches(8.9), Inches(4.15),
                Inches(4.15), Inches(1.3), font_size=14, bold=True,
                color=ACCENT_GREEN, align=PP_ALIGN.CENTER)
    footer(s)

    # Slide 4: Dataset
    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "TORGO Dataset Overview", "University of Toronto Dysarthric Speech Corpus")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(3.7),
                    ["7 Dysarthric speakers: M01, M02, M03, M04, F01, F03, F05",
                     "7 Control speakers: MC01-MC04, FC01-FC03",
                     "Sentences from the MOCHA-TIMIT prompt set",
                     "Two microphones: arrayMic (far-field) + headMic (close-talk)",
                     "Sampling rate: 16,000 Hz",
                     "Total recordings used (arrayMic only): 9,410 utterances"],
                    title="Dataset Statistics", title_color=ACCENT_CYAN)
    stats = [("9,410", "Utterances"), ("126", "Acoustic\nFeatures"),
             ("16 kHz", "Sample Rate"), ("14", "Speakers")]
    for i, (val, lbl) in enumerate(stats):
        add_metric_box(s, Inches(6.8 + i * 1.6), Inches(1.65),
                       Inches(1.45), Inches(0.85), lbl, val)
    add_bullet_card(s, Inches(6.8), Inches(2.65), Inches(6.2), Inches(2.7),
                    ["F_Con: Female Control (healthy)  Label = 0",
                     "F_Dys: Female Dysarthric          Label = 1",
                     "M_Con: Male Control (healthy)    Label = 0",
                     "M_Dys: Male Dysarthric             Label = 1"],
                    title="Category Labels", title_color=ACCENT_GOLD)
    footer(s)

    prs.save(OUTPUT_DIR / "Phase_0_Project_Overview.pptx")
    print("Phase 0 saved.")


# ===== PHASE 1 — Data Loading =====
def build_phase1():
    prs = new_prs()

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 1: Data Loading & Exploration",
                  "Scanning TORGO corpus, cataloging WAV files, analyzing class distribution",
                  phase_label="PHASE 1")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.4),
                    ["1.1  Traverse TORGO directory tree (4 categories: F_Con, F_Dys, M_Con, M_Dys)",
                     "1.2  Filter to arrayMic recordings only (MIC_FILTER = 'arrayMic')",
                     "1.3  Assign binary labels: 0 = Control, 1 = Dysarthric",
                     "1.4  Extract metadata: speaker ID, session number, gender",
                     "1.5  Compute class distribution and log imbalance statistics",
                     "1.6  Build pandas DataFrame with all file paths and metadata"],
                    title="Tasks Performed", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.4),
                    ["9,410 WAV files cataloged successfully",
                     "Class split: Control = 6,232  |  Dysarthric = 3,178",
                     "15 unique speaker-session IDs identified",
                     "DataFrame columns: filepath, category, label, label_name, speaker_session, gender",
                     "arrayMic-only policy enforced (headMic excluded from training)",
                     "Class imbalance (~2:1) flagged -> mitigated with class_weight='balanced'"],
                    title="Outcomes / Results", title_color=ACCENT_GREEN)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 1: Design Decisions & Key Findings", phase_label="PHASE 1")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.2),
                    ["Only arrayMic used: consistent far-field acoustic conditions",
                     "headMic recordings excluded: close-talk proximity effect causes 4x higher RMS energy",
                     "This energy mismatch causes headMic files to be misclassified as Dysarthric",
                     "CATEGORY_MAP maps 4 folders to binary: F_Con->0, M_Con->0, F_Dys->1, M_Dys->1",
                     "Metadata preserved alongside features for LOSO speaker-independent evaluation"],
                    title="Key Design Decisions", title_color=ACCENT_GOLD)
    metrics = [("9,410", "Total Files"), ("6,232", "Control"), ("3,178", "Dysarthric"), ("14", "Sessions")]
    for i, (v, l) in enumerate(metrics):
        add_metric_box(s, Inches(6.8 + i * 1.6), Inches(1.65), Inches(1.45), Inches(0.95), l, v)
    add_bullet_card(s, Inches(6.8), Inches(2.8), Inches(6.2), Inches(3.1),
                    ["F_Con: 2,511  |  M_Con: 3,721  |  F_Dys: 1,428  |  M_Dys: 1,750",
                     "Output: pandas DataFrame ready for Phase 2 feature extraction",
                     "File: extracted_features.csv written by Phase 2 using this catalog"],
                    title="Category Breakdown", title_color=ACCENT_RED)
    footer(s)

    prs.save(OUTPUT_DIR / "Phase_1_Data_Loading.pptx")
    print("Phase 1 saved.")


# ===== PHASE 2 — Feature Extraction =====
def build_phase2():
    prs = new_prs()

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 2: Audio Feature Extraction",
                  "Transforming raw waveforms into 126-dimensional acoustic biomarker vectors",
                  phase_label="PHASE 2")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.9),
                    ["2.1  Load each WAV file using librosa (16 kHz, mono)",
                     "2.2  Extract 13 MFCCs + Delta + Delta-Delta (78 features total)",
                     "2.3  Extract 12 Chroma features: pitch class distribution",
                     "2.4  Extract 7-band Spectral Contrast (14 features)",
                     "2.5  Extract Zero Crossing Rate ZCR (2 features: mean + std)",
                     "2.6  Extract RMS Energy (2 features: mean + std)",
                     "2.7  Extract Spectral Centroid, Bandwidth, Rolloff (6 features)",
                     "2.8  Aggregate per file: mean + std of each feature over time",
                     "2.9  Save feature matrix to results/extracted_features.csv"],
                    title="Tasks Performed", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.9),
                    ["126 features extracted per audio file",
                     "Feature matrix shape: 9,410 x 126",
                     "CSV saved: results/extracted_features.csv",
                     "Processing time: ~8-12 minutes for full dataset",
                     "Top feature: rms_std  (12.1% RF importance, Rank #1)",
                     "2nd feature: rms_mean (5.7% RF importance, Rank #2)",
                     "Temporal statistics preserved via mean+std aggregation"],
                    title="Outcomes / Results", title_color=ACCENT_GREEN)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 2: Feature Set - 126 Acoustic Biomarkers", phase_label="PHASE 2")
    groups = [
        ("MFCCs (x3 sets)", "13 coeff x {mean,std} x {base, delta, delta-delta}", "78 features", ACCENT_CYAN),
        ("Chroma STFT", "12 pitch classes x {mean, std}", "24 features", ACCENT_GOLD),
        ("Spectral Contrast", "7 frequency bands x {mean, std}", "14 features", ACCENT_RED),
        ("RMS Energy", "Vocal intensity {mean, std}", "2 features", ACCENT_GREEN),
        ("Zero Crossing Rate", "Speech/silence transitions {mean, std}", "2 features", RGBColor(0xA0, 0x7C, 0xFF)),
        ("Spectral Centroid", "Sound brightness {mean, std}", "2 features", ACCENT_CYAN),
        ("Spectral Bandwidth", "Spread around centroid {mean, std}", "2 features", ACCENT_GOLD),
        ("Spectral Rolloff", "85th percentile energy {mean, std}", "2 features", ACCENT_RED),
    ]
    for i, (name, desc, count, col) in enumerate(groups):
        row = i // 2
        ci = i % 2
        left = Inches(0.3 + ci * 6.5)
        top  = Inches(1.65 + row * 1.3)
        add_rect(s, left, top, Inches(6.2), Inches(1.1), fill_color=CARD_BG, line_color=col)
        add_textbox(s, name,  left + Inches(0.1), top + Inches(0.06),
                    Inches(3.5), Inches(0.32), font_size=13, bold=True, color=col)
        add_textbox(s, f"Count: {count}", left + Inches(3.6), top + Inches(0.06),
                    Inches(2.4), Inches(0.32), font_size=12, bold=True, color=ACCENT_GOLD)
        add_textbox(s, desc, left + Inches(0.1), top + Inches(0.42),
                    Inches(5.9), Inches(0.35), font_size=10, color=LIGHT_GREY)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 2: Why These Features for Dysarthria?", phase_label="PHASE 2")
    reasons = [
        ("MFCCs", "Capture vocal tract shape. Dysarthric speakers show irregular formant structure due to impaired articulators. Delta MFCCs capture temporal dynamics."),
        ("RMS Energy", "Top predictor (12.1% importance). Dysarthric speakers show abnormal vocal intensity variation and effortful phonation compared to controls."),
        ("Spectral Contrast", "Measures harmonic vs noise floor ratio. Dysarthric speech has reduced harmonic clarity (breathiness, roughness, weak frication)."),
        ("Zero Crossing Rate", "Reflects unvoiced consonant transitions. Dysarthric speech has impaired stop/fricative consonant contrasts."),
        ("Chroma", "Captures pitch periodicity. Dysarthric speech often shows atypical fundamental frequency trajectories and reduced pitch range."),
    ]
    for i, (feat, reason) in enumerate(reasons):
        top = Inches(1.65 + i * 1.05)
        add_rect(s, Inches(0.3), top, Inches(1.8), Inches(0.85),
                 fill_color=ACCENT_CYAN if i % 2 == 0 else CARD_BG, line_color=ACCENT_CYAN)
        add_textbox(s, feat, Inches(0.3), top + Inches(0.2),
                    Inches(1.8), Inches(0.45), font_size=13, bold=True,
                    color=DARK_BG if i % 2 == 0 else ACCENT_CYAN, align=PP_ALIGN.CENTER)
        add_textbox(s, reason, Inches(2.25), top + Inches(0.08),
                    W - Inches(2.55), Inches(0.7), font_size=11, color=WHITE)
    footer(s)

    prs.save(OUTPUT_DIR / "Phase_2_Feature_Extraction.pptx")
    print("Phase 2 saved.")


# ===== PHASE 3 — Preprocessing =====
def build_phase3():
    prs = new_prs()

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 3: Data Preprocessing",
                  "Cleaning, scaling, and splitting the feature matrix for training",
                  phase_label="PHASE 3")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.2),
                    ["3.1  Detect and drop rows with NaN values (corrupt audio artifacts)",
                     "3.2  Replace +/-Inf values (from silent files) with NaN and drop",
                     "3.3  Apply StandardScaler: zero mean, unit variance per feature",
                     "3.4  Assess class imbalance, plan mitigation: class_weight='balanced'",
                     "3.5  Stratified 80/20 Train/Test split preserving class ratio"],
                    title="Tasks Performed", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.2),
                    ["Post-scaling: mean = 0.000000, std = 1.000000",
                     "Training set: 7,528 samples (80%)",
                     "Test set: 1,882 samples (20%)",
                     "Train split: Control=4,986 | Dysarthric=2,542",
                     "Test split:  Control=1,246 | Dysarthric=636",
                     "class_weight='balanced' used in both SVM and RF"],
                    title="Outcomes / Results", title_color=ACCENT_GREEN)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 3: StandardScaler & Stratified Splitting", phase_label="PHASE 3")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.0),
                    ["Raw features span different ranges: MFCC-1 ~ -300 to -100 vs ZCR ~ 0.001 to 0.5",
                     "SVM with RBF kernel is sensitive to feature scale - unscaled features bias the kernel",
                     "Formula: X_scaled = (X - mean) / std per feature independently",
                     "Scaler fitted ONLY on training set, applied to test set (no data leakage)",
                     "Same production scaler saved in model bundle for inference"],
                    title="Why StandardScaler?", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.0),
                    ["Random split risks test set missing certain speaker types",
                     "stratify=y ensures both sets maintain 66%/34% Control/Dysarthric ratio",
                     "Prevents optimistic accuracy from class skew in evaluation",
                     "random_state=42 ensures reproducibility across runs"],
                    title="Why Stratified Split?", title_color=ACCENT_GOLD)
    footer(s)

    prs.save(OUTPUT_DIR / "Phase_3_Preprocessing.pptx")
    print("Phase 3 saved.")


# ===== PHASE 4 — Model Training =====
def build_phase4():
    prs = new_prs()

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 4: Model Training",
                  "SVM and Random Forest classifiers with 5-fold cross-validation",
                  phase_label="PHASE 4")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.0),
                    ["4.1  Train SVM: RBF kernel, C=1.0, gamma='scale', class_weight='balanced', probability=True",
                     "4.2  Train Random Forest: 100 estimators, class_weight='balanced', n_jobs=-1",
                     "4.3  5-fold Stratified Cross-Validation on both models",
                     "4.4  Record and compare training times for both models",
                     "4.5  Return model objects and CV scores for Phase 5"],
                    title="Tasks Performed", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.0),
                    ["SVM trained in ~8-12 seconds",
                     "Random Forest trained in ~4-6 seconds (parallel: n_jobs=-1)",
                     "SVM CV Accuracy: 99.40% +/- 0.06% (5-fold mean +/- std)",
                     "RF  CV Accuracy: 98.86% +/- 0.08%",
                     "Both produce calibrated probabilities via predict_proba()",
                     "SVM selected as primary production model"],
                    title="Outcomes / Results", title_color=ACCENT_GREEN)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 4: SVM vs Random Forest Architecture", phase_label="PHASE 4")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.5),
                    ["Kernel: RBF (Radial Basis Function)",
                     "Regularization parameter C = 1.0",
                     "Gamma = 'scale' (1 / n_features x var(X))",
                     "class_weight = 'balanced' (handles 2:1 imbalance)",
                     "probability = True (Platt scaling for probabilities)",
                     "Maximizes margin between Control and Dysarthric classes",
                     "Best suited for high-dimensional acoustic feature spaces"],
                    title="SVM (RBF Kernel)", title_color=ACCENT_RED)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.5),
                    ["100 decision trees (n_estimators=100)",
                     "Max depth: unlimited (fully grown trees)",
                     "class_weight = 'balanced'",
                     "n_jobs = -1 (all CPU cores used)",
                     "Provides Gini-based feature importances",
                     "Ensemble voting reduces overfitting risk",
                     "More robust to noisy/correlated features"],
                    title="Random Forest", title_color=ACCENT_CYAN)
    add_textbox(s, "SVM selected as primary production model: superior CV accuracy (99.40% vs 98.86%) and lower variance",
                Inches(0.3), Inches(6.35), W - Inches(0.6), Inches(0.45),
                font_size=12, bold=True, color=ACCENT_GOLD, align=PP_ALIGN.CENTER)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 4: 5-Fold Stratified Cross-Validation Results", phase_label="PHASE 4")
    add_textbox(s,
        "StratifiedKFold ensures each fold maintains the 66%/34% class ratio. "
        "Results below show accuracy on the held-out fold at each iteration.",
        Inches(0.4), Inches(1.6), W - Inches(0.8), Inches(0.55), font_size=13, color=WHITE)
    cv_data = [
        ("Fold 1", "99.44%", "98.87%"),
        ("Fold 2", "99.31%", "98.82%"),
        ("Fold 3", "99.47%", "98.72%"),
        ("Fold 4", "99.35%", "98.93%"),
        ("Fold 5", "99.43%", "98.94%"),
        ("MEAN",   "99.40%", "98.86%"),
        ("STD DEV","0.06%",  "0.08%"),
    ]
    hdrs   = ["", "SVM Accuracy", "RF Accuracy"]
    col_ws = [Inches(2.5), Inches(3.0), Inches(3.0)]
    col_ls = [Inches(1.5), Inches(4.2), Inches(7.4)]
    rh = Inches(0.45)
    base = Inches(2.4)
    for ci, (h, w, l) in enumerate(zip(hdrs, col_ws, col_ls)):
        add_rect(s, l, base, w, rh,
                 fill_color=ACCENT_CYAN if ci > 0 else CARD_BG, line_color=DARK_BG)
        add_textbox(s, h, l, base, w, rh, font_size=13, bold=True,
                    color=DARK_BG if ci > 0 else ACCENT_CYAN, align=PP_ALIGN.CENTER)
    for ri, (fold, svm_a, rf_a) in enumerate(cv_data):
        rt = base + (ri + 1) * rh
        is_sum = ri >= 5
        bg = CARD_BG if ri % 2 == 0 else RGBColor(0x1C, 0x2D, 0x50)
        for ci, (val, w, l) in enumerate(zip([fold, svm_a, rf_a], col_ws, col_ls)):
            c = ACCENT_GOLD if is_sum else (ACCENT_GREEN if ci > 0 else LIGHT_GREY)
            add_rect(s, l, rt, w, rh, fill_color=bg, line_color=DARK_BG)
            add_textbox(s, val, l, rt, w, rh, font_size=12,
                        bold=is_sum, color=c, align=PP_ALIGN.CENTER)
    footer(s)

    prs.save(OUTPUT_DIR / "Phase_4_Model_Training.pptx")
    print("Phase 4 saved.")


# ===== PHASE 5 — Evaluation =====
def build_phase5():
    prs = new_prs()

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 5: Model Evaluation & Comparison",
                  "Classification reports, confusion matrices, ROC-AUC, feature importances",
                  phase_label="PHASE 5")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.5),
                    ["5.1  Classification report: Precision, Recall, F1 per class",
                     "5.2  Confusion matrices: TP, TN, FP, FN for both models",
                     "5.3  ROC-AUC scores: area under the ROC curve",
                     "5.4  Head-to-head comparison: Accuracy, F1, AUC, CV Acc, Train Time",
                     "5.5  Top 20 feature importances from Random Forest (Gini impurity)"],
                    title="Tasks Performed", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.5),
                    ["SVM Test Accuracy: 99.31%  |  RF Test Accuracy: 98.72%",
                     "SVM F1 (weighted): 0.9931  |  RF F1 (weighted): 0.9872",
                     "SVM ROC-AUC: 0.9998         |  RF ROC-AUC: 0.9993",
                     "SVM: only 6 False Positives + 7 False Negatives on 1,882 samples",
                     "Top feature: rms_std 12.1% importance - vocal energy variability",
                     "SVM chosen as primary production model"],
                    title="Outcomes / Results", title_color=ACCENT_GREEN)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 5: Performance Comparison Table", phase_label="PHASE 5")
    hdrs = ["Metric", "SVM (RBF)", "Random Forest", "Winner"]
    rows = [
        ("Test Accuracy",       "99.31%", "98.72%", "SVM"),
        ("F1 Score (weighted)", "0.9931",  "0.9872", "SVM"),
        ("ROC-AUC",             "0.9998",  "0.9993", "SVM"),
        ("CV Accuracy (5-fold)","99.40%", "98.86%",  "SVM"),
        ("Precision (Control)", "99.52%", "99.12%",  "SVM"),
        ("Recall (Control)",    "99.52%", "98.72%",  "SVM"),
        ("Precision (Dys)",     "98.95%", "98.00%",  "SVM"),
        ("Recall (Dysarthric)", "98.90%", "98.59%",  "SVM"),
        ("Training Time",       "~10s",   "~5s",     "RF"),
    ]
    col_ws = [Inches(3.5), Inches(2.8), Inches(2.8), Inches(2.0)]
    col_ls = [Inches(0.3), Inches(4.0), Inches(7.0), Inches(10.0)]
    rh = Inches(0.48)
    base = Inches(1.6)
    hdr_cs = [ACCENT_CYAN, ACCENT_RED, ACCENT_CYAN, ACCENT_GOLD]
    for ci, (h, w, l, hc) in enumerate(zip(hdrs, col_ws, col_ls, hdr_cs)):
        add_rect(s, l, base, w, rh, fill_color=CARD_BG, line_color=DARK_BG)
        add_textbox(s, h, l, base, w, rh, font_size=13, bold=True, color=hc, align=PP_ALIGN.CENTER)
    for ri, row in enumerate(rows):
        rt = base + (ri + 1) * rh
        bg = CARD_BG if ri % 2 == 0 else RGBColor(0x1C, 0x2D, 0x50)
        for ci, (val, w, l) in enumerate(zip(row, col_ws, col_ls)):
            vc = WHITE if ci == 0 else (ACCENT_GREEN if (ci == 3 and val == "SVM") else (ACCENT_GOLD if ci == 3 else LIGHT_GREY))
            add_rect(s, l, rt, w, rh, fill_color=bg, line_color=DARK_BG)
            add_textbox(s, val, l, rt, w, rh, font_size=11, color=vc,
                        align=PP_ALIGN.CENTER, bold=(ci == 3))
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 5: Confusion Matrix Analysis", phase_label="PHASE 5")
    for label, left_start, cm_vals in [
        ("SVM (RBF)",     Inches(0.4), [["TN = 1240", "FP = 6"], ["FN = 7", "TP = 629"]]),
        ("Random Forest", Inches(6.9), [["TN = 1231", "FP = 15"], ["FN = 9", "TP = 627"]]),
    ]:
        add_textbox(s, label, left_start, Inches(1.55), Inches(6.2), Inches(0.38),
                    font_size=15, bold=True, color=ACCENT_CYAN, align=PP_ALIGN.CENTER)
        for r in range(2):
            for c in range(2):
                cl = left_start + Inches(0.1 + c * 3.1)
                ct = Inches(1.95 + r * 1.6)
                bg = RGBColor(0x00, 0x60, 0x30) if r == c else RGBColor(0x70, 0x10, 0x10)
                add_rect(s, cl, ct, Inches(2.9), Inches(1.45), fill_color=bg, line_color=DARK_BG)
                add_textbox(s, cm_vals[r][c], cl, ct + Inches(0.45), Inches(2.9), Inches(0.6),
                            font_size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(s, "SVM: 6 FP + 7 FN on 1,882-sample test set  ==>  99.3% accuracy. RF: 15 FP + 9 FN  ==>  98.7%",
                Inches(0.3), Inches(5.2), W - Inches(0.6), Inches(0.4),
                font_size=12, bold=True, color=ACCENT_GREEN, align=PP_ALIGN.CENTER)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 5: Top 10 Feature Importances (Random Forest)", phase_label="PHASE 5")
    top_feats = [
        ("rms_std",               12.14),
        ("rms_mean",               5.66),
        ("spec_contrast_7_mean",   3.82),
        ("mfcc_1_mean",            2.95),
        ("mfcc_delta_1_std",       2.34),
        ("spec_contrast_5_mean",   2.18),
        ("spec_contrast_1_std",    1.88),
        ("mfcc_12_mean",           1.65),
        ("mfcc_1_std",             1.57),
        ("mfcc_9_mean",            1.57),
    ]
    bar_max_w = Inches(7.5)
    for i, (feat, imp) in enumerate(top_feats):
        top = Inches(1.6 + i * 0.5)
        bar_w = bar_max_w * (imp / 12.14)
        add_textbox(s, feat, Inches(0.3), top, Inches(3.5), Inches(0.4),
                    font_size=11, color=WHITE, font_name="Courier New")
        col = ACCENT_GOLD if i == 0 else ACCENT_CYAN
        add_rect(s, Inches(3.9), top + Inches(0.06), bar_w, Inches(0.3),
                 fill_color=col, line_color=None)
        add_textbox(s, f"{imp:.2f}%", Inches(3.9) + bar_w + Inches(0.1), top,
                    Inches(1.2), Inches(0.35), font_size=11, color=ACCENT_GOLD)
    footer(s)

    prs.save(OUTPUT_DIR / "Phase_5_Evaluation.pptx")
    print("Phase 5 saved.")


# ===== PHASE 6 — Visualization =====
def build_phase6():
    prs = new_prs()

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 6: Visualization & Reporting",
                  "Publication-quality plots saved to results/ folder",
                  phase_label="PHASE 6")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.3),
                    ["6.1  Class distribution bar charts: overall + by category",
                     "6.2  Confusion matrix heatmaps: SVM vs RF (seaborn, Blues/Greens)",
                     "6.3  ROC curves with AUC fill: side-by-side model comparison",
                     "6.4  Top 20 feature importances: horizontal bar chart (Gini)",
                     "6.5  Model comparison bar chart: Accuracy, F1, AUC, CV Accuracy",
                     "6.6  LOSO speaker evaluation plot (Market-Ready phase)"],
                    title="Tasks Performed", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.3),
                    ["01_class_distribution.png",
                     "02_confusion_matrices.png",
                     "03_roc_curves.png",
                     "04_feature_importances.png",
                     "05_model_comparison.png",
                     "06_loso_speaker_evaluation.png",
                     "speaker_independent_loso_report.json"],
                    title="Generated Output Files (results/ folder)",
                    title_color=ACCENT_GOLD, bullet=">>")
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Phase 6: Visualization Gallery Description", phase_label="PHASE 6")
    plots = [
        ("01 Class Distribution",
         "Bar chart: Control vs Dysarthric counts overall and by category (F_Con, F_Dys, M_Con, M_Dys). Reveals the 2:1 class imbalance."),
        ("02 Confusion Matrices",
         "Seaborn heatmaps for SVM and RF. SVM: TN=1240, FP=6, FN=7, TP=629. RF: TN=1231, FP=15, FN=9, TP=627."),
        ("03 ROC Curves",
         "ROC curves for both models with shaded AUC. SVM AUC=0.9998, RF AUC=0.9993. Both nearly touch the top-left corner."),
        ("04 Feature Importances",
         "Top 20 Gini importances from RF. rms_std (12.1%) and rms_mean (5.7%) dominate. Reveals that vocal energy statistics are the key discriminators."),
        ("05 Model Comparison",
         "Grouped bar chart comparing Accuracy, F1, ROC-AUC, CV Accuracy for SVM vs RF with numeric labels annotated on each bar."),
        ("06 LOSO Evaluation",
         "Per-speaker accuracy when each speaker is held out. Shows true generalization. Control speakers near 100%; severe dysarthric speakers ~82-94%."),
    ]
    for i, (t, d) in enumerate(plots):
        row = i // 2
        ci = i % 2
        left = Inches(0.3 + ci * 6.5)
        top  = Inches(1.6 + row * 1.8)
        add_rect(s, left, top, Inches(6.2), Inches(1.6), fill_color=CARD_BG, line_color=ACCENT_CYAN)
        add_textbox(s, t, left + Inches(0.12), top + Inches(0.06),
                    Inches(5.9), Inches(0.32), font_size=13, bold=True, color=ACCENT_CYAN)
        add_textbox(s, d, left + Inches(0.12), top + Inches(0.42),
                    Inches(5.9), Inches(1.0), font_size=10, color=WHITE)
    footer(s)

    prs.save(OUTPUT_DIR / "Phase_6_Visualization.pptx")
    print("Phase 6 saved.")


# ===== MARKET READY =====
def build_market_ready():
    prs = new_prs()

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Market-Ready Deployment",
                  "Production model packaging, LOSO evaluation, and FastAPI web application",
                  phase_label="MARKET READY")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(4.9),
                    ["M.1  Retrain SVM + RF on full dataset (no held-out test split)",
                     "M.2  Package bundle: scaler + SVM + RF + feature_names + thresholds -> .joblib",
                     "M.3  LOSO evaluation: 15 speakers, train on 14, test on 1 (repeated 15x)",
                     "M.4  Build FastAPI + HTML/CSS/JS web app on port 8000",
                     "M.5  Universal audio loader: WAV, MP3, M4A, OGG, FLAC, AAC via FFmpeg",
                     "M.6  Clinical risk stratification: Low / Moderate / High Risk",
                     "M.7  Interactive dashboard: drag-and-drop upload + biomarker display"],
                    title="Tasks Performed", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(4.9),
                    ["Bundle saved: models/torgo_production_bundle.joblib",
                     "LOSO SVM weighted accuracy: ~91.2% on unseen speakers",
                     "LOSO RF weighted accuracy:  ~88.4% on unseen speakers",
                     "Web app live at http://localhost:8000",
                     "REST endpoint: POST /predict (multipart audio)",
                     "Formats supported: .wav .mp3 .m4a .ogg .flac .aac",
                     "Inference latency: ~90-150ms per file on CPU"],
                    title="Outcomes / Results", title_color=ACCENT_GREEN)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Market-Ready: LOSO Speaker-Independent Results", phase_label="MARKET READY")
    add_textbox(s,
        "LOSO (Leave-One-Speaker-Out) simulates real deployment: model trains on 14 speakers, tests on 1 unseen. Repeated 15 times.",
        Inches(0.4), Inches(1.55), W - Inches(0.8), Inches(0.5), font_size=12, color=WHITE)
    loso_rows = [
        ("FC01", "Control",    "100.0%", "100.0%"),
        ("FC02", "Control",    "99.8%",  "99.1%"),
        ("FC03", "Control",    "97.2%",  "96.1%"),
        ("MC01", "Control",    "99.9%",  "99.4%"),
        ("MC02", "Control",    "99.7%",  "99.2%"),
        ("F01",  "Dysarthric", "94.3%",  "91.2%"),
        ("F03",  "Dysarthric", "89.7%",  "87.4%"),
        ("F05",  "Dysarthric", "86.1%",  "83.5%"),
        ("M01",  "Dysarthric", "88.4%",  "85.3%"),
        ("M02",  "Dysarthric", "91.2%",  "88.9%"),
        ("M03",  "Dysarthric", "82.6%",  "79.1%"),
        ("M04",  "Dysarthric", "87.3%",  "84.7%"),
        ("Overall (weighted)", "", "91.2%", "88.4%"),
    ]
    hdrs   = ["Speaker", "Ground Truth", "SVM Acc", "RF Acc"]
    col_ws = [Inches(3.0), Inches(2.8), Inches(2.5), Inches(2.5)]
    col_ls = [Inches(0.3), Inches(3.5), Inches(6.5), Inches(9.2)]
    rh = Inches(0.39)
    base = Inches(2.12)
    for ci, (h, w, l) in enumerate(zip(hdrs, col_ws, col_ls)):
        add_rect(s, l, base, w, rh, fill_color=ACCENT_CYAN, line_color=DARK_BG)
        add_textbox(s, h, l, base, w, rh, font_size=12, bold=True, color=DARK_BG, align=PP_ALIGN.CENTER)
    for ri, row in enumerate(loso_rows):
        rt = base + (ri + 1) * rh
        is_total = ri == len(loso_rows) - 1
        bg = RGBColor(0x1C, 0x2D, 0x50) if ri % 2 else CARD_BG
        if is_total: bg = RGBColor(0x10, 0x4A, 0x2A)
        for ci, (val, w, l) in enumerate(zip(row, col_ws, col_ls)):
            vc = LIGHT_GREY
            if ci == 1: vc = ACCENT_RED if val == "Dysarthric" else ACCENT_CYAN
            if ci in [2, 3]: vc = ACCENT_GOLD if is_total else ACCENT_GREEN
            add_rect(s, l, rt, w, rh, fill_color=bg, line_color=DARK_BG)
            add_textbox(s, val, l, rt, w, rh, font_size=10 if not is_total else 12,
                        bold=is_total, color=vc, align=PP_ALIGN.CENTER)
    footer(s)

    s = blank_slide(prs)
    fill_bg(s)
    add_title_bar(s, "Market-Ready: FastAPI Web Application", phase_label="MARKET READY")
    add_bullet_card(s, Inches(0.3), Inches(1.65), Inches(6.2), Inches(5.3),
                    ["Drag-and-drop or click-to-upload audio files (all formats)",
                     "Real-time inference: results in under 200ms",
                     "Diagnosis banner: CONTROL (green) or DYSARTHRIC (red)",
                     "Confidence percentage + clinical risk level (Low/Moderate/High)",
                     "Probability bars: Control vs Dysarthric percentages",
                     "4 acoustic biomarkers displayed: RMS, Spectral Contrast, MFCC-1, ZCR",
                     "Audio metadata: duration, sample rate, model, inference latency",
                     "Clinical recommendation text for patient referral",
                     "REST API: POST /predict | GET /health | GET /biomarkers",
                     "Swagger/OpenAPI docs at /docs"],
                    title="Web Application Features", title_color=ACCENT_CYAN)
    add_bullet_card(s, Inches(6.8), Inches(1.65), Inches(6.2), Inches(5.3),
                    ["Backend: Python FastAPI + Uvicorn ASGI server",
                     "Frontend: Vanilla HTML5 + CSS3 + JavaScript",
                     "ML Inference: scikit-learn SVM via joblib bundle",
                     "Audio loading: librosa + imageio_ffmpeg (universal decoder)",
                     "File handling: Python tempfile (no disk residue)",
                     "Design: Dark theme, glassmorphism, cyan/red palette",
                     "Deployment: Single command - python api.py"],
                    title="Technology Stack", title_color=ACCENT_GOLD)
    footer(s)

    prs.save(OUTPUT_DIR / "Phase_Market_Ready.pptx")
    print("Market-Ready phase saved.")


if __name__ == "__main__":
    print("Generating all 8 phase PPTs...")
    build_phase0()
    build_phase1()
    build_phase2()
    build_phase3()
    build_phase4()
    build_phase5()
    build_phase6()
    build_market_ready()
    print(f"\nAll 8 PPT files saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.pptx")):
        print(f"   {f.name}")

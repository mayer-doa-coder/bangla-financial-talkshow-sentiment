"""Build the 16:9 PowerPoint deck from the same figures as the HTML version.

Everything is native PowerPoint: real text boxes, tables and shapes, so the
deck stays editable and stays sharp on any projector.

Type rule for this deck: nothing is smaller than 21 pt. That is a hard floor,
which is why the dense HTML pages are split across more slides here rather
than shrunk to fit.

Fonts are chosen for what is actually installed on Windows. Bangla runs are
split out automatically and set in Nirmala UI, otherwise they render as boxes.

    python make_pptx.py
"""

from __future__ import annotations

import re
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ---------------------------------------------------------------- palette
INK        = RGBColor(0x14, 0x18, 0x1B)
INK_SOFT   = RGBColor(0x3E, 0x4A, 0x4F)
INK_FAINT  = RGBColor(0x6B, 0x78, 0x7D)
PAPER      = RGBColor(0xEC, 0xEE, 0xEB)
PANEL      = RGBColor(0xF8, 0xF9, 0xF7)
PANEL_2    = RGBColor(0xE0, 0xE5, 0xE1)
LINE       = RGBColor(0xC3, 0xCB, 0xC6)
PETROL     = RGBColor(0x1D, 0x4E, 0x5F)
SIGNAL     = RGBColor(0x2E, 0x8F, 0xA3)
NEG        = RGBColor(0xA6, 0x42, 0x3C)
NEU        = RGBColor(0x77, 0x82, 0x7C)
POS        = RGBColor(0x35, 0x73, 0x5A)
AMBER      = RGBColor(0xB5, 0x81, 0x1F)
NEG_WASH   = RGBColor(0xF2, 0xE0, 0xDE)
NEU_WASH   = RGBColor(0xE6, 0xE9, 0xE6)
POS_WASH   = RGBColor(0xDD, 0xE9, 0xE2)
AMBER_WASH = RGBColor(0xF5, 0xEA, 0xD2)
CODE_BG    = RGBColor(0x1A, 0x22, 0x26)
CODE_INK   = RGBColor(0xDF, 0xE7, 0xE6)
CODE_COM   = RGBColor(0x8F, 0xA8, 0xAA)

# ---------------------------------------------------------------- fonts
SANS  = "Segoe UI"
MONO  = "Consolas"
BANGLA = "Nirmala UI"

# Hard floor for this deck.
MIN_PT = 21

# ---------------------------------------------------------------- geometry
SLIDE_W, SLIDE_H = 13.333, 7.5
ML, MR = 0.72, 0.72          # left and right margin
MT, MB = 0.46, 0.44          # top and bottom margin
CONTENT_W = SLIDE_W - ML - MR

BENGALI = re.compile(r"[ঀ-৿‌‍।॥]")


def _blank(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = PAPER
    return slide


def _segments(text: str):
    """Split a string into Bangla and non-Bangla pieces so each can take the
    font that actually renders it."""
    out, buf, buf_bn = [], "", None
    for ch in text:
        is_bn = bool(BENGALI.match(ch))
        if buf_bn is None:
            buf_bn = is_bn
        if is_bn != buf_bn and ch != " ":
            out.append((buf, buf_bn))
            buf, buf_bn = ch, is_bn
        else:
            buf += ch
    if buf:
        out.append((buf, buf_bn))
    return out


def _write(para, text, size, color, bold=False, mono=False, italic=False):
    """Add runs to a paragraph, honouring **bold** markers and swapping the
    font for Bangla so nothing renders as empty boxes."""
    assert size >= MIN_PT, f"{size} pt is below the {MIN_PT} pt floor"
    for chunk in re.split(r"(\*\*.+?\*\*)", text):
        if not chunk:
            continue
        is_bold = bold
        if chunk.startswith("**") and chunk.endswith("**"):
            chunk, is_bold = chunk[2:-2], True
        for piece, is_bn in _segments(chunk):
            if not piece:
                continue
            run = para.add_run()
            run.text = piece
            f = run.font
            f.size = Pt(size)
            f.bold = is_bold
            f.italic = italic
            f.color.rgb = color
            f.name = BANGLA if is_bn else (MONO if mono else SANS)


def textbox(slide, x, y, w, h, blocks, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, b in enumerate(blocks):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.line_spacing = b.get("line", 1.18)
        para.space_after = Pt(b.get("after", 6))
        para.space_before = Pt(b.get("before", 0))
        para.alignment = b.get("align", PP_ALIGN.LEFT)
        text = b["t"]
        if b.get("bullet"):
            text = "•   " + text
        _write(para, text, b.get("size", 22), b.get("color", INK),
               b.get("bold", False), b.get("mono", False))
    return box


def rect(slide, x, y, w, h, fill=None, line=None, line_w=1.0):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    sh.text_frame.word_wrap = True
    return sh


def header(slide, eyebrow, title, lede=None, title_size=36):
    y = MT
    if eyebrow:
        textbox(slide, ML, y, CONTENT_W, 0.34,
                [{"t": eyebrow, "size": 21, "color": SIGNAL, "bold": True, "mono": True, "after": 0}])
        y += 0.44
    tb = textbox(slide, ML, y, CONTENT_W, 0.9,
                 [{"t": title, "size": title_size, "color": INK, "bold": True, "line": 1.06, "after": 0}])
    y += 0.62 if len(title) < 52 else 1.18
    if lede:
        textbox(slide, ML, y, CONTENT_W, 0.9,
                [{"t": lede, "size": 24, "color": INK_SOFT, "line": 1.24, "after": 0}])
        y += 0.44 * (1 + len(lede) // 96)
    return y + 0.30


def accent_panel(slide, x, y, w, h, accent, blocks, fill=PANEL):
    rect(slide, x, y, w, h, fill=fill, line=LINE)
    if accent is not None:
        rect(slide, x, y, 0.055, h, fill=accent)
    textbox(slide, x + 0.30, y + 0.20, w - 0.55, h - 0.40, blocks)


def stat_strip(slide, y, items, height=1.02):
    """Equal-width figure tiles. Numbers are the point, so they get the size."""
    n = len(items)
    w = CONTENT_W / n
    rect(slide, ML, y, CONTENT_W, height, fill=PANEL, line=LINE)
    for i, (value, label) in enumerate(items):
        x = ML + i * w
        if i:
            rect(slide, x, y + 0.10, 0.008, height - 0.20, fill=LINE)
        textbox(slide, x + 0.20, y + 0.13, w - 0.34, 0.46,
                [{"t": value, "size": 32, "color": INK, "bold": True, "after": 0}])
        textbox(slide, x + 0.20, y + 0.62, w - 0.30, 0.32,
                [{"t": label, "size": 21, "color": INK_FAINT, "mono": True, "after": 0}])
    return y + height + 0.26


def table(slide, x, y, w, rows, col_w, aligns=None, size=21, head_size=21,
          row_h=0.42, highlight=None, rowfill=None):
    """rows[0] is the header. col_w are relative weights."""
    n_rows, n_cols = len(rows), len(rows[0])
    total = sum(col_w)
    gfx = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y),
                                 Inches(w), Inches(row_h * n_rows))
    tbl = gfx.table
    tbl.first_row = False
    tbl.horz_banding = False
    for c, weight in enumerate(col_w):
        tbl.columns[c].width = Inches(w * weight / total)
    for r in range(n_rows):
        tbl.rows[r].height = Inches(row_h)
        for c in range(n_cols):
            cell = tbl.cell(r, c)
            cell.margin_left = Inches(0.11)
            cell.margin_right = Inches(0.11)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if r == 0:
                cell.fill.fore_color.rgb = PANEL_2
            elif highlight is not None and r == highlight:
                cell.fill.fore_color.rgb = PANEL_2
            elif rowfill and r in rowfill:
                cell.fill.fore_color.rgb = rowfill[r]
            else:
                cell.fill.fore_color.rgb = PANEL
            tf = cell.text_frame
            tf.word_wrap = True
            para = tf.paragraphs[0]
            para.alignment = (aligns[c] if aligns else PP_ALIGN.LEFT)
            is_head = r == 0
            _write(para, str(rows[r][c]),
                   head_size if is_head else size,
                   INK_FAINT if is_head else INK,
                   bold=is_head or (highlight is not None and r == highlight),
                   mono=is_head or (aligns is not None and aligns[c] == PP_ALIGN.RIGHT))
    return y + row_h * n_rows + 0.24


def code_box(slide, x, y, w, lines, size=21, label=None):
    """Comment lines start with # and are dimmed, the way they are in the
    HTML deck."""
    lh = size * 1.24 / 72
    top = y
    if label:
        textbox(slide, x, y, w, 0.3,
                [{"t": label, "size": 21, "color": INK_FAINT, "mono": True, "bold": True, "after": 0}])
        top = y + 0.38
    h = lh * len(lines) + 0.30
    # At a 21 pt floor a code block runs out of page quickly, so refuse to
    # draw one that would overflow rather than letting it slide off.
    room = SLIDE_H - MB - top
    assert h <= room, (f"code block needs {h:.2f} in but only {room:.2f} in remains; "
                       f"cut {int((h - room) / lh) + 1} line(s)")
    rect(slide, x, top, w, h, fill=CODE_BG, line=LINE)
    box = slide.shapes.add_textbox(Inches(x + 0.20), Inches(top + 0.16),
                                   Inches(w - 0.34), Inches(h - 0.30))
    tf = box.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.line_spacing = 1.24
        para.space_after = Pt(0)
        colour = CODE_COM if line.lstrip().startswith("#") else CODE_INK
        _write(para, line if line else " ", size, colour, mono=True)
    return top + h + 0.24


def quote(slide, x, y, w, bangla, gloss, accent=SIGNAL, size=26, h=None):
    """A transcript line with its English reading underneath."""
    height = h or (0.62 + 0.34 * (1 + len(gloss) // 58))
    rect(slide, x, y, w, height, fill=PANEL, line=LINE)
    rect(slide, x, y, 0.055, height, fill=accent)
    textbox(slide, x + 0.30, y + 0.16, w - 0.55, 0.46,
            [{"t": bangla, "size": size, "color": INK, "after": 0, "line": 1.3}])
    textbox(slide, x + 0.30, y + 0.16 + 0.44, w - 0.55, height - 0.62,
            [{"t": gloss, "size": 21, "color": INK_SOFT, "after": 0, "line": 1.2}])
    return y + height + 0.22


def chip(slide, x, y, text, fg, bg, w=1.5):
    sh = rect(slide, x, y, w, 0.34, fill=bg)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _write(p, text, 21, fg, bold=True, mono=True)
    return sh


def page_number(slide, n):
    textbox(slide, SLIDE_W - MR - 1.0, SLIDE_H - MB - 0.16, 1.0, 0.3,
            [{"t": f"{n:02d}", "size": 21, "color": INK_FAINT, "mono": True,
              "align": PP_ALIGN.RIGHT, "after": 0}])


def build_result_figures(root: Path) -> dict[str, Path]:
    """Create reproducible visual evidence for the trained baseline.

    The saved Logistic Regression run does not contain epoch-level loss
    history.  We therefore show a learning curve over training-set size,
    which is the honest analogue for this batch-trained baseline, plus the
    held-out class metrics and two readable confusion-matrix colour views.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    data = root / "sentiment" / "data"
    metrics_path = root / "sentiment" / "runs" / "baseline" / "metrics.json"
    pred_path = root / "sentiment" / "runs" / "baseline" / "predictions.jsonl"
    out = root / "presentation_figures"
    out.mkdir(exist_ok=True)
    labels = ["negative", "non_evaluative", "positive"]
    short_labels = ["negative", "non-evaluative", "positive"]

    def load_jsonl(path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def encode(row):
        surfaces = " ".join(dict.fromkeys(m["surface"] for m in row["mentions"]))
        return f"{surfaces} ‖ {row['text']}"

    train = load_jsonl(data / "train.jsonl")
    val = load_jsonl(data / "val.jsonl")
    test = load_jsonl(data / "test.jsonl")
    metric = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = {}
    for row in load_jsonl(pred_path):
        predictions[row["pair_id"]] = row["pred"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "axes.titleweight": "bold", "axes.edgecolor": "#9AA5A1",
        "axes.labelcolor": "#3E4A4F", "xtick.color": "#3E4A4F",
        "ytick.color": "#3E4A4F", "figure.facecolor": "#ECF0ED",
        "axes.facecolor": "#F8F9F7", "savefig.facecolor": "#ECF0ED",
    })

    # Learning curve: retrain the exact baseline at fixed fractions.  This is
    # not called a loss curve because sklearn's saved batch run has no epochs.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=3, max_features=200_000, sublinear_tf=True)
    x_train = vectorizer.fit_transform([encode(row) for row in train])
    x_val = vectorizer.transform([encode(row) for row in val])
    y_train = np.array([row["silver_label"] for row in train])
    y_val = np.array([row["silver_label"] for row in val])
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    train_scores, val_scores = [], []
    for fraction in fractions:
        n_rows = len(train) if fraction == 1.0 else int(len(train) * fraction)
        if n_rows < len(labels):
            n_rows = len(labels)
        if fraction == 1.0:
            indices = np.arange(len(train))
        else:
            indices, _ = train_test_split(
                np.arange(len(train)), train_size=n_rows,
                stratify=y_train, random_state=17)
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        clf.fit(x_train[indices], y_train[indices])
        train_pred = clf.predict(x_train[indices])
        val_pred = clf.predict(x_val)
        train_scores.append(f1_score(y_train[indices], train_pred, average="macro",
                                     labels=labels, zero_division=0))
        val_scores.append(f1_score(y_val, val_pred, average="macro",
                                   labels=labels, zero_division=0))

    fig, ax = plt.subplots(figsize=(10.8, 4.7), dpi=180)
    x = np.array(fractions) * 100
    ax.plot(x, train_scores, marker="o", linewidth=3, color="#1D4E5F", label="Training macro F1")
    ax.plot(x, val_scores, marker="o", linewidth=3, color="#A6423C", label="Validation silver macro F1")
    ax.axhline(metric["test_gold_macro_f1"], linestyle="--", linewidth=2,
               color="#B5811F", label=f"Held-out human test: {metric['test_gold_macro_f1']:.3f}")
    ax.set_title("TF-IDF + balanced Logistic Regression learning curve")
    ax.set_xlabel("Percentage of silver-labelled training data")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    learning = out / "baseline_learning_curve.png"
    fig.savefig(learning, bbox_inches="tight")
    plt.close(fig)

    report = metric["test_gold_report"]
    fig, ax = plt.subplots(figsize=(9.8, 4.9), dpi=180)
    idx = np.arange(3)
    width = 0.24
    for offset, key, colour in [(-width, "precision", "#1D4E5F"),
                                (0, "recall", "#2E8FA3"),
                                (width, "f1-score", "#A6423C")]:
        vals = [report[name][key] for name in labels]
        ax.bar(idx + offset, vals, width, label=key.replace("-score", ""), color=colour)
        for x0, value in zip(idx + offset, vals):
            ax.text(x0, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Held-out human test performance: 59 target pairs")
    ax.set_ylabel("Score")
    ax.set_xticks(idx, short_labels)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    class_metrics = out / "baseline_class_metrics.png"
    fig.savefig(class_metrics, bbox_inches="tight")
    plt.close(fig)

    gold = [row for row in test if row.get("gold_label")]
    y_true = [row["gold_label"] for row in gold]
    y_pred = [predictions[row["pair_id"]] for row in gold]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=180)
    for ax, matrix, cmap, title, fmt in [
        (axes[0], cm, "Blues", "Raw counts", "d"),
        (axes[1], row_norm, "YlOrRd", "Row-normalized recall", ".0%"),
    ]:
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=cm.max() if fmt == "d" else 1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Human label")
        ax.set_xticks(range(3), short_labels, rotation=20, ha="right")
        ax.set_yticks(range(3), short_labels)
        threshold = (cm.max() if fmt == "d" else 1) * 0.55
        for i in range(3):
            for j in range(3):
                value = matrix[i, j]
                text_colour = "white" if value > threshold else "#14201F"
                text = f"{value:{fmt}}" if fmt == "d" else f"{value:.0%}"
                ax.text(j, i, text, ha="center", va="center", color=text_colour,
                        fontsize=16, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Baseline confusion matrix on unseen episodes", fontweight="bold", y=1.01)
    fig.tight_layout()
    confusion = out / "baseline_confusion_variants.png"
    fig.savefig(confusion, bbox_inches="tight")
    plt.close(fig)

    return {"learning": learning, "class_metrics": class_metrics, "confusion": confusion}


# ================================================================ build
def build(path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    n = [0]

    def new(eyebrow=None, title=None, lede=None, title_size=36, numbered=True):
        s = _blank(prs)
        n[0] += 1
        if numbered:
            page_number(s, n[0])
        y = header(s, eyebrow, title, lede, title_size) if title else MT
        return s, y

    # ---------------------------------------------------------- 1 title
    s = _blank(prs); n[0] += 1
    rect(s, 0, 0, SLIDE_W, 0.14, fill=PETROL)
    textbox(s, ML, 1.30, CONTENT_W, 0.4,
            [{"t": "MACHINE LEARNING LABORATORY   /   CSE 4112", "size": 21,
              "color": PETROL, "bold": True, "mono": True, "after": 0}])
    textbox(s, ML, 1.92, CONTENT_W, 2.5,
            [{"t": "Speaker-Attributed Financial", "size": 52, "color": INK, "bold": True, "line": 1.02, "after": 0},
             {"t": "Discourse Analysis of", "size": 52, "color": INK, "bold": True, "line": 1.02, "after": 0},
             {"t": "Bangla Talk Shows", "size": 52, "color": INK, "bold": True, "line": 1.02, "after": 0}])
    textbox(s, ML, 4.52, 10.4, 0.9,
            [{"t": "One pipeline from raw broadcast audio to a polarity judgement about a "
                   "named financial target: who spoke, what they said, and how they judged the economy.",
              "size": 24, "color": INK_SOFT, "line": 1.26, "after": 0}])
    rect(s, ML, 5.72, CONTENT_W, 0.03, fill=PETROL)
    meta = [("ROLLS", "2107001, 2107004, 2107006,\n2107009, 2107010, 2107015"),
            ("DIARIZATION", "pyannote\nspeaker-diarization-community-1"),
            ("SPEECH RECOGNITION", "Whisper medium,\nBengali adapted"),
            ("SENTIMENT", "TF-IDF + LogReg\n(BanglaBERT not run)")]
    cw = CONTENT_W / 4
    for i, (k, v) in enumerate(meta):
        textbox(s, ML + i * cw, 5.92, cw - 0.24, 0.3,
                [{"t": k, "size": 21, "color": INK_FAINT, "mono": True, "bold": True, "after": 4}])
        textbox(s, ML + i * cw, 6.30, cw - 0.24, 0.8,
                [{"t": v, "size": 21, "color": INK, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 2 corpus
    s, y = new("01 / THE DATA", "The corpus at a glance")
    y = stat_strip(s, y, [("26", "EPISODES"), ("16.88", "HOURS"), ("1,867", "SPEAKER TURNS"),
                          ("130,750", "WORDS"), ("3,532", "UTTERANCES")], height=1.18)
    y = table(s, ML, y, CONTENT_W,
              [["Roll", "Episodes", "Words", "Programmes"],
               ["2107001", "4", "21,500", "Business Talk, ATN"],
               ["2107004", "6", "29,383", "Star Biz, Dhaka Stream"],
               ["2107006", "5", "25,736", "RTV Business Talk"],
               ["2107009", "5", "23,447", "Money Talks, Desh TV"],
               ["2107010", "1", "2,960", "Star Biz Dialogue"],
               ["2107015", "5", "27,724", "ATN, Gtv"],
               ["Total", "26", "130,750", "16.88 hours"]],
              [1.1, 1.0, 1.0, 2.6],
              [PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT],
              row_h=0.415, highlight=7)

    # ---------------------------------------------------------- 3 hard case
    s, y = new("01 / THE DATA", "Why this audio is the hard case",
               "Long-form broadcast panel discussion breaks most of the assumptions a speech model is trained on.")
    for t in ["Panel format, so two people frequently talk over each other",
              "Studio stings and background music between segments",
              "Guests joining by phone at much lower bandwidth",
              "Heavy code switching: English finance terms inside Bangla sentences, "
              "for example রেভিনিউ and ফিসকাল পলিসি",
              "Episodes run 24 to 48 minutes, far past what a speech model reads in one pass"]:
        textbox(s, ML, y, CONTENT_W, 0.5, [{"t": t, "size": 24, "color": INK, "bullet": True, "line": 1.2, "after": 0}])
        y += 0.52 + (0.30 if len(t) > 84 else 0)
    accent_panel(s, ML, y + 0.14, CONTENT_W, 0.98, PETROL,
                 [{"t": "Episode identifiers are namespaced by roll, so ep001 from one contributor and "
                        "ep001 from another stay distinct, and speaker identities are never merged "
                        "across recordings.", "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 4 architecture
    s, y = new("02 / ARCHITECTURE", "Five stages, each writing output the next one reads")
    stages = [("STAGE 1", "Audio", "conditioning"), ("STAGE 2", "Diarization", "1,867 turns"),
              ("STAGE 3", "Recognition", "130,750 words"), ("STAGE 4", "Fusion", "3,532 utterances"),
              ("STAGE 5", "Sentiment", "3,109 pairs")]
    bw, gap = 2.22, 0.20
    total = len(stages) * bw + (len(stages) - 1) * gap
    x0 = (SLIDE_W - total) / 2
    by = y + 0.30
    for i, (num, name, sub) in enumerate(stages):
        x = x0 + i * (bw + gap)
        rect(s, x, by, bw, 1.72, fill=PANEL, line=PETROL, line_w=1.4)
        textbox(s, x + 0.18, by + 0.18, bw - 0.34, 0.3,
                [{"t": num, "size": 21, "color": SIGNAL, "mono": True, "bold": True, "after": 0}])
        textbox(s, x + 0.18, by + 0.62, bw - 0.34, 0.4,
                [{"t": name, "size": 24, "color": INK, "bold": True, "after": 0}])
        textbox(s, x + 0.18, by + 1.12, bw - 0.34, 0.4,
                [{"t": sub, "size": 21, "color": INK_FAINT, "mono": True, "after": 0, "line": 1.1}])
        if i:
            ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x - gap - 0.02),
                                    Inches(by + 0.78), Inches(gap + 0.04), Inches(0.17))
            ar.fill.solid(); ar.fill.fore_color.rgb = PETROL
            ar.line.fill.background(); ar.shadow.inherit = False
    y = by + 2.10
    for t in ["Diarization runs on the whole episode and does not depend on the transcript.",
              "Speech recognition runs in 28 second chunks and does not depend on the speaker labels.",
              "Fusion is the only stage that needs both, which is why its errors are the ones that compound."]:
        textbox(s, ML, y, CONTENT_W, 0.36, [{"t": t, "size": 22, "color": INK_SOFT, "after": 0}])
        y += 0.40

    # ---------------------------------------------------------- 5 audio config
    s, y = new("03 / STAGE 1, AUDIO", "Getting broadcast audio into a state a model can read",
               "Both speech models expect 16 kHz mono at a predictable loudness. Broadcast MP3 is none of those things.")
    table(s, ML, y, 6.05,
          [["Setting", "Value"],
           ["source rate", "44.1 and 48 kHz"],
           ["source channels", "2, stereo"],
           ["target rate", "16 kHz mono"],
           ["loudness standard", "EBU R128 integrated"],
           ["loudness target", "-16 LUFS"],
           ["measured range", "-28.3 to -16.0 LUFS"],
           ["true peak guard", "checked before gain"]],
          [1.3, 1.4], [PP_ALIGN.LEFT, PP_ALIGN.RIGHT], row_h=0.44)
    accent_panel(s, ML + 6.45, y, CONTENT_W - 6.45, 1.62, PETROL,
                 [{"t": "Gain is applied selectively", "size": 23, "color": INK, "bold": True, "after": 6},
                  {"t": "Only when an episode is genuinely off target and the true peak leaves headroom. "
                        "Normalising everything blindly would clip the loud ones.",
                   "size": 21, "color": INK_SOFT, "line": 1.18, "after": 0}])
    accent_panel(s, ML + 6.45, y + 1.80, CONTENT_W - 6.45, 1.94, SIGNAL,
                 [{"t": "Why it matters downstream", "size": 23, "color": INK, "bold": True, "after": 6},
                  {"t": "Diarization clusters speaker embeddings by distance. If one episode arrives "
                        "12 dB quieter, the embeddings shift and a threshold that worked on one "
                        "recording stops working on the next.",
                   "size": 21, "color": INK_SOFT, "line": 1.18, "after": 0}])

    # ---------------------------------------------------------- 6 diarization config
    s, y = new("04 / STAGE 2, DIARIZATION", "Deciding who is speaking, before knowing what was said")
    table(s, ML, y, 6.05,
          [["Setting", "Value"],
           ["backbone", "pyannote community-1"],
           ["embedding", "wespeaker resnet34-LM"],
           ["min_duration_off", "0.1 s"],
           ["min_cluster_size", "20"],
           ["minimum segment", "0.75 s"],
           ["stitch gap", "0.17 s"],
           ["minimum speaker airtime", "9.0 s"],
           ["scoring collar", "0.25 s"]],
          [1.5, 1.3], [PP_ALIGN.LEFT, PP_ALIGN.RIGHT], row_h=0.42)
    yy = stat_strip_right = y
    xs = ML + 6.45
    ww = CONTENT_W - 6.45
    rect(s, xs, yy, ww, 1.05, fill=PANEL, line=LINE)
    for i, (v, l) in enumerate([("1,867", "TURNS"), ("3", "MEDIAN SPK"), ("1 to 7", "RANGE")]):
        cx = xs + i * (ww / 3)
        if i:
            rect(s, cx, yy + 0.10, 0.008, 0.85, fill=LINE)
        textbox(s, cx + 0.18, yy + 0.14, ww / 3 - 0.3, 0.44,
                [{"t": v, "size": 30, "color": INK, "bold": True, "after": 0}])
        textbox(s, cx + 0.18, yy + 0.64, ww / 3 - 0.3, 0.3,
                [{"t": l, "size": 21, "color": INK_FAINT, "mono": True, "after": 0}])
    yy += 1.28
    textbox(s, xs, yy, ww, 0.34, [{"t": "The default embedding is swapped for wespeaker "
                                        "resnet34-LM, which separates voices better on noisy "
                                        "broadcast audio.", "size": 21, "color": INK_SOFT,
                                   "line": 1.18, "after": 0}])
    accent_panel(s, xs, yy + 1.00, ww, 1.72, AMBER,
                 [{"t": "No diarization error rate is reported", "size": 23, "color": INK, "bold": True, "after": 6},
                  {"t": "That metric needs a human made reference of who spoke when, and none exists "
                        "for this corpus. Scoring the output against itself would be meaningless.",
                   "size": 21, "color": INK_SOFT, "line": 1.18, "after": 0}])

    # ---------------------------------------------------------- 7 diarization thresholds
    s, y = new("04 / STAGE 2, DIARIZATION", "What the post-processing thresholds are for")
    for t in ["Segments under 0.75 s are dropped, so a cough or a jingle does not become a speaker",
              "Gaps under 0.17 s are stitched, so one sentence does not split into three turns",
              "Any speaker with under 9 s of total airtime is deleted, which removes spurious clusters",
              "Both the raw and the post-processed output are kept, so the effect of each threshold "
              "can be inspected rather than assumed"]:
        textbox(s, ML, y, CONTENT_W, 0.5, [{"t": t, "size": 24, "color": INK, "bullet": True, "line": 1.2, "after": 0}])
        y += 0.56 + (0.30 if len(t) > 86 else 0)
    accent_panel(s, ML, y + 0.20, CONTENT_W, 1.30, PETROL,
                 [{"t": "Speaker identity stays episode local", "size": 24, "color": INK, "bold": True, "after": 6},
                  {"t": "Speaker 2 in one episode has no relationship to Speaker 2 in another. The labels "
                        "are cluster identifiers, not people, and they are never joined across recordings.",
                   "size": 22, "color": INK_SOFT, "line": 1.18, "after": 0}])

    # ---------------------------------------------------------- 8 ASR config
    s, y = new("05 / STAGE 3, SPEECH RECOGNITION", "Transcribing 40 minute episodes with a 30 second model",
               "Whisper reads a fixed 30 second window, so every long-form system is really a chunking problem.")
    table(s, ML, y, 6.05,
          [["Setting", "Value"],
           ["checkpoint", "Whisper medium, bn"],
           ["runtime", "CTranslate2, float16"],
           ["language", "bn, forced"],
           ["chunk length", "28.0 s"],
           ["beam size", "5"],
           ["condition on previous", "False"],
           ["source separation", "on detection"]],
          [1.5, 1.3], [PP_ALIGN.LEFT, PP_ALIGN.RIGHT], row_h=0.44)
    xs, ww = ML + 6.45, CONTENT_W - 6.45
    rect(s, xs, y, ww, 1.05, fill=PANEL, line=LINE)
    for i, (v, l) in enumerate([("2,184", "CHUNKS"), ("4,006", "SEGMENTS"), ("130,750", "WORDS")]):
        cx = xs + i * (ww / 3)
        if i:
            rect(s, cx, y + 0.10, 0.008, 0.85, fill=LINE)
        textbox(s, cx + 0.16, y + 0.14, ww / 3 - 0.26, 0.44,
                [{"t": v, "size": 27, "color": INK, "bold": True, "after": 0}])
        textbox(s, cx + 0.16, y + 0.64, ww / 3 - 0.26, 0.3,
                [{"t": l, "size": 21, "color": INK_FAINT, "mono": True, "after": 0}])
    accent_panel(s, xs, y + 1.28, ww, 2.44, SIGNAL,
                 [{"t": "Why a Bengali adapted checkpoint", "size": 23, "color": INK, "bold": True, "after": 6},
                  {"t": "It produces মূল্যস্ফীতি, "
                        "খেলাপি ঋণ and "
                        "বৈদেশিক মুদ্রার "
                        "রিজার্ভ as clean tokens rather than phonetic "
                        "approximations. The whole downstream lexicon depends on that.",
                   "size": 21, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 9 ASR settings rationale
    s, y = new("05 / STAGE 3, SPEECH RECOGNITION", "Why each decoding setting is set that way")
    items = [("28 s rather than 30 s", "leaves headroom inside the receptive field, so the window edge never lands exactly at the model limit"),
             ("Language forced to Bengali", "stops the detector switching to Hindi or Assamese on a noisy chunk"),
             ("No conditioning on previous text", "stops a hallucination in one chunk being fed as context into the next, which is how one error becomes a whole bad minute"),
             ("Source separation on detection", "measures spectral flux per chunk and separates music only when present, rather than degrading clean speech")]
    for head, body in items:
        textbox(s, ML, y, CONTENT_W, 0.34, [{"t": head, "size": 24, "color": PETROL, "bold": True, "after": 0}])
        textbox(s, ML, y + 0.38, CONTENT_W, 0.5,
                [{"t": body, "size": 22, "color": INK_SOFT, "line": 1.18, "after": 0}])
        y += 0.94 + (0.28 if len(body) > 100 else 0)

    # ---------------------------------------------------------- 10 defect 1
    s, y = new("06 / TRANSCRIPT QUALITY", "Defect one: chunk boundary truncation",
               "Bangla never begins a word with a dependent vowel sign. When a segment does, "
               "the stitcher has eaten the opening consonant.")
    chip(s, ML, y, "400 OF 3,532", AMBER, AMBER_WASH, w=2.7)
    y += 0.56
    y = quote(s, ML, y, CONTENT_W,
              "ুক্তা উন্নতি লক্ষ্য করা যাচ্ছে",
              "The word should be কিছুটা. The first token of 11.3 percent of "
              "utterances is not a real word.", accent=AMBER, size=30, h=1.34)
    accent_panel(s, ML, y + 0.10, CONTENT_W, 1.44, POS,
                 [{"t": "Fix", "size": 24, "color": INK, "bold": True, "after": 6},
                  {"t": "Chunks are currently cut back to back with no overlap. Cutting them with a "
                        "2 to 3 second overlap and stitching on the shared region removes the boundary "
                        "word entirely. This is a rerun setting, not new code.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 11 defect 2
    s, y = new("06 / TRANSCRIPT QUALITY", "Defect two: decoder repetition loops",
               "The decoder locks onto one token and emits it for many seconds.")
    chip(s, ML, y, "46 OF 3,532", AMBER, AMBER_WASH, w=2.5)
    y += 0.56
    y = quote(s, ML, y, CONTENT_W,
              "বাজার বাজার বাজার "
              "বাজার বাজার বাজার",
              "These carry no readable meaning, so they are dropped from supervision rather than "
              "quietly labelled neutral.", accent=AMBER, size=30, h=1.34)
    accent_panel(s, ML, y + 0.10, CONTENT_W, 1.52, NEG,
                 [{"t": "Likely cause, and it is in the configuration", "size": 24, "color": INK, "bold": True, "after": 6},
                  {"t": "The run sets a repetition penalty of 0.8. A penalty above 1.0 suppresses tokens "
                        "already produced. A value below 1.0 does the opposite and makes repeating them "
                        "more likely. Raising it above 1.0 is the direct fix.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 12 fusion
    s, y = new("07 / STAGE 4, FUSION", "Attaching every word to a speaker",
               "Diarization gives intervals with speaker labels. Recognition gives words with timestamps. "
               "Neither knows about the other.")
    for t in ["Take the midpoint of the word. If it falls inside one speaker interval, that speaker owns it",
              "If the midpoint falls in a gap, fall back to the interval with the largest overlap",
              "If two speakers overlap in time, the one who started first keeps the word",
              "Consecutive words from one speaker are grouped into an utterance, the unit used downstream"]:
        textbox(s, ML, y, CONTENT_W, 0.5, [{"t": t, "size": 23, "color": INK, "bullet": True, "line": 1.18, "after": 0}])
        y += 0.54 + (0.28 if len(t) > 88 else 0)
    textbox(s, ML, y + 0.10, CONTENT_W, 0.4,
            [{"t": "The midpoint is used rather than the word start because start timestamps drift at "
                   "chunk boundaries and the midpoint is the more stable of the two.",
              "size": 21, "color": INK_SOFT, "line": 1.18, "after": 0}])
    stat_strip(s, y + 0.76, [("3,532", "UTTERANCES"), ("85.1%", "WORDS ASSIGNED")], height=1.0)
    textbox(s, ML, y + 1.86, CONTENT_W, 0.32,
            [{"t": "Overall coverage; 99.5% applies only to the pyannote subset. Neither is speaker accuracy.",
              "size": 21, "color": INK_FAINT, "mono": True, "after": 0}])

    # ---------------------------------------------------------- 13 stem matching
    s, y = new("08 / STAGE 5, TARGETS", "Finding what a sentence is actually about",
               "Bangla attaches case and definiteness as suffixes, so exact string matching sees six "
               "unrelated words where there is one target.")
    table(s, ML, y, 5.6,
          [["Surface form", "Count"],
           ["ব্যাংক", "164"],
           ["ব্যাংকের", "139"],
           ["ব্যাংকিং", "99"],
           ["ব্যাংকে", "50"],
           ["ব্যাংকগুলো", "10"],
           ["ব্যাংকগুলোর", "10"],
           ["one stem covers", "555"]],
          [2.0, 1.0], [PP_ALIGN.LEFT, PP_ALIGN.RIGHT], row_h=0.43, highlight=7)
    xs, ww = ML + 6.0, CONTENT_W - 6.0
    textbox(s, xs, y, ww, 0.4,
            [{"t": "A prefix rule is blunt, so each entry carries an exclusion list.",
              "size": 22, "color": INK_SOFT, "line": 1.18, "after": 0}])
    table(s, xs, y + 0.62, ww,
          [["Stem", "Blocked", "Reason"],
           ["সুদ", "সুদান", "interest vs Sudan"],
           ["ভালো", "ভালোবাসা", "good vs love"],
           ["সুবিধা", "অসুবিধা", "benefit vs drawback"],
           ["ঋণ", "ঋণখেলাপি", "credit vs default"]],
          [1.0, 1.3, 1.7], row_h=0.46)

    # ---------------------------------------------------------- 14 lexicon coverage
    s, y = new("08 / STAGE 5, TARGETS", "Target lexicon coverage")
    y = stat_strip(s, y, [("52", "TARGETS"), ("3,109", "TARGET PAIRS"), ("48", "SEEN IN CORPUS")], height=1.18)
    y = table(s, ML, y, CONTENT_W,
              [["Target type", "Pairs", "Examples"],
               ["MACRO_INDICATOR", "1,121", "মূল্যস্ফীতি, রিজার্ভ, বিনিয়োগ"],
               ["REGULATOR", "980", "বাংলাদেশ ব্যাংক, বাজেট"],
               ["SECTOR", "901", "গ্যাস, কৃষি, শিল্প"],
               ["MARKET", "107", "শেয়ারবাজার, পুঁজিবাজার"]],
              [1.6, 0.9, 2.6], [PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT], row_h=0.48)
    accent_panel(s, ML, y + 0.10, CONTENT_W, 1.14, PETROL,
                 [{"t": "One utterance produces one row per distinct target it mentions, so a sentence "
                        "naming both inflation and reserves is scored twice, once for each.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 15 polarity idea
    s, y = new("09 / THE POLARITY IDEA", "A rising number is not good or bad until you know what is rising",
               "Ordinary sentiment analysis treats an increase word as positive. In economic talk that is "
               "wrong about half the time.", title_size=34)
    half = (CONTENT_W - 0.4) / 2
    rect(s, ML, y, half, 2.42, fill=PANEL, line=LINE)
    rect(s, ML, y, 0.055, 2.42, fill=POS)
    textbox(s, ML + 0.30, y + 0.22, half - 0.6, 0.5,
            [{"t": "বৈদেশিক মুদ্রার "
                   "রিজার্ভ বেড়েছে",
              "size": 28, "color": INK, "after": 0}])
    textbox(s, ML + 0.30, y + 0.82, half - 0.6, 0.4,
            [{"t": "Foreign exchange reserves have increased.", "size": 21, "color": INK_SOFT, "after": 0}])
    textbox(s, ML + 0.30, y + 1.32, half - 0.6, 0.9,
            [{"t": "Reserves carry orientation +1. The increase cue keeps its sign.",
              "size": 22, "color": INK, "line": 1.2, "after": 6}])
    chip(s, ML + 0.30, y + 1.94, "POSITIVE", POS, POS_WASH, w=1.8)

    x2 = ML + half + 0.4
    rect(s, x2, y, half, 2.42, fill=PANEL, line=LINE)
    rect(s, x2, y, 0.055, 2.42, fill=NEG)
    textbox(s, x2 + 0.30, y + 0.22, half - 0.6, 0.5,
            [{"t": "মূল্যস্ফীতি বেড়েছে",
              "size": 28, "color": INK, "after": 0}])
    textbox(s, x2 + 0.30, y + 0.82, half - 0.6, 0.4,
            [{"t": "Inflation has increased.", "size": 21, "color": INK_SOFT, "after": 0}])
    textbox(s, x2 + 0.30, y + 1.32, half - 0.6, 0.9,
            [{"t": "Inflation carries orientation -1. The same cue flips.",
              "size": 22, "color": INK, "line": 1.2, "after": 6}])
    chip(s, x2 + 0.30, y + 1.94, "NEGATIVE", NEG, NEG_WASH, w=1.9)

    textbox(s, ML, y + 2.66, CONTENT_W, 0.6,
            [{"t": "Because each target is scored separately, one sentence can carry two labels at once. "
                   "A sentence level model cannot represent that.",
              "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 16 orientation classes
    s, y = new("09 / THE POLARITY IDEA", "Every target carries an orientation")
    third = (CONTENT_W - 0.5) / 3
    cards = [(POS, "+1", "A rise is good news", "14 targets: reserves, remittance, GDP, exports, "
                                                "investment, employment, revenue, output, the market index."),
             (NEG, "-1", "A rise is bad news", "9 targets: inflation, consumer prices, bad loans, foreign "
                                               "debt, capital flight, the policy rate, public spending."),
             (NEU, "0", "An entity, no direction", "29 targets such as banks, the government or the energy "
                                                   "sector. Scored only by evaluative words.")]
    for i, (col, sign, head, body) in enumerate(cards):
        x = ML + i * (third + 0.25)
        rect(s, x, y, third, 3.12, fill=PANEL, line=LINE)
        rect(s, x, y, third, 0.075, fill=col)
        textbox(s, x + 0.26, y + 0.30, third - 0.5, 0.5,
                [{"t": f"ORIENTATION {sign}", "size": 21, "color": col, "bold": True, "mono": True, "after": 0}])
        textbox(s, x + 0.26, y + 0.80, third - 0.5, 0.6,
                [{"t": head, "size": 24, "color": INK, "bold": True, "line": 1.14, "after": 0}])
        textbox(s, x + 0.26, y + 1.62, third - 0.5, 1.3,
                [{"t": body, "size": 21, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 17 scoring rule
    s, y = new("10 / SUPERVISION", "Creating labels that can be traced back to words")
    code_box(s, ML, y, 7.55, [
        "# every cue token within 12 tokens of the target",
        "polarity = cue.polarity",
        "# direction words depend on the target",
        "if cue.kind == \"direction\":",
        "    if orientation == 0: skip",
        "    polarity *= orientation",
        "# adjectives and verbs can be negated, nouns cannot",
        "if cue.negatable and negator within 3 tokens:",
        "    polarity *= -1",
        "# nearer evidence counts for more",
        "decay = 1 / (1 + distance / 4)",
        "score += polarity * cue.weight * decay",
    ], size=21)
    accent_panel(s, ML + 7.95, y, CONTENT_W - 7.95, 3.60, PETROL,
                 [{"t": "Decision", "size": 24, "color": INK, "bold": True, "after": 8},
                  {"t": "Above +0.55 is positive.", "size": 22, "color": INK, "after": 4},
                  {"t": "Below -0.55 is negative.", "size": 22, "color": INK, "after": 4},
                  {"t": "Anything between is non evaluative.", "size": 22, "color": INK, "after": 12},
                  {"t": "The cue set holds 104 stems, each with a weight and a negation flag.",
                   "size": 21, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 18 negation
    s, y = new("10 / SUPERVISION", "Two corrections found by reading the output")
    textbox(s, ML, y, CONTENT_W, 0.4,
            [{"t": "Negation has scope. A first version flipped any cue near a negator, and scored "
                   "crisis as positive.", "size": 23, "color": INK, "bold": True, "line": 1.18, "after": 0}])
    y += 0.52
    y = quote(s, ML, y, CONTENT_W,
              "সংকট কাটছে না",
              "The crisis is not passing. The না negates the verb, not the noun, and it is still "
              "bad news. Only adjectives and verbs are now marked negatable.", accent=NEG, size=28, h=1.42)
    textbox(s, ML, y + 0.14, CONTENT_W, 0.4,
            [{"t": "Bangla builds a negated verb complex. The particle can sit three tokens from the "
                   "word it negates.", "size": 23, "color": INK, "bold": True, "line": 1.18, "after": 0}])
    quote(s, ML, y + 0.66, CONTENT_W,
          "স্থিতিশীল করতে "
          "পারবেন না",
          "You will not be able to stabilise it. Scored positive until the negation window was widened "
          "to three.", accent=NEG, size=28, h=1.30)

    # ---------------------------------------------------------- 19 label distribution
    s, y = new("10 / SUPERVISION", "What the rule set produced",
               "The class balance here is the single most important fact for reading every result "
               "that follows.")
    y = stat_strip(s, y, [("3,019", "TRAINING LABELS"), ("284", "NEGATIVE 9.4%"),
                          ("2,583", "NON EVAL 85.6%"), ("152", "POSITIVE 5.0%")], height=1.24)
    accent_panel(s, ML, y + 0.16, CONTENT_W, 1.34, AMBER,
                 [{"t": "Rows flagged as repetition loops, and rows under six words, are excluded from "
                        "supervision rather than silently labelled neutral. That is 90 of 3,109 pairs.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])
    accent_panel(s, ML, y + 1.68, CONTENT_W, 1.34, NEG,
                 [{"t": "Only 152 positive examples exist in the whole corpus. That thinness is why the "
                        "positive class is the weakest in every result that follows.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 20 split why
    s, y = new("11 / SPLIT", "Split by episode, never by row")
    textbox(s, ML, y, CONTENT_W, 1.0,
            [{"t": "Two utterances from the same talk show share the panel, the topic, the host's "
                   "phrasing and the same recognition error pattern. If rows from one episode land in "
                   "both training and test, the model can recognise the episode instead of the sentiment.",
              "size": 24, "color": INK, "line": 1.24, "after": 0}])
    y += 1.36
    accent_panel(s, ML, y, CONTENT_W, 1.30, NEG,
                 [{"t": "What a random row split would do", "size": 23, "color": INK, "bold": True, "after": 6},
                  {"t": "The reported score would rise, and none of that rise would survive contact with "
                        "a new episode.", "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])
    accent_panel(s, ML, y + 1.50, CONTENT_W, 1.30, POS,
                 [{"t": "What is done instead", "size": 23, "color": INK, "bold": True, "after": 6},
                  {"t": "All four test episodes are held out whole, and the split asserts that no episode "
                        "appears in more than one partition.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 21 split table
    s, y = new("11 / SPLIT", "How the corpus divides")
    y = table(s, ML, y, CONTENT_W,
              [["Split", "Episodes", "Pairs", "Human", "Negative", "Non eval", "Positive"],
               ["Train", "19", "2,026", "99", "164", "1,771", "91"],
               ["Validation", "3", "320", "22", "24", "278", "18"],
               ["Test", "4", "673", "59", "96", "534", "43"]],
              [1.5, 1.1, 1.0, 1.0, 1.1, 1.1, 1.1],
              [PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 6, row_h=0.52, highlight=3)
    accent_panel(s, ML, y + 0.18, CONTENT_W, 1.62, PETROL,
                 [{"t": "Validation is used only to pick the stopping epoch and is never fitted on. "
                        "Because the human labelled slice inside validation is small, model selection "
                        "uses the rule label agreement score, which is the honest consequence of having "
                        "limited human labels.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 22 code: defects
    s, y = new("12 / CODE", "Preprocessing: detecting the transcript defects")
    code_box(s, ML, y, CONTENT_W, [
        "# A word may never begin with a dependent sign. When a",
        "# segment does, the chunk stitcher ate a consonant.",
        "def leading_truncated(text):",
        "    s = text.lstrip()",
        "    return bool(s) and s[0] in COMBINING",
        "# Longest run of one token, plus its lexical variety.",
        "def repetition_score(text):",
        "    t = text.split()",
        "    longest = run = 1",
        "    for i in range(1, len(t)):",
        "        run = run + 1 if t[i] == t[i-1] else 1",
        "        longest = max(longest, run)",
        "    return longest, len(set(t)) / len(t)",
    ], size=21)

    # ---------------------------------------------------------- 23 code: scoring
    s, y = new("12 / CODE", "Target aware polarity scoring")
    code_box(s, ML, y, CONTENT_W, [
        "for i, token in enumerate(tokens):",
        "    distance = min(abs(i - t) for t in target_idx)",
        "    if distance > WINDOW:",
        "        continue",
        "    polarity = cue[\"polarity\"]",
        "    if cue[\"kind\"] == \"direction\":",
        "        # A rise is only good or bad relative to",
        "        # what is rising.",
        "        if orientation == 0:",
        "            continue",
        "        polarity *= orientation",
        "    decay = 1.0 / (1.0 + distance / 4.0)",
        "    score += polarity * cue[\"weight\"] * decay",
    ], size=21)

    # ---------------------------------------------------------- 24 code: split
    s, y = new("12 / CODE", "The split, with a leak check")
    code_box(s, ML, y, CONTENT_W, [
        "for row in rows:",
        "    ep = row[\"global_episode\"]",
        "    name = (\"test\"  if ep in TEST_EPISODES",
        "            else \"val\" if ep in VAL_EPISODES",
        "            else \"train\")",
        "    splits[name].append(row)",
        "# A leak check is cheap and catches a bad edit to the",
        "# episode lists before it can inflate a score.",
        "train_eps = {r[\"global_episode\"] for r in splits[\"train\"]}",
        "for name in (\"val\", \"test\"):",
        "    overlap = train_eps & {r[\"global_episode\"]",
        "                           for r in splits[name]}",
        "    assert not overlap, f\"{name} shares episodes\"",
    ], size=21)

    # ---------------------------------------------------------- 25 code: train
    s, y = new("12 / CODE", "Training, validation and test")
    code_box(s, ML, y, CONTENT_W, [
        "# Sentence pair is what makes the task target aware:",
        "#     [CLS] target [SEP] utterance [SEP]",
        "enc = tokenizer(target_text(row), row[\"text\"],",
        "                truncation=True, max_length=128)",
        "# 86% of rows are non evaluative. Unweighted, the model",
        "# predicts that one class for everything.",
        "loss_fn = CrossEntropyLoss(weight=class_weights)",
        "for epoch in range(1, args.epochs + 1):",
        "    for batch in train_loader:",
        "        loss = loss_fn(model(**batch).logits, labels)",
        "        loss.backward(); optimizer.step()",
        "    if evaluate(model, val_rows) > best:  # stopping epoch",
        "        best = evaluate(model, val_rows); save(model)",
    ], size=21)

    # ---------------------------------------------------------- 26 metrics rejected
    s, y = new("13 / METRICS", "Why plain accuracy is not the headline",
               "The class balance decides the metric. With 85.6 percent of rows in one class, "
               "the obvious metric is actively misleading.")
    accent_panel(s, ML, y, CONTENT_W, 1.56, NEG,
                 [{"t": "Accuracy, rejected as a headline", "size": 24, "color": INK, "bold": True, "after": 6},
                  {"t": "A model that answers non evaluative every single time scores 85.6 percent on "
                        "this corpus and has learned nothing at all.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])
    accent_panel(s, ML, y + 1.76, CONTENT_W, 1.56, POS,
                 [{"t": "Macro F1, the headline", "size": 24, "color": INK, "bold": True, "after": 6},
                  {"t": "Averages F1 over the three classes with equal weight, so the 152 positive rows "
                        "count as much as the 2,583 neutral ones. That same all neutral model scores 0.31 here.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 27 metrics used
    s, y = new("13 / METRICS", "The other three, and what each one catches")
    rows_m = [("Per class precision and recall",
               "Macro F1 hides which class is broken. Splitting it out revealed that positive precision "
               "is the real weakness, at 0.417."),
              ("Cohen kappa",
               "Agreement corrected for chance. The rule set reaches 0.392, which is fair agreement. "
               "Raw agreement of 59.4 percent would sound far better than it is."),
              ("Confusion matrix",
               "The only view that says which direction the errors run. A single F1 cannot tell a model "
               "that misses negatives apart from one that invents positives.")]
    for head, body in rows_m:
        textbox(s, ML, y, CONTENT_W, 0.34, [{"t": head, "size": 24, "color": PETROL, "bold": True, "after": 0}])
        textbox(s, ML, y + 0.40, CONTENT_W, 0.6,
                [{"t": body, "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])
        y += 1.24
    textbox(s, ML, y - 0.06, CONTENT_W, 0.7,
            [{"t": "Deliberately not reported: word error rate, diarization error rate, and speaker "
                   "attribution accuracy. Each needs a human made reference, and this project has none.",
              "size": 21, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 28 results bars
    s, y = new("14 / RESULTS", "Measured against 180 transcripts read by hand",
               "A stratified sample of 180 target pairs was read in context and labelled by hand. "
               "59 fall inside the held out test episodes.")
    textbox(s, ML, y, CONTENT_W, 0.34,
            [{"t": "MACRO F1 ON THE 59 HELD OUT PAIRS", "size": 21, "color": INK_FAINT,
              "mono": True, "bold": True, "after": 0}])
    y += 0.50
    bar_x, bar_w, label_w = ML + 4.5, 6.0, 4.3
    series = [("Always non evaluative", 0.177, NEU), ("Char n-gram baseline", 0.466, SIGNAL),
              ("Rule labeller", 0.566, PETROL), ("BanglaBERT pair model", None, LINE)]
    for name, val, col in series:
        textbox(s, ML, y + 0.02, label_w, 0.4, [{"t": name, "size": 22, "color": INK, "after": 0}])
        rect(s, bar_x, y, bar_w, 0.42, fill=PANEL_2, line=LINE)
        rect(s, bar_x + bar_w / 2, y, 0.008, 0.42, fill=LINE)
        if val is not None:
            rect(s, bar_x, y, bar_w * val, 0.42, fill=col)
            textbox(s, bar_x + bar_w + 0.16, y + 0.02, 1.3, 0.4,
                    [{"t": f"{val:.3f}", "size": 22, "color": INK, "mono": True, "after": 0}])
        else:
            textbox(s, bar_x + bar_w + 0.16, y + 0.02, 1.6, 0.4,
                    [{"t": "not run", "size": 22, "color": INK_FAINT, "mono": True, "after": 0}])
        y += 0.56
    textbox(s, bar_x, y - 0.06, bar_w, 0.3,
            [{"t": "0.0                                 0.5                                 1.0",
              "size": 21, "color": INK_FAINT, "mono": True, "after": 0}])
    accent_panel(s, ML, y + 0.36, CONTENT_W, 1.30, NEG,
                 [{"t": "The result that matters", "size": 24, "color": INK, "bold": True, "after": 6},
                  {"t": "The trained model scores below the rule set that generated its training labels, "
                        "0.466 against 0.566. A student trained on noisy labels will not beat its teacher "
                        "unless it can generalise past the teacher's mistakes.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 29 per class
    s, y = new("14 / RESULTS", "Rule labeller, per class, all 180 pairs")
    y = table(s, ML, y, CONTENT_W,
              [["Class", "Precision", "Recall", "F1", "Support"],
               ["negative", "0.850", "0.607", "0.708", "84"],
               ["non evaluative", "0.517", "0.484", "0.500", "64"],
               ["positive", "0.417", "0.781", "0.543", "32"],
               ["macro average", "0.594", "0.624", "0.584", "180"]],
              [1.8, 1.2, 1.2, 1.2, 1.2],
              [PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4, row_h=0.56, highlight=4)
    textbox(s, ML, y + 0.06, CONTENT_W, 0.4,
            [{"t": "Accuracy 0.594.    Cohen kappa 0.392.", "size": 22, "color": INK_SOFT, "mono": True, "after": 0}])
    accent_panel(s, ML, y + 0.66, CONTENT_W, 1.42, PETROL,
                 [{"t": "Negative precision is high at 0.850, so when the system says negative it is "
                        "usually right. Positive precision at 0.417 is where the damage sits: it calls "
                        "things positive that are not.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 30 confusion
    s, y = new("14 / RESULTS", "Baseline confusion matrix on 59 held-out pairs")
    table(s, ML, y, 8.4,
          [["", "said negative", "said non eval", "said positive"],
           ["is negative", "10", "21", "2"],
           ["is non evaluative", "3", "13", "0"],
           ["is positive", "0", "6", "4"]],
          [1.6, 1.3, 1.3, 1.3],
          [PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER], row_h=0.62)
    accent_panel(s, ML + 8.8, y, CONTENT_W - 8.8, 2.48, NEG,
                 [{"t": "Read the top right", "size": 23, "color": INK, "bold": True, "after": 6},
                  {"t": "The largest error is 21 human-negative pairs predicted as non-evaluative. "
                        "The model is missing negative judgements rather than confidently finding them.",
                   "size": 21, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 31 error modes
    s, y = new("15 / ERROR ANALYSIS", "Where the errors actually are")
    y = table(s, ML, y, 8.0,
              [["Human said", "System said", "Count"],
               ["non evaluative", "positive", "24"],
               ["negative", "non evaluative", "22"],
               ["negative", "positive", "11"],
               ["non evaluative", "negative", "9"],
               ["positive", "non evaluative", "7"]],
              [1.5, 1.5, 0.9],
              [PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT], row_h=0.52)
    accent_panel(s, ML + 8.4, MT + 1.30, CONTENT_W - 8.4, 2.7, PETROL,
                 [{"t": "The structural cause", "size": 23, "color": INK, "bold": True, "after": 6},
                  {"t": "Class imbalance. 85.6 percent of rows are non evaluative and only 91 positive "
                        "examples reach training. Far too few for a 110 million parameter model, and thin "
                        "even for logistic regression.",
                   "size": 21, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 32 cause 1
    s, y = new("15 / ERROR ANALYSIS", "Cause one: advice read as praise",
               "24 errors, the largest single class. A speaker says what should happen, and the rule set "
               "counts the good word as a judgement about the present.")
    y = quote(s, ML, y, CONTENT_W,
              "অর্থনীতি স্থিতিশীল "
              "করতে হবে",
              "The economy must be stabilised. This is a demand, not a compliment. The system reads "
              "the word for stable and answers positive.", accent=NEG, size=30, h=1.46)
    accent_panel(s, ML, y + 0.14, CONTENT_W, 1.52, POS,
                 [{"t": "The fix is short", "size": 24, "color": INK, "bold": True, "after": 6},
                  {"t": "Bangla marks obligation with a small closed set of endings, chiefly "
                        "করতে হবে, উচিত, "
                        "দরকার and প্রয়োজন. "
                        "Detecting those and suppressing the polarity claim is a rule, not a research problem.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 33 cause 2 and 3
    s, y = new("15 / ERROR ANALYSIS", "Cause two and cause three")
    textbox(s, ML, y, CONTENT_W, 0.4,
            [{"t": "Negative meaning with no cue word. 22 errors.", "size": 23, "color": INK,
              "bold": True, "after": 0}])
    y += 0.48
    y = quote(s, ML, y, CONTENT_W,
              "বিনিয়োগ বাইশ পয়েন্ট "
              "চার পার্সেন্টে "
              "নেমে গেছে",
              "Investment has come down to 22.4 percent. Clearly negative, and not one word of it is "
              "in the 104 stem cue list.", accent=NEG, size=27, h=1.34)
    textbox(s, ML, y + 0.10, CONTENT_W, 0.4,
            [{"t": "Negation carried by a suffix.", "size": 23, "color": INK, "bold": True, "after": 0}])
    quote(s, ML, y + 0.58, CONTENT_W,
          "বাজেট বাস্তবায়ন "
          "হয়নি",
          "The budget was not implemented. Bangla negates the past tense by attaching নি to the "
          "verb, so there is no separate particle for the window to find.", accent=NEG, size=27, h=1.40)

    # ---------------------------------------------------------- 34 ASR ruled out
    s, y = new("15 / ERROR ANALYSIS", "The obvious suspect, tested and ruled out",
               "The natural assumption is that recognition noise flows downstream and breaks the "
               "sentiment. It was measured.")
    y = table(s, ML, y, 8.6,
              [["Stratum", "n", "Macro F1"],
               ["Cleaner transcripts, confidence above median", "90", "0.550"],
               ["Noisier transcripts, confidence below median", "90", "0.619"],
               ["No leading truncation", "161", "0.584"],
               ["Leading truncation present", "19", "0.562"]],
              [3.2, 0.8, 1.1],
              [PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT], row_h=0.54)
    accent_panel(s, ML, y + 0.16, CONTENT_W, 1.60, PETROL,
                 [{"t": "The cleaner half scores no better than the noisier half. At 90 pairs a side that "
                        "gap sits inside the noise, so the honest reading is that transcript quality does "
                        "not separate sentiment accuracy here. The label rules, not the transcripts, are "
                        "the bottleneck.",
                   "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])

    # ---------------------------------------------------------- 35 improve 1
    s, y = new("16 / IMPROVING IT", "Ranked by expected gain per hour of work")
    plan = [("1", "Label 800 to 1,000 pairs by hand, with two annotators",
             "The single biggest lever. It removes the ceiling rather than raising it, because the model "
             "is no longer bounded by the teacher. Six members at 150 pairs each is one working session."),
            ("2", "Correct the two recognition settings",
             "Raise the repetition penalty above 1.0 to stop the decoder loops. Cut chunks with a 2 to 3 "
             "second overlap to remove the truncated first word from 11.3 percent of utterances."),
            ("3", "Separate prescription from evaluation",
             "The largest single error class, 24 of 73. A short rule over a closed set of obligation "
             "endings, not a research problem.")]
    for num, head, body in plan:
        textbox(s, ML, y, 0.5, 0.4, [{"t": num, "size": 24, "color": SIGNAL, "bold": True, "mono": True, "after": 0}])
        textbox(s, ML + 0.55, y, CONTENT_W - 0.55, 0.4,
                [{"t": head, "size": 24, "color": INK, "bold": True, "after": 0}])
        textbox(s, ML + 0.55, y + 0.44, CONTENT_W - 0.55, 0.7,
                [{"t": body, "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])
        y += 1.42

    # ---------------------------------------------------------- 36 improve 2
    s, y = new("16 / IMPROVING IT", "And then these three")
    plan2 = [("4", "Handle the suffix negation",
              "Matching a verb final নি catches a class of sign flips that currently go "
              "straight into the training labels."),
             ("5", "Train the sentence pair model",
              "The training code is written and the checkpoint loads. It reads whole phrases rather than "
              "character n-grams, so it can learn that a fall in investment is bad without that phrase "
              "sitting in a cue list."),
             ("6", "Widen the corpus",
              "26 episodes is a small corpus, and the positive class at 152 rows is the thinnest part "
              "of it. The pipeline runs end to end on new material without a code change.")]
    for num, head, body in plan2:
        textbox(s, ML, y, 0.5, 0.4, [{"t": num, "size": 24, "color": SIGNAL, "bold": True, "mono": True, "after": 0}])
        textbox(s, ML + 0.55, y, CONTENT_W - 0.55, 0.4,
                [{"t": head, "size": 24, "color": INK, "bold": True, "after": 0}])
        textbox(s, ML + 0.55, y + 0.44, CONTENT_W - 0.55, 0.8,
                [{"t": body, "size": 22, "color": INK_SOFT, "line": 1.2, "after": 0}])
        y += 1.46

    # ---------------------------------------------------------- 37 visual evidence: learning curve
    figures = build_result_figures(Path(__file__).parent)
    s, y = new("17 / VISUAL EVIDENCE", "Learning curve for the trained baseline",
               "The saved Logistic Regression run has no neural epoch-loss history. This reproducible curve shows how macro F1 changes as more silver-labelled training data is added.")
    learning_h = 3.70
    learning_w = 8.50
    s.shapes.add_picture(str(figures["learning"]), Inches((SLIDE_W - learning_w) / 2), Inches(y),
                         height=Inches(learning_h))
    textbox(s, ML, 6.72, CONTENT_W, 0.30,
            [{"t": "Dashed line: final held-out human test macro F1 = 0.466. It is not used for model selection.",
              "size": 21, "color": INK_FAINT, "mono": True, "after": 0}])

    # ---------------------------------------------------------- 38 visual evidence: per-class plot
    s, y = new("17 / VISUAL EVIDENCE", "What the held-out score is made of",
               "The baseline was evaluated on 59 human-reviewed target pairs from four unseen episodes.")
    class_h = 3.90
    class_w = 7.80
    s.shapes.add_picture(str(figures["class_metrics"]), Inches((SLIDE_W - class_w) / 2), Inches(y),
                         height=Inches(class_h))
    textbox(s, ML, 6.72, CONTENT_W, 0.30,
            [{"t": "Negative precision is strong, but negative recall is low; non-evaluative recall is high but imprecise.",
              "size": 21, "color": INK_FAINT, "mono": True, "after": 0}])

    # ---------------------------------------------------------- 39 visual evidence: confusion variants
    s, y = new("17 / VISUAL EVIDENCE", "Confusion matrix: two colour views",
               "Rows are human labels and columns are baseline predictions. Counts and row-normalized recall show different characteristics of the same result.")
    confusion_h = 3.75
    confusion_w = 8.75
    s.shapes.add_picture(str(figures["confusion"]), Inches((SLIDE_W - confusion_w) / 2), Inches(y - 0.05),
                         height=Inches(confusion_h))
    textbox(s, ML, 6.72, CONTENT_W, 0.30,
            [{"t": "The largest error is human negative predicted as non-evaluative: 21 of 59 test pairs.",
              "size": 21, "color": INK_FAINT, "mono": True, "after": 0}])

    try:
        prs.save(str(path))
    except PermissionError:
        # The deck is open in PowerPoint, which locks it. Write alongside it
        # rather than throwing the build away.
        alt = path.with_name(path.stem + "_new" + path.suffix)
        prs.save(str(alt))
        print(f"{path.name} is open and locked, wrote {alt.name} instead")
        return len(prs.slides._sldIdLst), alt
    return len(prs.slides._sldIdLst), path


if __name__ == "__main__":
    out = Path(__file__).with_name("Bangla_Talk_Show_Pipeline.pptx")
    count, written = build(out)
    print(f"wrote {written.name}: {count} slides at 13.333 x 7.5 in (16:9)")

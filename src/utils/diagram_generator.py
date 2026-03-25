"""
src/utils/diagram_generator.py — Medical Anatomy Diagram Generator
====================================================================
Generates split-pane clinical anatomy diagrams using matplotlib.

Diagrams dispatched by condition keyword:
  tear_film   → Tear film layers (L) + gland anatomy (R)
  cornea      → Corneal layer cross-section (L) + anterior segment (R)
  glaucoma    → Aqueous outflow pathway (L) + optic disc cupping (R)
  cataract    → Crystalline lens structure (L) + sagittal eye section (R)
  retina      → Retinal layer OCT cross-section (L) + fundus schematic (R)
  default     → Full eye cross-section (L) + anterior segment detail (R)

Visual Sync Contract
--------------------
  Every term in the clinical text that describes anatomy MUST appear as a
  labeled element in the matching diagram panel — enabling a clinician to
  glance between text and diagram without searching.

SILENT FAIL CONTRACT
--------------------
  generate_clinical_diagram() never raises.
  Returns PNG bytes on success, None on any failure.
  Caller (app.py) skips the image element when None.
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive, thread-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch

# ── Clinical Color Palette ────────────────────────────────────────────────────
C: dict[str, str] = {
    # ── Tear film ─────────────────────────────────────────────────────────
    "lipid":        "#F5C518",   # gold / yellow — lipid/meibomian
    "aqueous":      "#4A90D9",   # mid blue — aqueous layer
    "mucin":        "#9B59B6",   # purple — mucin/goblet
    # ── Corneal layers ────────────────────────────────────────────────────
    "epithelium":   "#E57373",   # soft red-pink
    "bowman":       "#FFAB91",   # light orange
    "stroma":       "#90CAF9",   # light blue (water content)
    "descemet":     "#A5D6A7",   # light green
    "endothelium":  "#FFCC80",   # peach / amber
    # ── Eye structures ────────────────────────────────────────────────────
    "sclera":       "#F5F5F5",   # near-white
    "choroid":      "#C62828",   # dark red (vascular)
    "retina_bg":    "#C8E6C9",   # pale green
    "vitreous":     "#E3F2FD",   # very light blue
    "lens":         "#FFF9C4",   # pale yellow
    "lens_nuc":     "#FFD54F",   # amber — nuclear zone
    "iris":         "#3949AB",   # indigo
    "cornea_body":  "#BBDEFB",   # cornea fill
    "optic_nerve":  "#E65100",   # burnt orange
    "optic_disc":   "#FFF8E1",   # cream
    "optic_cup":    "#FFB300",   # amber
    "aqueous_fl":   "#BBDEFB",   # anterior chamber fill
    "trabecular":   "#4CAF50",   # green
    "schlemm":      "#1565C0",   # dark blue
    "ciliary":      "#AB47BC",   # purple
    # ── Glands / external ─────────────────────────────────────────────────
    "lacrimal":     "#EF9A9A",   # pink
    "meibomian":    "#FFCA28",   # amber yellow
    "skin":         "#FDDCBB",   # eyelid skin
    # ── Retinal layers ────────────────────────────────────────────────────
    "rnfl":         "#00897B",   # teal
    "gcl":          "#1E88E5",   # blue
    "ipl":          "#5E35B1",   # deep purple
    "inl":          "#D81B60",   # magenta
    "opl":          "#FB8C00",   # orange
    "onl":          "#7CB342",   # green
    "isos":         "#E64A19",   # deep orange
    "rpe":          "#6D4C41",   # brown
    "bruch":        "#455A64",   # blue-grey
    # ── Layout ────────────────────────────────────────────────────────────
    "bg":           "#F8F9FA",
    "panel_bg":     "#FFFFFF",
    "border":       "#DEE2E6",
    "text":         "#1A2530",
    "label":        "#1B4F72",
    "line":         "#546E7A",
    "title":        "#0D1B2A",
    "highlight":    "#E53935",
    "arrow":        "#0D47A1",
}

_FIG_W = 12.4   # figure width  (inches)
_FIG_H =  5.0   # figure height (inches)
_DPI   = 110


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def generate_clinical_diagram(
    condition_entity: str,
    query: str = "",
    answer_text: str = "",
) -> Optional[bytes]:
    """
    Return a split-pane PNG diagram (bytes) matched to `condition_entity`.

    Keyword matching falls through a priority chain; the default full-eye
    anatomy diagram fires when nothing specific matches.

    Returns None on any failure — never raises.
    """
    try:
        sig = (condition_entity + " " + query).lower()

        if _match(sig, ["tear film", "dry eye", "meibomian", "lacrimal",
                         "blepharitis", "sjögren", "sjogren", "aqueous deficiency"]):
            return _tear_film()

        if _match(sig, ["keratitis", "corneal ulcer", "corneal abrasion",
                         "corneal scar", "corneal infiltrate", "cornea"]):
            return _corneal_layers()

        if _match(sig, ["glaucoma", "iop", "intraocular pressure", "trabecular",
                         "schlemm", "optic disc", "cup-to-disc", "cup to disc",
                         "rnfl thinning"]):
            return _glaucoma()

        if _match(sig, ["cataract", "iol", "intraocular lens", "phaco",
                         "nuclear sclerosis", "lens opacity"]):
            return _cataract()

        if _match(sig, ["retina", "macula", "fovea", "diabetic retinopathy",
                         "armd", "amd", "macular", "detachment", "rpe",
                         "vitreous hemorrhage", "epiretinal"]):
            return _retinal_layers()

        return _eye_anatomy()   # default fallback
    except Exception:
        return None


def _match(sig: str, keywords: list[str]) -> bool:
    return any(k in sig for k in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _fig_axes(title_l: str, title_r: str):
    """Create the 1×2 split-pane figure with styled axes."""
    fig = plt.figure(
        figsize=(_FIG_W, _FIG_H), dpi=_DPI, facecolor=C["bg"]
    )
    fig.subplots_adjust(
        left=0.03, right=0.97, top=0.87, bottom=0.05, wspace=0.10
    )
    ax_l = fig.add_subplot(1, 2, 1)
    ax_r = fig.add_subplot(1, 2, 2)

    for ax, title in [(ax_l, title_l), (ax_r, title_r)]:
        ax.set_facecolor(C["panel_bg"])
        for sp in ax.spines.values():
            sp.set_edgecolor(C["border"])
            sp.set_linewidth(1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            title, fontsize=9.5, fontweight="bold",
            color=C["title"], pad=6, fontfamily="DejaVu Sans",
        )
    return fig, ax_l, ax_r


def _lbl(ax, x, y, txt, fs=8.0, bold=False,
         color=None, ha="left", va="center"):
    ax.text(
        x, y, txt,
        ha=ha, va=va,
        fontsize=fs,
        color=color or C["label"],
        fontweight="bold" if bold else "normal",
        fontfamily="DejaVu Sans",
        transform=ax.transData,
    )


def _leader(ax, x0, y0, x1, y1, color=None, lw=0.85):
    """Thin grey leader / connector line (no arrowhead)."""
    ax.plot([x0, x1], [y0, y1], "-",
            color=color or C["line"], lw=lw, alpha=0.80)


def _png(fig) -> bytes:
    """Render figure to PNG bytes and close."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _rect(ax, x, y, w, h, color, edge="white", lw=1.2, alpha=0.88, zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor=edge,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 1 — TEAR FILM
# ─────────────────────────────────────────────────────────────────────────────
def _tear_film() -> bytes:
    fig, ax_l, ax_r = _fig_axes(
        "Panel A — Tear Film Cross-Section (not to scale)",
        "Panel B — Ocular Surface Gland Anatomy",
    )
    _tear_layers(ax_l)
    _tear_glands(ax_r)
    return _png(fig)


def _tear_layers(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # (y_bottom, height, color, number, label, thickness_note)
    layers = [
        (0.5, 1.9, C["epithelium"], "0.",
         "Corneal Epithelium",         "~50 µm (non-keratinised)"),
        (2.5, 0.7, C["mucin"],       "I.",
         "Mucin Layer  (Glycocalyx)",  "~0.2–0.5 µm"),
        (3.3, 3.1, C["aqueous"],     "II.",
         "Aqueous Layer",              "~7–8 µm  (major volume)"),
        (6.5, 0.85, C["lipid"],      "III.",
         "Lipid Layer  (Meibomian)",   "~0.1–0.2 µm"),
    ]

    for y0, h, col, num, name, thick in layers:
        _rect(ax, 1.1, y0, 4.3, h, col)
        mid = y0 + h / 2
        _leader(ax, 5.4, mid, 5.9, mid)
        _lbl(ax, 6.0, mid + 0.18, f"{num}  {name}", fs=8.0, bold=True)
        _lbl(ax, 6.0, mid - 0.30, thick, fs=7.0, color="#546E7A")

    # Air interface dashed line
    ax.plot([1.1, 5.4], [7.35, 7.35], "--", color=C["line"], lw=0.8, alpha=0.7)
    _lbl(ax, 3.2, 7.65, "Air Interface", fs=7.8, color="#546E7A", ha="center")

    # Double-headed bracket for total tear film thickness
    ax.annotate("", xy=(0.7, 2.5), xytext=(0.7, 7.35),
                arrowprops=dict(arrowstyle="<->", color=C["highlight"], lw=1.2))
    _lbl(ax, 0.05, 4.95, "≈8 µm\ntotal", fs=7.2, color=C["highlight"],
         ha="left", va="center")

    # Axis labels
    _lbl(ax, 0.2, 1.5,  "CORNEA",    fs=7.5, color=C["line"],  bold=False, ha="left")
    _lbl(ax, 0.2, 5.0,  "TEAR\nFILM", fs=7.5, color=C["label"], bold=True,  ha="left")
    _lbl(ax, 3.2, 9.5,  "← Anterior (air-facing) surface →",
         fs=7.5, color="#546E7A", ha="center")


def _tear_glands(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Eye globe
    globe = Ellipse((5, 5), 8.2, 5.2,
                     facecolor=C["vitreous"], edgecolor=C["border"], lw=1.3)
    ax.add_patch(globe)

    # Iris
    iris = Ellipse((5, 5), 3.0, 2.8, facecolor=C["iris"],
                    edgecolor=C["border"], lw=1.0, alpha=0.85)
    ax.add_patch(iris)
    pupil = Ellipse((5, 5), 1.2, 1.2, facecolor="#0D0D0D", edgecolor="none")
    ax.add_patch(pupil)

    # Upper eyelid
    upper = FancyBboxPatch((0.8, 6.9), 8.4, 1.5,
                            boxstyle="round,pad=0.25",
                            facecolor=C["skin"], edgecolor="#B0744A", lw=1.1)
    ax.add_patch(upper)

    # Lower eyelid
    lower = FancyBboxPatch((0.8, 1.6), 8.4, 1.3,
                            boxstyle="round,pad=0.25",
                            facecolor=C["skin"], edgecolor="#B0744A", lw=1.1)
    ax.add_patch(lower)

    # Meibomian gland orifices along upper lid margin
    for xm in np.linspace(1.8, 8.4, 13):
        ax.plot([xm, xm], [6.92, 7.10], "-", color=C["meibomian"], lw=1.6, alpha=0.9)

    # Lacrimal gland (upper-outer / temporal corner)
    lac = FancyBboxPatch((7.9, 7.6), 1.6, 1.0,
                          boxstyle="round,pad=0.1",
                          facecolor=C["lacrimal"], edgecolor="#C62828", lw=1.0)
    ax.add_patch(lac)
    _lbl(ax, 8.7, 8.1, "Lacrimal\nGland", fs=7.5, bold=True,
         color="#7B241C", ha="center")

    # Arrow: lacrimal gland → corneal surface
    ax.annotate("", xy=(5, 7.2), xytext=(8.0, 7.9),
                arrowprops=dict(arrowstyle="-|>", color="#C62828",
                                lw=1.0, connectionstyle="arc3,rad=-0.25"))

    # Punctum (inner canthus)
    ax.plot(0.95, 5.0, "o", ms=5.5, color="#C62828", zorder=5)
    _leader(ax, 0.95, 5.0, 0.1, 3.8)
    _lbl(ax, 0.05, 3.5, "Punctum\n(drainage)", fs=7.2, color="#7B241C")

    # Meibomian label
    _leader(ax, 5, 7.1, 5, 7.55)
    _lbl(ax, 5, 7.65, "Meibomian Gland Orifices  (Lipid layer secretion)",
         fs=7.5, bold=True, color="#7B4F0B", ha="center")

    # Drainage note
    _lbl(ax, 5, 0.55,
         "▼  Tear drainage:  Punctum  →  Canaliculus  →  Nasolacrimal duct",
         fs=7.5, color="#546E7A", ha="center")

    # Goblet cell label (on corneal surface)
    _lbl(ax, 5, 9.3,
         "Goblet cells (conjunctiva) secrete mucin → anchors aqueous layer",
         fs=7.2, color="#4A148C", ha="center")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 2 — CORNEAL LAYERS / KERATITIS
# ─────────────────────────────────────────────────────────────────────────────
def _corneal_layers() -> bytes:
    fig, ax_l, ax_r = _fig_axes(
        "Panel A — Corneal Layers Cross-Section  (not to scale)",
        "Panel B — Anterior Eye Anatomy",
    )
    _draw_corneal_cross_section(ax_l)
    _draw_anterior_segment(ax_r)
    return _png(fig)


def _draw_corneal_cross_section(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Layer definitions: (y_bottom, height_px, color, layer#, name, thickness)
    # Heights scaled ~proportionally to real µm thicknesses (total ~550 µm)
    layers = [
        (0.6,  0.75, C["endothelium"],  "5.",
         "Endothelium",           "~5 µm  (single-cell mosaic, non-regenerating)"),
        (1.4,  0.45, C["descemet"],     "4.",
         "Descemet's Membrane",   "~10 µm  (thickens with age)"),
        (1.9,  5.10, C["stroma"],       "3.",
         "Stroma",                "~500 µm  ≈ 90% of corneal thickness"),
        (7.1,  0.45, C["bowman"],       "2.",
         "Bowman's Layer",        "~10 µm  (acellular — does NOT regenerate)"),
        (7.6,  0.95, C["epithelium"],   "1.",
         "Epithelium",            "~50 µm  (5–6 cell layers, regenerates in 7 days)"),
    ]

    for y0, h, col, num, name, thick in layers:
        _rect(ax, 0.9, y0, 4.5, h, col)
        mid = y0 + h / 2
        _leader(ax, 5.4, mid, 5.8, mid)
        _lbl(ax, 5.85, mid + 0.18, f"{num}  {name}", fs=8.0, bold=True)
        _lbl(ax, 5.85, mid - 0.32, thick, fs=6.8, color="#546E7A")

    # Total thickness bracket
    ax.annotate("", xy=(0.55, 0.6), xytext=(0.55, 8.55),
                arrowprops=dict(arrowstyle="<->", color=C["highlight"], lw=1.2))
    _lbl(ax, 0.0, 4.6, "~550\nµm", fs=7.2, color=C["highlight"],
         ha="left", va="center")

    # Surface labels
    _lbl(ax, 3.1, 9.2, "← Anterior (air-facing) surface →",
         fs=7.8, bold=True, color=C["label"], ha="center")
    _lbl(ax, 3.1, 0.2, "← Posterior (aqueous-facing) surface →",
         fs=7.8, bold=True, color=C["label"], ha="center")


def _draw_anterior_segment(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Sclera globe
    sclera = Ellipse((5, 5), 9.0, 9.0,
                      facecolor=C["sclera"], edgecolor="#BDBDBD", lw=1.4)
    ax.add_patch(sclera)

    # Vitreous fill
    vit = Ellipse((5, 5), 7.6, 7.6,
                   facecolor=C["vitreous"], edgecolor="none", alpha=0.5)
    ax.add_patch(vit)

    # Cornea wedge (left side, nasal view)
    co = mpatches.Wedge((5, 5), 4.45, 153, 207, width=0.50,
                          facecolor=C["cornea_body"], edgecolor=C["border"], lw=1.2)
    ax.add_patch(co)

    # Aqueous in anterior chamber
    ac = mpatches.Wedge((5, 5), 3.9, 153, 207,
                          facecolor=C["aqueous_fl"], edgecolor="none", alpha=0.45)
    ax.add_patch(ac)

    # Iris
    iris = mpatches.Wedge((5, 5), 3.35, 153, 207, width=0.50,
                            facecolor=C["iris"], edgecolor="none", alpha=0.90)
    ax.add_patch(iris)

    # Lens
    lens = Ellipse((5.60, 5), 1.0, 2.3,
                    facecolor=C["lens"], edgecolor=C["border"], lw=1.0, alpha=0.90)
    ax.add_patch(lens)

    # Optic nerve stub
    on = FancyBboxPatch((8.55, 4.55), 0.80, 0.90,
                         boxstyle="round,pad=0.05",
                         facecolor=C["optic_nerve"], edgecolor="none", alpha=0.85)
    ax.add_patch(on)

    # Retina arc
    ret = mpatches.Wedge((5, 5), 4.0, 0, 360, width=0.22,
                          facecolor=C["retina_bg"], edgecolor="none", alpha=0.60)
    ax.add_patch(ret)

    # Labels
    _anns = [
        (1.25, 5.0,  0.15, 5.0,  "Cornea\n(5 layers)"),
        (3.80, 4.20, 2.60, 3.00, "Iris"),
        (3.30, 5.0,  2.00, 5.60, "Anterior\nChamber"),
        (5.60, 5.0,  5.60, 7.80, "Lens"),
        (4.60, 2.20, 3.40, 1.20, "Retina"),
        (8.55, 5.0,  9.50, 3.60, "Optic\nNerve"),
    ]
    for x0, y0, xl, yl, txt in _anns:
        _leader(ax, x0, y0, xl, yl)
        _lbl(ax, xl, yl, txt, fs=7.8, bold=True,
             ha="right" if xl < x0 else "left")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 3 — GLAUCOMA
# ─────────────────────────────────────────────────────────────────────────────
def _glaucoma() -> bytes:
    fig, ax_l, ax_r = _fig_axes(
        "Panel A — Aqueous Outflow Pathway",
        "Panel B — Optic Disc Cupping (Glaucomatous vs. Normal)",
    )
    _draw_aqueous_outflow(ax_l)
    _draw_optic_disc_comparison(ax_r)
    return _png(fig)


def _draw_aqueous_outflow(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Cornea wedge
    co = mpatches.Wedge((3.5, 5), 4.2, 138, 222, width=0.52,
                          facecolor=C["cornea_body"], edgecolor=C["border"], lw=1.2)
    ax.add_patch(co)

    # Anterior chamber (aqueous fill)
    ac = mpatches.Wedge((3.5, 5), 3.65, 138, 222,
                          facecolor=C["aqueous_fl"], edgecolor="none", alpha=0.50)
    ax.add_patch(ac)

    # Iris
    iris = mpatches.Wedge((3.5, 5), 3.1, 148, 212, width=0.55,
                            facecolor=C["iris"], edgecolor="none", alpha=0.90)
    ax.add_patch(iris)

    # Ciliary body (source of aqueous)
    cil = mpatches.Wedge((3.5, 5), 3.6, 215, 240, width=0.55,
                           facecolor=C["ciliary"], edgecolor="none", alpha=0.85)
    ax.add_patch(cil)

    # Trabecular meshwork block
    _rect(ax, 5.75, 6.30, 0.95, 0.60, C["trabecular"], alpha=0.95)
    # Schlemm's canal block
    _rect(ax, 6.72, 6.30, 0.55, 0.60, C["schlemm"], alpha=0.90)

    # Aqueous flow path (arrows)
    # ciliary body → posterior chamber → through pupil → anterior chamber → TM
    ax.annotate("", xy=(5.5, 5.35), xytext=(3.2, 3.65),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                lw=1.5, connectionstyle="arc3,rad=-0.30"))
    ax.annotate("", xy=(5.9, 6.30), xytext=(5.5, 5.35),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"], lw=1.5))

    # Labels
    _leader(ax, 5.75, 6.60, 5.00, 7.50)
    _lbl(ax, 3.70, 7.65, "Trabecular\nMeshwork", fs=8.0, bold=True, color="#1B5E20")

    _leader(ax, 6.72, 6.60, 7.50, 7.50)
    _lbl(ax, 7.55, 7.65, "Schlemm's\nCanal", fs=8.0, bold=True, color="#0D47A1")

    _leader(ax, 3.2, 3.65, 2.00, 2.40)
    _lbl(ax, 0.10, 2.05, "Ciliary Body\n(aqueous production)", fs=7.8,
         bold=True, color="#4A148C")

    _lbl(ax, 4.80, 5.10, "Aqueous\nflow →", fs=8.0, color=C["arrow"], ha="left")

    # IOP warning
    ax.plot([0.8, 9.2], [8.4, 8.4], "--", color=C["highlight"], lw=0.9, alpha=0.55)
    _lbl(ax, 5.0, 8.75,
         "⚠  IOP rises when trabecular outflow is obstructed  (Glaucoma)",
         fs=8.2, bold=True, color=C["highlight"], ha="center")


def _draw_optic_disc_comparison(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # ── Left sub-panel: Normal disc ───────────────────────────────────────
    # Fundus background
    fn = Ellipse((2.5, 5), 4.2, 4.2,
                  facecolor="#FFEFD5", edgecolor="#D2691E", lw=1.2)
    ax.add_patch(fn)
    # Normal disc
    dn = Ellipse((2.5, 5), 1.8, 2.0,
                  facecolor=C["optic_disc"], edgecolor="#E0C060", lw=1.2)
    ax.add_patch(dn)
    # Healthy rim (pink neural rim — wide)
    rn = mpatches.Wedge((2.5, 5), 0.90, 0, 360, width=0.45,
                          facecolor="#FFCDD2", edgecolor="none", alpha=0.85)
    ax.add_patch(rn)
    # Small cup (CDR ~0.3)
    cn = Ellipse((2.5, 5), 0.9, 1.0,
                  facecolor="#FFD54F", edgecolor="#FFA000", lw=1.0, alpha=0.85)
    ax.add_patch(cn)
    _lbl(ax, 2.5, 7.5, "NORMAL", fs=8.5, bold=True,
         color="#1B5E20", ha="center")
    _lbl(ax, 2.5, 7.05, "CDR ≈ 0.3", fs=7.5, color="#546E7A", ha="center")

    # ── Right sub-panel: Glaucomatous disc ────────────────────────────────
    fg = Ellipse((7.5, 5), 4.2, 4.2,
                  facecolor="#FFEFD5", edgecolor="#D2691E", lw=1.2)
    ax.add_patch(fg)
    # Glaucomatous disc
    dg = Ellipse((7.5, 5), 1.9, 2.1,
                  facecolor=C["optic_disc"], edgecolor="#E0C060", lw=1.2)
    ax.add_patch(dg)
    # Thinned neural rim (narrower pink zone)
    rg = mpatches.Wedge((7.5, 5), 0.95, 0, 360, width=0.22,
                          facecolor="#FFCDD2", edgecolor="none", alpha=0.75)
    ax.add_patch(rg)
    # Large cup (CDR ~0.7)
    cg = Ellipse((7.5, 5), 1.42, 1.56,
                  facecolor=C["optic_cup"], edgecolor="#E65100", lw=1.2, alpha=0.90)
    ax.add_patch(cg)
    # RNFL defect markers (superior + inferior poles)
    for angle_d in [90, 270]:
        theta = np.radians(angle_d)
        rx = 7.5 + 0.73 * np.cos(theta)
        ry = 5.0 + 0.80 * np.sin(theta)
        ax.plot(rx, ry, "D", ms=6, color=C["highlight"], alpha=0.90, zorder=5)

    _lbl(ax, 7.5, 7.5, "GLAUCOMA", fs=8.5, bold=True,
         color=C["highlight"], ha="center")
    _lbl(ax, 7.5, 7.05, "CDR ≈ 0.7", fs=7.5, color=C["highlight"], ha="center")

    # Shared annotation: RNFL thinning
    _lbl(ax, 5.0, 2.35,
         "◆  RNFL thinning  (superior & inferior poles — earliest sign)",
         fs=7.8, bold=True, color=C["highlight"], ha="center")

    # Divider line
    ax.plot([5.0, 5.0], [2.8, 9.2], "--", color=C["border"], lw=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 4 — CATARACT / LENS
# ─────────────────────────────────────────────────────────────────────────────
def _cataract() -> bytes:
    fig, ax_l, ax_r = _fig_axes(
        "Panel A — Crystalline Lens Structure",
        "Panel B — Sagittal Eye Section (Lens Position)",
    )
    _draw_lens(ax_l)
    _draw_eye_sagittal(ax_r)
    return _png(fig)


def _draw_lens(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Capsule (outermost)
    cap = Ellipse((5, 5), 5.8, 9.0,
                   facecolor="#E3F2FD", edgecolor="#90CAF9", lw=1.4, alpha=0.55)
    ax.add_patch(cap)

    # Cortex
    cortex = Ellipse((5, 5), 5.2, 8.3,
                      facecolor=C["lens"], edgecolor="#FFD54F", lw=1.2, alpha=0.75)
    ax.add_patch(cortex)

    # Epinucleus
    epinuc = Ellipse((5, 5), 3.8, 5.8,
                      facecolor="#FFE082", edgecolor="#FFB300", lw=1.2, alpha=0.80)
    ax.add_patch(epinuc)

    # Nucleus (dense — site of nuclear sclerosis)
    nuc = Ellipse((5, 5), 2.5, 3.8,
                   facecolor=C["lens_nuc"], edgecolor="#F57F17", lw=1.4, alpha=0.88)
    ax.add_patch(nuc)

    # Nuclear sclerosis opacity (darker central smear — graded cataract)
    ns = Ellipse((5, 5), 1.6, 2.4,
                  facecolor="#C17900", edgecolor="none", alpha=0.45)
    ax.add_patch(ns)

    # Labels
    _anns = [
        (5.0, 9.3,  6.5, 9.7,  "Anterior Capsule"),
        (7.6, 6.8,  8.4, 6.8,  "Cortex"),
        (6.9, 5.0,  8.4, 5.0,  "Epinucleus"),
        (6.2, 5.0,  8.4, 3.6,  "Nucleus"),
        (5.5, 5.0,  8.4, 2.2,  "Nuclear Sclerosis\n(Cataract opacity)"),
        (5.0, 0.7,  6.5, 0.3,  "Posterior Capsule"),
        (2.4, 8.5,  1.0, 8.9,  "Epithelium\n(ant. surface)"),
    ]
    for x0, y0, xl, yl, txt in _anns:
        _leader(ax, x0, y0, xl, yl)
        _lbl(ax, xl, yl, txt, fs=7.8, bold=True,
             ha="right" if xl < x0 else "left")

    _lbl(ax, 5, 9.8, "← Anterior", fs=7.2, color="#546E7A", ha="center")
    _lbl(ax, 5, 0.0, "Posterior →", fs=7.2, color="#546E7A", ha="center")
    _lbl(ax, 5, 9.5,
         "Phacoemulsification removes nucleus + cortex — capsule retained for IOL",
         fs=7.2, color="#1B4F72", ha="center")


def _draw_eye_sagittal(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    sclera = Ellipse((5, 5), 9.0, 9.0,
                      facecolor=C["sclera"], edgecolor="#BDBDBD", lw=1.4)
    ax.add_patch(sclera)

    choroid = mpatches.Wedge((5, 5), 4.5, 0, 360, width=0.28,
                              facecolor=C["choroid"], edgecolor="none", alpha=0.60)
    ax.add_patch(choroid)

    ret = mpatches.Wedge((5, 5), 4.22, 0, 360, width=0.28,
                          facecolor=C["retina_bg"], edgecolor="none", alpha=0.65)
    ax.add_patch(ret)

    vit = Ellipse((5, 5), 7.6, 7.6,
                   facecolor=C["vitreous"], edgecolor="none", alpha=0.45)
    ax.add_patch(vit)

    co = mpatches.Wedge((5, 5), 4.5, 153, 207, width=0.50,
                          facecolor=C["cornea_body"], edgecolor=C["border"], lw=1.2)
    ax.add_patch(co)

    # Lens (natural — shown opacified)
    lens_nat = Ellipse((5.58, 5), 1.0, 2.4,
                        facecolor=C["lens_nuc"], edgecolor=C["lens"], lw=1.3, alpha=0.85)
    ax.add_patch(lens_nat)

    # IOL overlay (dashed — post-phaco position)
    iol = Ellipse((5.58, 5), 1.5, 3.0,
                   facecolor="none", edgecolor="#E53935",
                   lw=1.8, linestyle="--", alpha=0.80)
    ax.add_patch(iol)

    on = FancyBboxPatch((8.55, 4.58), 0.80, 0.85,
                         boxstyle="round,pad=0.05",
                         facecolor=C["optic_nerve"], edgecolor="none", alpha=0.85)
    ax.add_patch(on)

    _anns = [
        (1.25, 5.0,  0.2,  5.0,  "Cornea"),
        (5.58, 5.0,  5.58, 8.3,  "Lens\n(IOL = red dash)"),
        (8.55, 5.0,  9.55, 3.60, "Optic\nNerve"),
        (6.80, 5.2,  7.60, 7.60, "Retina"),
        (4.80, 4.20, 3.60, 3.00, "Vitreous"),
    ]
    for x0, y0, xl, yl, txt in _anns:
        _leader(ax, x0, y0, xl, yl)
        _lbl(ax, xl, yl, txt, fs=7.8, bold=True,
             ha="right" if xl < x0 else "left")

    _lbl(ax, 5, 0.45,
         "IOL (---) placed in capsular bag after phacoemulsification",
         fs=7.2, color="#B71C1C", bold=True, ha="center")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 5 — RETINAL LAYERS
# ─────────────────────────────────────────────────────────────────────────────
def _retinal_layers() -> bytes:
    fig, ax_l, ax_r = _fig_axes(
        "Panel A — Retinal Layers  (OCT Cross-Section, not to scale)",
        "Panel B — Posterior Fundus Schematic",
    )
    _draw_retinal_stack(ax_l)
    _draw_fundus(ax_r)
    return _png(fig)


def _draw_retinal_stack(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # (y_bottom, height, color, abbrev, full name, thickness)
    layers = [
        (9.35, 0.35, C["vitreous"],   "—",   "Vitreous Humor",                   ""),
        (8.85, 0.45, "#E0E0E0",       "ILM", "Inner Limiting Membrane",          "~0.01 µm"),
        (8.20, 0.60, C["rnfl"],       "RNFL","Retinal Nerve Fiber Layer",        "~100 µm peripapillary"),
        (7.45, 0.70, C["gcl"],        "GCL", "Ganglion Cell Layer",              "~10 µm"),
        (6.65, 0.75, C["ipl"],        "IPL", "Inner Plexiform Layer",            "~40 µm"),
        (5.80, 0.80, C["inl"],        "INL", "Inner Nuclear Layer",              "~40 µm"),
        (4.90, 0.85, C["opl"],        "OPL", "Outer Plexiform Layer",            "~10 µm"),
        (4.00, 0.85, C["onl"],        "ONL", "Outer Nuclear Layer",              "~50 µm"),
        (3.10, 0.85, C["isos"],       "IS/OS","Photoreceptor Inner/Outer Segments","~30 µm"),
        (2.20, 0.85, C["rpe"],        "RPE", "Retinal Pigment Epithelium",       "~10 µm"),
        (1.35, 0.80, C["bruch"],      "BrM", "Bruch's Membrane",                 "~2–4 µm"),
        (0.45, 0.85, C["choroid"],    "—",   "Choroid  (vascular)",              "~300 µm"),
    ]

    for y0, h, col, abbr, name, thick in layers:
        _rect(ax, 0.4, y0, 3.8, h, col)
        mid = y0 + h / 2

        # Abbreviation inside the bar (if short enough)
        if abbr != "—":
            _lbl(ax, 2.3, mid, abbr, fs=7.5, bold=True,
                 color="white", ha="center")

        # Leader to label
        _leader(ax, 4.2, mid, 4.6, mid)
        _lbl(ax, 4.65, mid + 0.13, name, fs=7.0, bold=(col != C["vitreous"]))
        if thick:
            _lbl(ax, 9.85, mid, thick, fs=6.2, color="#546E7A", ha="right")

    # Light direction arrow
    ax.annotate("", xy=(0.2, 9.8), xytext=(0.2, 8.7),
                arrowprops=dict(arrowstyle="-|>", color="#FFB300", lw=1.5))
    _lbl(ax, 0.3, 9.3, "Light\n↓", fs=7.2, color="#FFB300", ha="left")


def _draw_fundus(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Fundus background (orange-red retinal reflex)
    fundus = Ellipse((5, 5), 9.0, 9.0,
                      facecolor="#FFEFD5", edgecolor="#D2691E", lw=1.5)
    ax.add_patch(fundus)

    # Macula (central avascular zone — ~4.5 mm)
    mac = Ellipse((5, 5), 3.8, 3.8,
                   facecolor="#FFE082", edgecolor="#FFA000", lw=1.2, alpha=0.55)
    ax.add_patch(mac)

    # Foveal avascular zone
    faz = Ellipse((5, 5), 1.5, 1.5,
                   facecolor="#FF8F00", edgecolor="#FF6D00", lw=1.3, alpha=0.65)
    ax.add_patch(faz)

    # Foveal pit (central depression)
    fp = Ellipse((5, 5), 0.55, 0.55,
                  facecolor="#E65100", edgecolor="none", alpha=0.80)
    ax.add_patch(fp)

    # Optic disc (nasal, slightly below horizontal — right in fundus view)
    disc = Ellipse((7.4, 5.1), 1.5, 1.8,
                    facecolor=C["optic_disc"], edgecolor="#E0C060", lw=1.3)
    ax.add_patch(disc)

    cup = Ellipse((7.4, 5.1), 0.75, 0.90,
                   facecolor="#FFD54F", edgecolor="#FFA000", lw=1.0, alpha=0.85)
    ax.add_patch(cup)

    # Retinal vessels from disc
    vessel_angles = [25, -25, 155, -155]
    for i, ang in enumerate(vessel_angles):
        rad = np.radians(ang)
        xv = 7.4 + 3.2 * np.cos(rad)
        yv = 5.1 + 3.2 * np.sin(rad)
        clr = "#D32F2F" if i % 2 == 0 else "#1565C0"
        ax.plot([7.4, xv], [5.1, yv], "-", color=clr, lw=1.3, alpha=0.80)

    # Labels
    _anns = [
        (5.0, 5.8,  3.2, 7.8,  "Macula  (~4.5 mm dia)"),
        (5.0, 5.0,  3.2, 5.0,  "Fovea  (~1.5 mm)"),
        (5.0, 4.7,  3.2, 3.0,  "Foveal pit\n(thinnest point)"),
        (7.4, 5.9,  6.2, 8.0,  "Optic Disc\n(blind spot)"),
        (7.4, 4.7,  8.5, 3.3,  "Optic Cup"),
    ]
    for x0, y0, xl, yl, txt in _anns:
        _leader(ax, x0, y0, xl, yl)
        _lbl(ax, xl, yl, txt, fs=7.8, bold=True,
             ha="right" if xl < x0 else "left")

    _lbl(ax, 5, 0.55,
         "Red = central retinal artery  ·  Blue = central retinal vein",
         fs=7.5, color="#546E7A", ha="center")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 6 — DEFAULT FULL EYE ANATOMY
# ─────────────────────────────────────────────────────────────────────────────
def _eye_anatomy() -> bytes:
    fig, ax_l, ax_r = _fig_axes(
        "Panel A — Eye Cross-Section  (Sagittal View)",
        "Panel B — Anterior Segment Detail",
    )
    _draw_full_eye(ax_l)
    _draw_anterior_segment(ax_r)
    return _png(fig)


def _draw_full_eye(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    sclera = Ellipse((5, 5), 9.0, 9.0,
                      facecolor=C["sclera"], edgecolor="#BDBDBD", lw=1.5, zorder=1)
    ax.add_patch(sclera)

    choroid = mpatches.Wedge((5, 5), 4.5, 0, 360, width=0.28,
                              facecolor=C["choroid"], edgecolor="none",
                              alpha=0.65, zorder=2)
    ax.add_patch(choroid)

    ret = mpatches.Wedge((5, 5), 4.22, 0, 360, width=0.28,
                          facecolor=C["retina_bg"], edgecolor="none",
                          alpha=0.70, zorder=3)
    ax.add_patch(ret)

    vit = Ellipse((5, 5), 7.6, 7.6,
                   facecolor=C["vitreous"], edgecolor="none", alpha=0.45, zorder=4)
    ax.add_patch(vit)

    co = mpatches.Wedge((5, 5), 4.5, 153, 207, width=0.50,
                          facecolor=C["cornea_body"], edgecolor=C["border"],
                          lw=1.2, zorder=5)
    ax.add_patch(co)

    ac = mpatches.Wedge((5, 5), 3.95, 153, 207,
                          facecolor=C["aqueous_fl"], edgecolor="none",
                          alpha=0.40, zorder=5)
    ax.add_patch(ac)

    iris = mpatches.Wedge((5, 5), 3.35, 153, 207, width=0.52,
                            facecolor=C["iris"], edgecolor="none",
                            alpha=0.90, zorder=6)
    ax.add_patch(iris)

    lens = Ellipse((5.58, 5), 1.0, 2.3,
                    facecolor=C["lens_nuc"], edgecolor=C["lens"],
                    lw=1.3, alpha=0.90, zorder=7)
    ax.add_patch(lens)

    on = FancyBboxPatch((8.55, 4.58), 0.80, 0.85,
                         boxstyle="round,pad=0.05",
                         facecolor=C["optic_nerve"], edgecolor="none",
                         alpha=0.85, zorder=5)
    ax.add_patch(on)

    # Macula
    mac = Ellipse((3.55, 5), 0.65, 0.65,
                   facecolor="#FFD700", edgecolor="#FFA000", lw=1.0, zorder=6)
    ax.add_patch(mac)

    # Labels
    _anns = [
        (1.30, 5.0,  0.15, 5.0,  "Cornea"),
        (3.35, 4.30, 2.50, 3.00, "Iris"),
        (3.10, 5.20, 2.00, 6.60, "Anterior\nChamber"),
        (5.58, 5.0,  5.58, 8.20, "Lens"),
        (4.55, 2.20, 3.20, 1.10, "Retina"),
        (8.55, 5.0,  9.50, 3.60, "Optic\nNerve"),
        (3.55, 5.0,  2.20, 7.30, "Macula"),
        (5.20, 5.0,  5.20, 3.00, "Vitreous"),
    ]
    for x0, y0, xl, yl, txt in _anns:
        _leader(ax, x0, y0, xl, yl)
        _lbl(ax, xl, yl, txt, fs=8.0, bold=True,
             ha="right" if xl < x0 else "left",
             va="bottom" if yl > y0 else "top")


# ── keep mpatches accessible for Wedge calls ─────────────────────────────────
import matplotlib.patches as mpatches  # noqa: E402 (re-import to ensure availability)

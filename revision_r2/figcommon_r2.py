"""Shared page geometry for the R2 figures."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))

JOURNAL = "generic"
WIDTH_MM = 183.0
# ScholarOne (SAGMB) prints running heads and line numbers on three sides of every figure page
# without rescaling the figure; measured on the R1 proof (2026-09-03), plus 1.5 mm of margin.
S1_PAD_MM = {"left": 5.0, "top": 6.5, "bottom": 14.5}


def layout_for_scholarone(fig):
    """Enlarge the page by the overprint bands and return the rect that keeps the drawing inside."""
    w_in, h_in = fig.get_size_inches()
    left, top, bottom = (S1_PAD_MM[k] / 25.4 for k in ("left", "top", "bottom"))
    fig.set_size_inches(w_in + left, h_in + top + bottom)
    W, H = fig.get_size_inches()
    return [left / W, bottom / H, 1.0, 1.0 - top / H]


def save_tif(fig, stem, dpi=600):
    """The upload file: the journal asks for "JPG, TIF or EPS ... at least 300 dpi". It is rasterized
    from the PDF that savefig() has just written, the file every figure gate checks, rather than drawn
    again by matplotlib: an Agg TIFF differs from the PDF in its text and thin lines (local tiles up to
    109 grey levels on the R2 mock figures, 2026-10-22, so check_tiff_matches_pdf.py failed all three).
    White RGB, no alpha (an RGBA TIFF's transparent pixels print black on some RIPs, measured on the R1
    figures, 2026-09-03); LZW-compressed. `fig` is unused and kept for the callers."""
    import fitz
    from PIL import Image
    pdf, path = f"{stem}.pdf", f"{stem}.tif"
    if not os.path.exists(pdf):
        sys.exit(f"save_tif: {pdf} has not been written; call savefig() first")
    with fitz.open(pdf) as doc:
        pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(
            path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    return path

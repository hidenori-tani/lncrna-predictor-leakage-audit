#!/usr/bin/env python
"""Detect text/graphic collisions in the manuscript figures by measuring rendered boxes.

Eyeballing a 2160px render misses overlaps that are obvious at print size. This executes
make_figures.py, then for every Text artist computes its rendered bounding box and tests it
against the bars, the markers, and the other text in the same axes.

Reported overlaps are in display pixels; anything above a couple of px of intersection is a
real collision in the PDF.
"""
import os
import runpy
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

HERE = os.path.dirname(os.path.abspath(__file__))


def inter(a, b):
    """Intersection area of two Bbox objects, in square display pixels."""
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if (dx > 0 and dy > 0) else 0.0


def marker_boxes(ax, renderer):
    """Bounding boxes for each plotted marker, sized by its actual drawn extent.

    A square ms x ms box is wrong for '_': it is a horizontal dash, ms wide but only
    linewidth tall. Approximating it as square over-reports vertical collisions and sends
    you chasing phantoms, so size that one from (ms, markeredgewidth).
    """
    out = []
    for ln in ax.lines:
        if not isinstance(ln, Line2D):
            continue
        ms = ln.get_markersize() or 0
        mk = ln.get_marker()
        if ms <= 0 or mk in ("", "None", None):
            continue
        px_per_pt = ax.figure.dpi / 72.0
        if mk == "_":
            half_w = ms * px_per_pt / 2.0
            half_h = max(ln.get_markeredgewidth(), 1.0) * px_per_pt / 2.0
        elif mk == "|":
            half_w = max(ln.get_markeredgewidth(), 1.0) * px_per_pt / 2.0
            half_h = ms * px_per_pt / 2.0
        else:
            half_w = half_h = ms * px_per_pt / 2.0
        pts = ax.transData.transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
        for px, py in pts:
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            out.append((matplotlib.transforms.Bbox([[px - half_w, py - half_h],
                                                    [px + half_w, py + half_h]]),
                        f"marker '{mk}'"))
    return out


def seg_hits_box(ax, bb):
    """Names of drawn line segments whose stroke passes through the box bb.

    Bars and markers are not the only shapes: Fig. 2 has grey connector segments and the
    reference lines at 0.87 / 0.5. Text printed across any of them is still text on a shape.
    Reference lines that a label deliberately sits *beside* are excluded by the caller.
    """
    out = []
    for ln in ax.lines:
        ls = ln.get_linestyle()
        if ls in ("None", "none", "", None) or ln.get_linewidth() <= 0:
            continue
        xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if xd.size < 2:
            continue
        pts = ax.transData.transform(np.column_stack([xd, yd]))
        pad = ln.get_linewidth() * ax.figure.dpi / 72.0 / 2.0
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if not np.all(np.isfinite([x0, y0, x1, y1])):
                continue
            sb = matplotlib.transforms.Bbox([[min(x0, x1) - pad, min(y0, y1) - pad],
                                             [max(x0, x1) + pad, max(y0, y1) + pad]])
            if inter(sb, bb) > 4:
                out.append(ln)
                break
    return out


def bar_boxes(ax, renderer):
    out = []
    for p in ax.patches:
        if isinstance(p, Rectangle) and p.get_width() > 0 and p.get_height() != 0:
            try:
                out.append((p.get_window_extent(renderer), "bar"))
            except Exception:
                pass
    return out


def text_only_extent(t, renderer):
    """Extent of the glyphs alone.

    Annotation.get_window_extent() unions the text with its arrow, so an arrow correctly
    pointing at its data point reads as a collision. We only care whether the *letters*
    land on something, so call the base Text implementation directly.

    Do NOT do this by setting arrow_patch = None: Annotation.get_window_extent() then
    raises AttributeError inside set_mutation_scale, and if the caller swallows that, the
    text silently gets no box and the checker reports "clean" because it failed to measure.
    """
    return Text.get_window_extent(t, renderer)


def title_artists(ax):
    """Every non-empty title artist.

    set_title(..., loc="left") does NOT populate ax.title — that is the centre title only;
    the left/right ones live on private attributes. Checking ax.title alone silently sees an
    empty string and passes, which is how a real title/legend overlap went unreported here.
    """
    out = []
    for attr in ("title", "_left_title", "_right_title"):
        t = getattr(ax, attr, None)
        if t is not None and t.get_text().strip():
            out.append(t)
    return out


def texts_in(ax):
    """Text drawn inside the axes — excludes tick labels, axis labels and the titles."""
    skip = {ax.title, ax.xaxis.label, ax.yaxis.label}
    skip |= set(title_artists(ax))
    skip |= set(ax.get_xticklabels()) | set(ax.get_yticklabels())
    return [t for t in ax.texts if t not in skip and t.get_text().strip()]


def check(fig, name):
    renderer = fig.canvas.get_renderer()
    hits = []
    for ai, ax in enumerate(fig.axes):
        tx = texts_in(ax)
        # No try/except here on purpose: a checker that swallows a measurement error reports
        # "clean" because it failed, which is worse than not checking. Let it crash.
        boxes = {t: text_only_extent(t, renderer) for t in tx}
        obstacles = bar_boxes(ax, renderer) + marker_boxes(ax, renderer)

        inv = ax.transData.inverted()
        # Labels that name a reference line are meant to sit on/beside it.
        REF_LABELS = {"reported 0.87", "chance"}
        for t, tb in boxes.items():
            label = t.get_text().replace("\n", " / ")[:44]
            if t.get_text().strip() not in REF_LABELS:
                for ln in seg_hits_box(ax, tb):
                    hits.append((name, ai, "TEXT vs line",
                                 f"{label}  [ls={ln.get_linestyle()!r}]", 5.0))
            for ob, kind in obstacles:
                a = inter(tb, ob)
                if a > 4:
                    # report where, in data coords, so the fix is not guesswork
                    ox, oy = inv.transform(((ob.x0 + ob.x1) / 2, (ob.y0 + ob.y1) / 2))
                    hits.append((name, ai, f"TEXT vs {kind}",
                                 f"{label}  [obstacle at x={ox:.2f}, y={oy:.3f}]", a))
        seen = list(boxes.items())
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                a = inter(seen[i][1], seen[j][1])
                if a > 4:
                    hits.append((name, ai, "TEXT vs TEXT",
                                 f"{seen[i][0].get_text()[:20]!r} x {seen[j][0].get_text()[:20]!r}", a))
        # legend box vs markers/bars, and vs the title — Fig. 3 parks its legend directly
        # under the title, which is exactly where a too-small title pad collides.
        leg = ax.get_legend()
        if leg is not None:
            lb = leg.get_window_extent(renderer)
            for ob, kind in obstacles:
                a = inter(lb, ob)
                if a > 4:
                    hits.append((name, ai, f"LEGEND vs {kind}", "legend box", a))
            for ti in title_artists(ax):
                a = inter(lb, ti.get_window_extent(renderer))
                if a > 4:
                    hits.append((name, ai, "LEGEND vs TITLE", ti.get_text()[:40], a))
            for t, tb in boxes.items():
                a = inter(lb, tb)
                if a > 4:
                    hits.append((name, ai, "LEGEND vs TEXT", t.get_text()[:40], a))

        # every title (centre / left / right) vs in-axes text and vs data
        for ti in title_artists(ax):
            tib = ti.get_window_extent(renderer)
            for ob, kind in obstacles:
                a = inter(tib, ob)
                if a > 4:
                    hits.append((name, ai, f"TITLE vs {kind}", ti.get_text()[:40], a))
            for t, tb in boxes.items():
                a = inter(tib, tb)
                if a > 4:
                    hits.append((name, ai, "TITLE vs TEXT", t.get_text()[:40], a))
    return hits


def main():
    runpy.run_path(os.path.join(HERE, "make_figures.py"), run_name="__not_main__")
    figs = [plt.figure(n) for n in plt.get_fignums()]
    names = ["fig1_rnalight", "fig2_dvmnet", "fig3_synthesis"]
    all_hits = []
    for f, n in zip(figs, names):
        f.canvas.draw()
        all_hits += check(f, n)

    if not all_hits:
        print("no collisions above 4 px^2")
        return 0
    print(f"{len(all_hits)} collision(s):\n")
    for name, ai, kind, label, area in sorted(all_hits, key=lambda h: -h[4]):
        print(f"  {name:14s} ax{ai}  {kind:22s} {area:8.0f} px^2  {label}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

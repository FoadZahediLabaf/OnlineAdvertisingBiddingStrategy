"""Repair Req3 slide layout and apply real math sub/superscripts."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
PPTX_PATH = ROOT / "slides" / "Requirement3_PrimalDual.pptx"
PROFILE_ASSET = ROOT / "slides" / "generated_current" / "constellation_profiles.png"
CURVES_ASSET = ROOT / "slides" / "generated_current" / "all_learning_curves.png"
DISTRIBUTION_ASSET = ROOT / "slides" / "generated_current" / "all_setting_final_utility_distribution.png"

NAVY = "102A43"
BLUE = "2F6DF6"
TEAL = "108477"
ORANGE = "E2741F"
PURPLE = "7C3AED"
GREEN = "16856B"
RED = "C2413B"
INK = "1F2937"
MUTED = "64748B"
PALE = "F4F7FB"
WHITE = "FFFFFF"


def rgb(value):
    return RGBColor.from_string(value)


def clear_slide(slide):
    tree = slide.shapes._spTree
    for shape in list(slide.shapes):
        tree.remove(shape._element)


def add_shape(slide, x, y, w, h, fill=WHITE, line="D9E2EC", rounded=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    return shape


def add_text(slide, x, y, w, h, text, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.03)
    frame.margin_top = frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = rgb(color)
    return box


def add_rich(slide, x, y, w, h, segments, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT):
    """Add one rich-text line; baseline is 'sub', 'super', or None."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    for segment in segments:
        if isinstance(segment, str):
            segment = {"text": segment}
        run = paragraph.add_run()
        run.text = segment["text"]
        run.font.name = "Aptos"
        run.font.size = Pt(segment.get("size", size))
        run.font.bold = segment.get("bold", bold)
        run.font.italic = segment.get("italic", False)
        run.font.color.rgb = rgb(segment.get("color", color))
        baseline = segment.get("baseline")
        if baseline:
            run._r.get_or_add_rPr().set("baseline", "-25000" if baseline == "sub" else "30000")
    return box


def title(slide, number, heading, subtitle):
    add_shape(slide, 0, 0, 13.333, 0.16, BLUE, BLUE, rounded=False)
    add_text(slide, 0.67, 0.35, 11.8, 0.48, heading, 25, True, NAVY)
    add_text(slide, 0.68, 0.88, 11.6, 0.35, subtitle, 11, False, MUTED)
    add_text(slide, 12.45, 7.02, 0.35, 0.23, str(number), 9, False, MUTED, PP_ALIGN.RIGHT)


def accent_card(slide, x, y, w, h, accent, heading):
    add_shape(slide, x, y, w, h, WHITE, "D9E2EC", rounded=False)
    add_shape(slide, x, y, 0.08, h, accent, accent, rounded=False)
    add_text(slide, x + 0.28, y + 0.16, w - 0.48, 0.3, heading, 12, True, NAVY)


def sym(base, sub=None, superscript=None, **kwargs):
    parts = [{"text": base, **kwargs}]
    if sub is not None:
        parts.append({"text": str(sub), "baseline": "sub", "size": kwargs.get("size", 18) * 0.72, **{k: v for k, v in kwargs.items() if k != "size"}})
    if superscript is not None:
        parts.append({"text": str(superscript), "baseline": "super", "size": kwargs.get("size", 18) * 0.72, **{k: v for k, v in kwargs.items() if k != "size"}})
    return parts


def rebuild_notation(slide):
    clear_slide(slide)
    title(slide, 1, "Notation", "Symbols are formatted exactly as they appear in the model and update rules.")
    entries = [
        (["T"], "horizon; T = 3000 rounds", BLUE),
        (["N"], "number of campaigns; N = 5", GREEN),
        (["G"], "finite bid grid: {0.0, 0.1, ..., 1.0}", BLUE),
        (["|G|"], "number of bid levels; |G| = 11", GREEN),
        (sym("b", "t", size=20), "bid vector selected at round t", BLUE),
        (sym("m", "t", size=20), "highest-competing-bid vector at round t", GREEN),
        (sym("v", "i", size=20), "campaign i value; normalized to 1", BLUE),
        (sym("Î»", "t", size=20), "dual price controlling expenditure at round t", GREEN),
    ]
    for idx, (symbol, description, accent) in enumerate(entries):
        row, col = divmod(idx, 2)
        x = 0.78 + col * 6.1
        y = 1.42 + row * 1.28
        accent_card(slide, x, y, 5.7, 0.98, accent, "")
        add_rich(slide, x + 0.3, y + 0.18, 0.85, 0.42, symbol, 20, NAVY, True)
        add_text(slide, x + 1.18, y + 0.16, 4.2, 0.5, description, 12, False, INK)


def rebuild_auction(slide):
    clear_slide(slide)
    title(slide, 3, "One round of the auction model", "A compact view of the action, feedback, utility, and cost at round t.")

    cards = [(0.75, BLUE, "1. Choose bids"), (4.55, ORANGE, "2. Observe market"), (8.35, GREEN, "3. Determine wins")]
    for x, accent, heading in cards:
        accent_card(slide, x, 1.45, 3.35, 1.65, accent, heading)

    add_rich(slide, 1.05, 2.0, 2.75, 0.38, sym("b", "t", size=20) + [" = ("] + sym("b", "t,1", size=20) + [", ..., "] + sym("b", "t,N", size=20) + [")"], 20, NAVY, True)
    add_text(slide, 1.05, 2.42, 2.75, 0.43, "One bid for every campaign.", 13)
    add_rich(slide, 4.85, 2.0, 2.75, 0.38, sym("m", "t", size=20) + [" = ("] + sym("m", "t,1", size=20) + [", ..., "] + sym("m", "t,N", size=20) + [")"], 20, NAVY, True)
    add_text(slide, 4.85, 2.42, 2.8, 0.43, "Highest competing bids.", 13)
    add_rich(slide, 8.65, 1.98, 2.85, 0.42, sym("b", "t,i", size=20) + [" â‰¥ "] + sym("m", "t,i", size=20), 20, NAVY, True)
    add_text(slide, 8.65, 2.42, 2.75, 0.43, "Campaign i is won iff this holds.", 13)

    accent_card(slide, 0.95, 3.55, 5.55, 1.65, PURPLE, "Utility")
    utility = sym("u", "t", size=18) + ["("] + sym("b", "t", size=18) + [") = Î£"] + sym("", "i=1", superscript="N", size=18) + [" ("] + sym("v", "i", size=18) + [" âˆ’ "] + sym("b", "t,i", size=18) + [") 1{"] + sym("b", "t,i", size=18) + [" â‰¥ "] + sym("m", "t,i", size=18) + ["}"]
    add_rich(slide, 1.28, 4.08, 4.85, 0.48, utility, 18, NAVY)
    add_text(slide, 1.28, 4.58, 4.85, 0.35, "Value minus payment, summed over won auctions.", 12, False, MUTED)

    accent_card(slide, 6.82, 3.55, 5.55, 1.65, RED, "Cost")
    cost = sym("c", "t", size=18) + ["("] + sym("b", "t", size=18) + [") = Î£"] + sym("", "i=1", superscript="N", size=18) + [" "] + sym("b", "t,i", size=18) + [" 1{"] + sym("b", "t,i", size=18) + [" â‰¥ "] + sym("m", "t,i", size=18) + ["}"]
    add_rich(slide, 7.15, 4.08, 4.85, 0.48, cost, 18, NAVY)
    add_text(slide, 7.15, 4.58, 4.85, 0.35, "Only won auctions consume budget.", 12, False, MUTED)

    add_shape(slide, 0.95, 5.72, 11.42, 0.82, PALE, "D9E2EC")
    bound = sym("c", "t", size=18) + ["("] + sym("b", "t", size=18) + [") â‰¤ Î£"] + sym("", "i=1", superscript="N", size=18) + [" "] + sym("b", "t,i", size=18)
    add_rich(slide, 1.25, 5.91, 3.25, 0.38, bound, 18, NAVY, True)
    add_text(slide, 4.55, 5.85, 7.35, 0.47, "This worst-case upper bound enables a deterministic budget guarantee.", 14, True, BLUE)


def rebuild_environment(slide):
    clear_slide(slide)
    title(slide, 4, "Shared non-stationary market environment", "sKL controls how different regimes are; frequency controls how often the regime changes.")

    cards = [
        (0.75, 1.42, BLUE, "1. Three Gaussian modes"),
        (6.78, 1.42, PURPLE, "2. Separation between modes"),
        (0.75, 3.28, ORANGE, "3. Stochastic mode switching"),
        (6.78, 3.28, GREEN, "4. Freeze the realized market"),
    ]
    for x, y, accent, heading in cards:
        accent_card(slide, x, y, 5.55, 1.48, accent, heading)

    mode_formula = sym("z", "t", size=18) + [" âˆˆ {0, 1, 2},    "] + sym("m", "t", size=18) + [" ~ N("] + sym("Î¼", "zâ‚œ", size=18) + [", "] + sym("Î£", "zâ‚œ", size=18) + [")"]
    add_rich(slide, 1.08, 1.96, 4.75, 0.4, mode_formula, 18, NAVY, True)
    add_text(slide, 1.08, 2.38, 4.75, 0.3, "Each mode is a 10-dimensional distribution of competing bids.", 11, False, MUTED)

    skl_formula = ["sKL("] + sym("P", "a", size=18) + [", "] + sym("P", "b", size=18) + [") âˆˆ {20, 200, 2000}"]
    add_rich(slide, 7.12, 1.96, 4.72, 0.4, skl_formula, 18, NAVY, True)
    add_text(slide, 7.12, 2.38, 4.72, 0.3, "Larger sKL means a more visible distribution shift.", 11, False, MUTED)

    switch_formula = ["Pr("] + sym("z", "t", size=18) + [" â‰  "] + sym("z", "tâˆ’1", size=18) + [") = p"]
    add_rich(slide, 1.08, 3.82, 4.75, 0.4, switch_formula, 18, NAVY, True)
    add_text(slide, 1.08, 4.24, 4.75, 0.3, "p = 1/100, 1/50, or 1/25; larger p gives more change points.", 11, False, MUTED)

    matrix_formula = ["M = ["] + sym("m", "1", size=18) + [", ..., "] + sym("m", "T", size=18) + ["]"] + sym("", superscript="âŠ¤", size=18) + [" âˆˆ [0, 1]"] + sym("", superscript="3000Ã—5", size=18)
    add_rich(slide, 7.12, 3.82, 4.72, 0.4, matrix_formula, 18, NAVY, True)
    add_text(slide, 7.12, 4.24, 4.72, 0.3, "Once sampled, this exact matrix is reused by every algorithm.", 11, False, MUTED)

    add_shape(slide, 0.9, 5.18, 11.55, 1.18, PALE, "D9E2EC")
    add_text(slide, 1.18, 5.36, 1.35, 0.28, "EXAMPLE PATH", 9, True, MUTED)
    timeline = [
        (2.55, 2.05, BLUE, "Mode 0"),
        (4.72, 1.62, ORANGE, "Mode 2"),
        (6.46, 2.25, GREEN, "Mode 1"),
        (8.83, 2.3, BLUE, "Mode 0"),
    ]
    for x, w, color, label in timeline:
        add_shape(slide, x, 5.63, w, 0.42, color, color, rounded=False)
        add_text(slide, x, 5.64, w, 0.38, label, 10, True, WHITE, PP_ALIGN.CENTER)
    add_text(slide, 11.25, 5.67, 0.7, 0.28, "round", 9, False, MUTED)


def rebuild_policy(slide):
    clear_slide(slide)
    title(slide, 5, "Factorized primal-dual bid policy", "One Hedge distribution per campaign; no enumeration of the full joint action space.")
    steps = [
        (0.78, BLUE, "1. Maintain weights"),
        (4.55, PURPLE, "2. Sample per campaign"),
        (8.32, GREEN, "3. Assemble the vector"),
    ]
    for x, accent, heading in steps:
        accent_card(slide, x, 1.65, 3.25, 2.25, accent, heading)
    add_rich(slide, 1.08, 2.25, 2.7, 0.45, sym("w", "t,i", size=22) + ["(g),  g âˆˆ G"], 22, NAVY, True)
    add_text(slide, 1.08, 2.78, 2.62, 0.65, "Scores every bid g for campaign i.", 13)
    add_rich(slide, 4.85, 2.25, 2.7, 0.45, sym("b", "t,i", size=22) + [" ~ "] + sym("p", "t,i", size=22), 22, NAVY, True)
    add_text(slide, 4.85, 2.78, 2.62, 0.65, "Draw one affordable bid per campaign.", 13)
    add_rich(slide, 8.62, 2.25, 2.7, 0.45, sym("b", "t", size=22) + [" = ("] + sym("b", "t,1", size=22) + [", ..., "] + sym("b", "t,N", size=22) + [")"], 22, NAVY, True)
    add_text(slide, 8.62, 2.78, 2.62, 0.65, "The complete 10-dimensional action.", 13)

    add_shape(slide, 0.9, 4.45, 11.55, 1.55, PALE, "D9E2EC")
    add_text(slide, 1.2, 4.7, 2.35, 0.35, "Current setup", 14, True, TEAL)
    add_text(slide, 1.2, 5.08, 3.0, 0.48, "5-cycle conflicts: five single campaigns and five non-adjacent pairs remain available.", 13)
    add_text(slide, 4.55, 4.7, 2.6, 0.35, "Computational benefit", 14, True, PURPLE)
    add_text(slide, 4.55, 5.08, 3.05, 0.48, "10 Ã— 11 arm weights instead of 11^10 joint actions.", 13)
    add_text(slide, 8.15, 4.7, 2.4, 0.35, "Budget safety", 14, True, GREEN)
    add_text(slide, 8.15, 5.08, 3.55, 0.48, "Sampling is filtered by the remaining budget.", 13)


def rebuild_update(slide):
    clear_slide(slide)
    title(slide, 6, "Primal-dual update", "Primal weights learn useful bids; the dual price reacts to spending pressure.")
    positions = [(0.75, 1.5), (6.75, 1.5), (0.75, 3.65), (6.75, 3.65)]
    headings = [("1. Arm score", BLUE), ("2. Hedge update", PURPLE), ("3. Dual update", ORANGE), ("4. Interpretation", GREEN)]
    for (x, y), (heading, accent) in zip(positions, headings):
        accent_card(slide, x, y, 5.55, 1.7, accent, heading)

    lagrangian = sym("ℓ", "t,i", size=18) + ["(g) = "] + sym("u", "t,i", size=18) + ["(g) − "] + sym("λ", "t", size=18) + [" "] + sym("c", "t,i", size=18) + ["(g)"]
    add_rich(slide, 1.08, 2.08, 4.75, 0.42, lagrangian, 18, NAVY, True)
    add_text(slide, 1.08, 2.53, 4.7, 0.43, "Higher dual prices penalize expensive bids.", 12, False, MUTED)

    hedge = sym("w", "t+1,i", size=17) + ["(g) = "] + sym("w", "t,i", size=17) + ["(g) exp(η "] + sym("ℓ̃", "t,i", size=17) + ["(g))"]
    add_rich(slide, 7.08, 2.08, 4.75, 0.42, hedge, 17, NAVY, True)
    normalized_score = (
        sym("ℓ̃", "t,i", size=10.5)
        + ["(g) = ("]
        + sym("ℓ", "t,i", size=10.5)
        + ["(g) - min_h "]
        + sym("ℓ", "t,i", size=10.5)
        + ["(h)) / (max_h "]
        + sym("ℓ", "t,i", size=10.5)
        + ["(h) - min_h "]
        + sym("ℓ", "t,i", size=10.5)
        + ["(h))"]
    )
    add_rich(slide, 7.08, 2.48, 4.95, 0.28, normalized_score, 10.5, NAVY, False)
    add_text(slide, 7.08, 2.82, 4.7, 0.3, "Normalized high-score bids gain multiplicative weight.", 11, False, MUTED)

    dual = sym("λ", "t+1", size=17) + [" = Π"] + sym("", "[0,1/ρ]", size=17) + ["("] + sym("λ", "t", size=17) + [" + "] + sym("η", "λ", size=17) + ["("] + sym("c", "t", size=17) + [" − ρ))"]
    add_rich(slide, 1.08, 4.23, 4.8, 0.42, dual, 17, NAVY, True)
    add_rich(slide, 1.08, 4.7, 4.5, 0.4, ["ρ = B / T"], 16, MUTED)

    add_text(slide, 7.08, 4.16, 4.75, 0.75, "If spending exceeds ρ, λ rises and costly bids become less attractive. Below ρ, the pressure can relax.", 13)
    add_text(slide, 7.08, 4.92, 4.75, 0.32, "Projection keeps the multiplier bounded.", 12, True, GREEN)

    add_shape(slide, 1.3, 5.85, 10.7, 0.68, PALE, "D9E2EC")
    feedback = ["Full-information feedback: after observing "] + sym("m", "t", size=14) + [", every campaign-bid arm can be evaluated."]
    add_rich(slide, 1.6, 6.0, 10.1, 0.32, feedback, 14, BLUE, True, PP_ALIGN.CENTER)


def rebuild_budget(slide):
    clear_slide(slide)
    title(slide, 7, "Hard budget filter", "The dual update encourages control; the feasibility filter provides the guarantee.")
    cards = [(0.72, BLUE, "Before sampling"), (4.55, GREEN, "Why it is safe"), (8.38, ORANGE, "Fallback action")]
    for x, accent, heading in cards:
        accent_card(slide, x, 1.65, 3.35, 2.15, accent, heading)

    feasible = ["Î£"] + sym("", "i=1", superscript="N", size=20) + [" "] + sym("b", "t,i", size=20) + [" â‰¤ "] + sym("B", "t", size=20)
    add_rich(slide, 1.03, 2.28, 2.75, 0.45, feasible, 20, NAVY, True)
    affordable = ["Affordable under remaining budget "] + sym("B", "t", size=12) + ["."]
    add_rich(slide, 1.03, 2.8, 2.7, 0.5, affordable, 12, INK)

    safe = sym("c", "t", size=18) + ["("] + sym("b", "t", size=18) + [") â‰¤ Î£"] + sym("", "i=1", superscript="N", size=18) + [" "] + sym("b", "t,i", size=18)
    add_rich(slide, 4.86, 2.28, 2.78, 0.45, safe, 18, NAVY, True)
    add_text(slide, 4.86, 2.8, 2.72, 0.5, "Only winning bids are actually paid.", 12)

    add_rich(slide, 8.7, 2.28, 2.7, 0.45, sym("b", "t", size=22) + [" = 0"], 22, NAVY, True)
    add_text(slide, 8.7, 2.8, 2.72, 0.5, "Choose the zero vector if no positive bid is affordable.", 12)

    add_shape(slide, 0.9, 4.35, 11.55, 1.75, PALE, "D9E2EC")
    add_text(slide, 1.22, 4.62, 10.9, 0.35, "Deterministic guarantee", 15, True, NAVY)
    guarantee = ["Î£"] + sym("", "t=1", superscript="T", size=20) + [" "] + sym("c", "t", size=20) + ["("] + sym("b", "t", size=20) + [") â‰¤ B"]
    add_rich(slide, 1.22, 5.08, 3.0, 0.43, guarantee, 20, BLUE, True)
    add_text(slide, 4.35, 4.98, 7.4, 0.62, "The filter is conservative, but cumulative realized spending can never exceed the total budget.", 14, False, INK)


def rebuild_protocol(slide):
    clear_slide(slide)
    title(slide, 9, "Shared-realization protocol", "Every algorithm is evaluated on the same saved market path for each setting and realization.")

    cards = [
        (0.75, 1.42, BLUE, "1. Generate a market setting"),
        (6.78, 1.42, TEAL, "2. Freeze one realization"),
        (0.75, 3.35, ORANGE, "3. Reuse the exact matrix"),
        (6.78, 3.35, GREEN, "4. Align and save outputs"),
    ]
    for x, y, accent, heading in cards:
        accent_card(slide, x, y, 5.55, 1.55, accent, heading)

    add_text(
        slide,
        1.08,
        1.98,
        4.75,
        0.68,
        "For each sKL-frequency pair, generate the Gaussian constellation, regime-change schedule, and 10 market realizations.",
        12,
    )

    matrix = sym("M", superscript="(s,r)", size=18) + [" âˆˆ [0, 1]"] + sym("", superscript="3000Ã—5", size=18)
    add_rich(slide, 7.12, 1.89, 4.65, 0.4, matrix, 18, NAVY, True)
    add_rich(slide, 7.12, 2.34, 4.7, 0.38, ["row "] + sym("t", size=12) + [" = one round; column "] + sym("i", size=12) + [" = one campaign"], 12, MUTED)

    add_text(
        slide,
        1.08,
        3.9,
        4.8,
        0.72,
        "Primal-dual, sliding-window CUCB, and change-detector CUCB load the same matrix for setting s and realization r. Only their bid decisions differ.",
        12,
    )
    add_text(
        slide,
        7.12,
        3.9,
        4.8,
        0.72,
        "Store per-round bids, utility, cost, and remaining budget; then save one final summary row for each algorithm-realization pair.",
        12,
    )

    add_shape(slide, 0.9, 5.28, 11.55, 1.22, PALE, "D9E2EC")
    add_text(slide, 1.2, 5.46, 1.55, 0.32, "WHY THIS IS FAIR", 10, True, BLUE)
    add_text(
        slide,
        2.72,
        5.38,
        9.15,
        0.48,
        "Performance differences come from the learning rules, not from different sampled auctions.",
        14,
        True,
        NAVY,
    )
    add_text(
        slide,
        1.2,
        5.89,
        10.75,
        0.36,
        "config.json records sKL, frequency, seeds, change points, and active modes; learner seeds remain algorithm-specific.",
        10,
        False,
        MUTED,
    )


def rebuild_distribution(slide):
    clear_slide(slide)
    title(
        slide,
        13,
        "Final utility distribution across all settings",
        "Each dot is one realization; black markers show mean +/- 1 SD.",
    )
    if DISTRIBUTION_ASSET.exists():
        slide.shapes.add_picture(str(DISTRIBUTION_ASSET), Inches(0.85), Inches(1.35), width=Inches(11.7), height=Inches(5.7))


def refine_generated_slides(prs):
    # Multi-value frequency card: keep it on one clean line.
    for shape in prs.slides[1].shapes:
        if getattr(shape, "text", "").strip() == "1/100 / 1/50 / 1/25":
            for paragraph in shape.text_frame.paragraphs:
                paragraph.font.size = Pt(16)
    for shape in prs.slides[3].shapes:
        value = getattr(shape, "text", "").strip()
        replacement = {"3 Gaussian modes": "3 modes", "Bernoulli changes": "Bernoulli p"}.get(value)
        if replacement:
            shape.text_frame.paragraphs[0].text = replacement
            shape.text_frame.paragraphs[0].font.name = "Aptos"
            shape.text_frame.paragraphs[0].font.size = Pt(18)
            shape.text_frame.paragraphs[0].font.bold = True
            shape.text_frame.paragraphs[0].font.color.rgb = rgb(NAVY)
    for shape in prs.slides[8].shapes:
        if getattr(shape, "text", "").strip() == ">":
            for paragraph in shape.text_frame.paragraphs:
                paragraph.font.size = Pt(14)
    for shape in prs.slides[14].shapes:
        if getattr(shape, "text", "").strip() == "1/100 / 1/50 / 1/25":
            for paragraph in shape.text_frame.paragraphs:
                paragraph.font.size = Pt(15)
    # Slightly reduce the two text blocks that sit exactly on their width limit.
    for slide_index, prefix, size in [
        (7, "Cumulative utility, cumulative cost", 14),
        (25, "The factorized primal-dual learner", 14),
    ]:
        for shape in prs.slides[slide_index].shapes:
            if getattr(shape, "text", "").startswith(prefix):
                for paragraph in shape.text_frame.paragraphs:
                    paragraph.font.size = Pt(size)
    # Current market vector notation on slide 4.
    for shape in prs.slides[3].shapes:
        if getattr(shape, "text", "").strip() == "m_t in [0,1]^5":
            x, y, w, h = shape.left, shape.top, shape.width, shape.height
            shape._element.getparent().remove(shape._element)
            add_rich(prs.slides[3], x / 914400, y / 914400, w / 914400, h / 914400,
                     sym("m", "t", size=17) + [" in [0, 1]"] + sym("", superscript="5", size=17),
                     17, NAVY, True, PP_ALIGN.CENTER)


def repair_presentation(prs):
    rebuild_notation(prs.slides[0])
    rebuild_auction(prs.slides[2])
    rebuild_environment(prs.slides[3])
    rebuild_policy(prs.slides[4])
    rebuild_update(prs.slides[5])
    rebuild_budget(prs.slides[6])
    rebuild_protocol(prs.slides[8])
    rebuild_distribution(prs.slides[12])
    refine_generated_slides(prs)
    replacements = {
        "Ale symmetric-KL convention": "Shared symmetric-KL definition",
        "ALE ENVIRONMENT": "MARKET GENERATOR",
        "Ale's environment": "the shared environment",
        "Lines show the mean; shaded bands show +/- 1 SD across 10 realizations.": "Line color identifies sKL; line style identifies change frequency.",
    }
    for slide in prs.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    for old, new in replacements.items():
                        run.text = run.text.replace(old, new)
    if PROFILE_ASSET.exists():
        for slide_index in (14, 15):
            slide = prs.slides[slide_index]
            pictures = [shape for shape in slide.shapes if shape.shape_type == 13]
            for picture in pictures:
                left, top, width, height = picture.left, picture.top, picture.width, picture.height
                picture._element.getparent().remove(picture._element)
                slide.shapes.add_picture(str(PROFILE_ASSET), left, top, width=width, height=height)
    if CURVES_ASSET.exists():
        for slide_index in (9,):
            slide = prs.slides[slide_index]
            pictures = [shape for shape in slide.shapes if shape.shape_type == 13]
            for picture in pictures:
                left, top, width, height = picture.left, picture.top, picture.width, picture.height
                picture._element.getparent().remove(picture._element)
                slide.shapes.add_picture(str(CURVES_ASSET), left, top, width=width, height=height)
    for number, slide in enumerate(prs.slides, start=1):
        for shape in slide.shapes:
            if shape.left < Inches(12) or shape.top < Inches(6.8):
                continue
            if not getattr(shape, "has_text_frame", False) or not shape.text.strip().isdigit():
                continue
            paragraph = shape.text_frame.paragraphs[0]
            if paragraph.runs:
                paragraph.runs[0].text = str(number)
            else:
                paragraph.text = str(number)
    return prs


def main():
    prs = Presentation(PPTX_PATH)
    repair_presentation(prs)
    prs.save(PPTX_PATH)
    print(f"repaired typography in {PPTX_PATH}")


if __name__ == "__main__":
    main()


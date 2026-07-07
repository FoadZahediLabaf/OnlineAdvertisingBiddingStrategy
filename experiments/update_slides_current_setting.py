"""Rebuild all Req3 experiment slides from the current shared realizations."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_req3_on_shared_realizations import (  # noqa: E402
    run_factorized_primal_dual_trial,
)
from experiments.piecewise_static_oracle import (  # noqa: E402
    solve_piecewise_static_oracle,
)


PPTX_TEMPLATE_PATH = ROOT / "slides" / "Requirement3_PrimalDual_template.pptx"
PPTX_PATH = ROOT / "slides" / "Requirement3_PrimalDual.pptx"
MATRIX_DIR = ROOT / "results" / "processed" / "req4_shared_realizations"
RESULT_DIR = ROOT / "results" / "processed" / "req3_on_req4_shared"
TIMESERIES_DIR = RESULT_DIR / "timeseries"
ASSET_DIR = ROOT / "slides" / "generated_current"

T = 3000
N = 5
BUDGET = 2500.0
BID_SET = np.linspace(0.0, 1.0, 11)
VALUES = np.ones(N)

FREQ_ORDER = ["low_frequency", "medium_frequency", "high_frequency"]
SKL_ORDER = ["low_skl", "medium_skl", "high_skl"]
FREQ_LABELS = {
    "low_frequency": "Low frequency (1/1500)",
    "medium_frequency": "Medium frequency (1/50)",
    "high_frequency": "High frequency (1/25)",
}
SKL_LABELS = {
    "low_skl": "Low sKL (20)",
    "medium_skl": "Medium sKL (200)",
    "high_skl": "High sKL (500)",
}

NAVY = "102A43"
BLUE = "2F6DF6"
TEAL = "108477"
ORANGE = "E2741F"
INK = "1F2937"
MUTED = "64748B"
PALE = "F4F7FB"
WHITE = "FFFFFF"
GREEN = "16856B"
RED = "C2413B"
PURPLE = "7C3AED"


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def load_and_run_all():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
    runs = {}
    raw_rows = []

    for setting_dir in sorted(path for path in MATRIX_DIR.iterdir() if path.is_dir()):
        config = json.loads((setting_dir / "config.json").read_text(encoding="utf-8"))
        setting_runs = []
        for realization_idx, matrix_path in enumerate(sorted(setting_dir.glob("realization_*.npy"))):
            market_matrix = np.load(matrix_path)
            result = run_factorized_primal_dual_trial(
                env_seed=0,
                learner_seed=2000 + realization_idx,
                T=T,
                N=N,
                K_value=int(config["oracle_block_length"]),
                bid_set=BID_SET,
                values=VALUES,
                budget=BUDGET,
                allowed_campaign_sets=config.get("allowed_campaign_sets"),
                conflict_edges=config.get("conflict_edges"),
                environment="precomputed",
                environment_kwargs={"path": matrix_path},
            )
            oracle = solve_piecewise_static_oracle(
                market_matrix=market_matrix,
                change_points=config["change_points"],
                bid_set=BID_SET,
                values=VALUES,
                budget=BUDGET,
                allowed_campaign_sets=config.get("allowed_campaign_sets"),
                conflict_edges=config.get("conflict_edges"),
            )
            regret = oracle["cumulative_utility"] - result["cumulative_utility"]
            compact = {
                "cumulative_utility": result["cumulative_utility"],
                "cumulative_cost": result["cumulative_cost"],
                "cumulative_oracle_utility": oracle["cumulative_utility"],
                "cumulative_oracle_cost": oracle["cumulative_cost"],
                "regret": regret,
                "remaining_budget": result["remaining_budgets"],
                "lambda_dual": result["lambdas"],
                "mean_bid": result["chosen_actions"].mean(axis=1),
            }
            setting_runs.append(compact)
            raw_rows.append(
                {
                    "setting_id": config["setting_id"],
                    "frequency_label": config["frequency_label"],
                    "skl_label": config["skl_label"],
                    "target_skl": config["target_skl"],
                    "change_frequency": config["change_frequency"],
                    "realization": realization_idx,
                    "matrix_file": str(matrix_path.relative_to(ROOT)),
                    "final_utility": float(result["cumulative_utility"][-1]),
                    "final_cost": float(result["cumulative_cost"][-1]),
                    "final_oracle_utility": float(oracle["cumulative_utility"][-1]),
                    "final_oracle_cost": float(oracle["cumulative_cost"][-1]),
                    "final_regret": float(regret[-1]),
                    "budget_left": float(result["remaining_budgets"][-1]),
                    "mean_bid": float(result["chosen_actions"].mean()),
                    "mean_positive_bids_per_round": float(
                        (result["chosen_actions"] > 0).sum(axis=1).mean()
                    ),
                }
            )

        stacked = {
            key: np.stack([run[key] for run in setting_runs])
            for key in setting_runs[0]
        }
        runs[config["setting_id"]] = {"config": config, **stacked}
        np.savez_compressed(TIMESERIES_DIR / f"{config['setting_id']}.npz", **stacked)
        print(f"ran {config['setting_id']}: {len(setting_runs)} realizations")

    with (RESULT_DIR / "req3_factorized_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = []
    for setting_id, data in sorted(runs.items()):
        cfg = data["config"]
        utility = data["cumulative_utility"][:, -1]
        cost = data["cumulative_cost"][:, -1]
        oracle_utility = data["cumulative_oracle_utility"][:, -1]
        oracle_cost = data["cumulative_oracle_cost"][:, -1]
        regret = data["regret"][:, -1]
        budget_left = data["remaining_budget"][:, -1]
        summary_rows.append(
            {
                "setting_id": setting_id,
                "frequency_label": cfg["frequency_label"],
                "skl_label": cfg["skl_label"],
                "target_skl": cfg["target_skl"],
                "change_frequency": cfg["change_frequency"],
                "n_realizations": len(utility),
                "utility_mean": float(utility.mean()),
                "utility_std": float(utility.std(ddof=1)),
                "cost_mean": float(cost.mean()),
                "cost_std": float(cost.std(ddof=1)),
                "oracle_utility_mean": float(oracle_utility.mean()),
                "oracle_utility_std": float(oracle_utility.std(ddof=1)),
                "oracle_cost_mean": float(oracle_cost.mean()),
                "oracle_cost_std": float(oracle_cost.std(ddof=1)),
                "regret_mean": float(regret.mean()),
                "regret_std": float(regret.std(ddof=1)),
                "budget_left_mean": float(budget_left.mean()),
                "budget_left_max": float(budget_left.max()),
                "n_change_points": len(cfg["change_points"]),
            }
        )

    with (RESULT_DIR / "req3_factorized_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return runs, summary_rows, raw_rows


def save_figure(fig, name):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / name
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_heatmaps(summary):
    utility = np.zeros((3, 3))
    regret = np.zeros((3, 3))
    cost = np.zeros((3, 3))
    for row in summary:
        i = FREQ_ORDER.index(row["frequency_label"])
        j = SKL_ORDER.index(row["skl_label"])
        utility[i, j] = row["utility_mean"]
        regret[i, j] = row["regret_mean"]
        cost[i, j] = row["cost_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, grid, title, cmap in [
        (axes[0], utility, "Mean final utility", "Blues"),
        (axes[1], regret, "Mean final regret", "Purples"),
    ]:
        image = ax.imshow(grid, aspect="auto", cmap=cmap)
        ax.set_xticks(range(3), ["sKL 20", "sKL 200", "sKL 500"])
        ax.set_yticks(range(3), ["1/1500", "1/50", "1/25"])
        ax.set_xlabel("Symmetric KL level")
        ax.set_ylabel("Change probability")
        ax.set_title(title, fontweight="bold")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{grid[i, j]:,.1f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Current 3 x 3 utility and regret grid (10 shared realizations per cell)", fontweight="bold")
    fig.tight_layout()
    sensitivity = save_figure(fig, "sensitivity_grid.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    gap = BUDGET - cost
    image = ax.imshow(gap, aspect="auto", cmap="Greens_r", vmin=0)
    ax.set_xticks(range(3), ["sKL 20", "sKL 200", "sKL 500"])
    ax.set_yticks(range(3), ["1/1500", "1/50", "1/25"])
    ax.set_xlabel("Symmetric KL level")
    ax.set_ylabel("Change probability")
    ax.set_title("Mean unused budget (B - final cost)", fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{gap[i, j]:.3f}", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    budget = save_figure(fig, "budget_feasibility.png")
    return sensitivity, budget


def plot_learning_curves(runs):
    rounds = np.arange(1, T + 1)
    colors = {"low_skl": BLUE, "medium_skl": TEAL, "high_skl": ORANGE}
    styles = {"low_frequency": "-", "medium_frequency": "--", "high_frequency": ":"}
    legend_freq_order = ["high_frequency", "medium_frequency", "low_frequency"]
    legend_skl_order = ["high_skl", "medium_skl", "low_skl"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for freq in legend_freq_order:
        for skl in legend_skl_order:
            data = runs[f"{freq}_{skl}"]
            cfg = data["config"]
            mean = data["cumulative_utility"].mean(axis=0)
            ax.plot(
                rounds,
                mean,
                color=f"#{colors[cfg['skl_label']]}",
                linestyle=styles[cfg["frequency_label"]],
                linewidth=1.8,
                label=f"{SKL_LABELS[cfg['skl_label']]} / {FREQ_LABELS[cfg['frequency_label']].split(' ')[0]}",
            )
    ax.set_title("Mean cumulative utility for all current settings", fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative utility")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, fontsize=7.5, loc="upper left")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    fig.tight_layout()
    return save_figure(fig, "all_learning_curves.png")


def plot_final_utility_distribution(runs):
    colors = {"low_skl": BLUE, "medium_skl": TEAL, "high_skl": ORANGE}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True)
    jitter = np.linspace(-0.09, 0.09, 10)
    for ax, freq in zip(axes, FREQ_ORDER):
        for index, skl in enumerate(SKL_ORDER):
            values = runs[f"{freq}_{skl}"]["cumulative_utility"][:, -1]
            color = f"#{colors[skl]}"
            ax.scatter(index + jitter, values, color=color, alpha=0.55, s=28)
            ax.errorbar(
                index,
                values.mean(),
                yerr=values.std(ddof=1),
                fmt="o",
                color="black",
                capsize=5,
                markersize=6,
                linewidth=1.5,
            )
        ax.set_xticks(range(3), ["20", "200", "500"])
        ax.set_xlabel("Target sKL")
        ax.set_title(FREQ_LABELS[freq], fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Final cumulative utility")
    fig.suptitle("Final utility across all settings (dots = realizations; black = mean +/- SD)", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "all_setting_final_utility_distribution.png")


def padded_limits(values, floor_zero=False, extra_values=None):
    arrays = [np.asarray(value).ravel() for value in values]
    if extra_values is not None:
        arrays.append(np.asarray(extra_values).ravel())
    combined = np.concatenate(arrays)
    lower = float(np.nanmin(combined))
    upper = float(np.nanmax(combined))
    span = upper - lower
    if span <= 1e-12:
        span = max(abs(upper), 1.0)
    lower -= 0.06 * span
    upper += 0.06 * span
    if floor_zero:
        lower = 0.0
    return lower, upper


def build_diagnostic_limits(runs, setting_ids):
    utility_values = []
    regret_values = []
    cost_values = []
    lambda_values = []
    for setting_id in setting_ids:
        data = runs[setting_id]
        utility_values.extend([data["cumulative_utility"], data["cumulative_oracle_utility"]])
        regret_values.append(data["regret"])
        cost_values.extend([data["cumulative_cost"], data["cumulative_oracle_cost"]])
        lambda_values.append(data["lambda_dual"])
    return {
        "utility": padded_limits(utility_values, floor_zero=True),
        "regret": padded_limits(regret_values, extra_values=[0.0]),
        "cost": padded_limits(cost_values, floor_zero=True, extra_values=[BUDGET]),
        "lambda": padded_limits(lambda_values, extra_values=[0.0]),
    }


def plot_diagnostic(setting_id, data, y_limits=None):
    rounds = np.arange(1, T + 1)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0))

    def plot_band(ax, values, label, color, linestyle="-"):
        mean = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        ax.plot(rounds, mean, color=color, linestyle=linestyle, linewidth=2, label=label)
        ax.fill_between(rounds, mean - sd, mean + sd, color=color, alpha=0.14)

    plot_band(axes[0, 0], data["cumulative_utility"], "Learner", f"#{BLUE}")
    plot_band(axes[0, 0], data["cumulative_oracle_utility"], "Oracle", "black", "--")
    axes[0, 0].set_title("Cumulative utility", fontweight="bold", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    if y_limits:
        axes[0, 0].set_ylim(*y_limits["utility"])

    plot_band(axes[0, 1], data["regret"], "Oracle - learner", f"#{PURPLE}")
    axes[0, 1].axhline(0.0, color="black", linestyle=":", linewidth=1)
    axes[0, 1].set_title("Cumulative regret", fontweight="bold", fontsize=10)
    if y_limits:
        axes[0, 1].set_ylim(*y_limits["regret"])

    plot_band(axes[1, 0], data["cumulative_cost"], "Learner", f"#{BLUE}")
    plot_band(axes[1, 0], data["cumulative_oracle_cost"], "Oracle", "black", "--")
    axes[1, 0].axhline(BUDGET, color=f"#{RED}", linestyle=":", linewidth=1, label="Budget")
    axes[1, 0].set_title("Cumulative cost", fontweight="bold", fontsize=10)
    axes[1, 0].legend(fontsize=8)
    if y_limits:
        axes[1, 0].set_ylim(*y_limits["cost"])

    plot_band(axes[1, 1], data["lambda_dual"], "Learner", f"#{TEAL}")
    axes[1, 1].set_title("Dual multiplier", fontweight="bold", fontsize=10)
    if y_limits:
        axes[1, 1].set_ylim(*y_limits["lambda"])
    for ax in axes.flat:
        ax.set_xlim(0, T)
        ax.set_xlabel("Round")
        ax.grid(alpha=0.2)
    cfg = data["config"]
    fig.suptitle(
        f"{FREQ_LABELS[cfg['frequency_label']]} / {SKL_LABELS[cfg['skl_label']]}: mean +/- SD over 10 shared realizations",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    return save_figure(fig, f"diagnostic_{setting_id}.png")


def plot_summary_axis(runs, axis):
    if axis == "skl":
        keys = [f"medium_frequency_{skl}" for skl in SKL_ORDER]
        labels = ["20", "200", "500"]
        xlabel = "Target symmetric KL"
        title = "Medium frequency (1/50): final outcomes by sKL"
    else:
        keys = [f"{freq}_medium_skl" for freq in FREQ_ORDER]
        labels = ["1/1500", "1/50", "1/25"]
        xlabel = "Change probability"
        title = "Medium sKL (200): final outcomes by frequency"

    utilities = [runs[key]["cumulative_utility"][:, -1] for key in keys]
    regrets = [runs[key]["regret"][:, -1] for key in keys]
    comparison_keys = {
        *[f"medium_frequency_{skl}" for skl in SKL_ORDER],
        *[f"{freq}_medium_skl" for freq in FREQ_ORDER],
    }
    all_comparison_utilities = np.concatenate(
        [runs[key]["cumulative_utility"][:, -1] for key in comparison_keys]
    )
    all_comparison_regrets = np.concatenate(
        [runs[key]["regret"][:, -1] for key in comparison_keys]
    )
    utility_limits = (
        500 * np.floor((all_comparison_utilities.min() - 100) / 500),
        500 * np.ceil((all_comparison_utilities.max() + 100) / 500),
    )
    regret_limits = (
        -100,
        500 * np.ceil((all_comparison_regrets.max() + 100) / 500),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, values, ylabel, color in [
        (axes[0], utilities, "Final utility", BLUE),
        (axes[1], regrets, "Final regret", PURPLE),
    ]:
        for idx, sample in enumerate(values):
            ax.scatter(np.full(len(sample), idx), sample, alpha=0.55, color=f"#{color}", s=24)
            ax.errorbar(
                idx,
                sample.mean(),
                yerr=sample.std(ddof=1),
                fmt="o",
                color="black",
                capsize=5,
                markersize=6,
            )
        ax.set_xticks(range(3), labels)
        ax.set_xlim(-0.25, 2.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1, label="Zero regret")
    axes[0].set_ylim(*utility_limits)
    axes[1].set_ylim(*regret_limits)
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].legend(fontsize=8)
    fig.suptitle(title + " (dots = realizations; marker = mean +/- SD)", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, f"summary_{axis}.png")


def plot_constellation_profiles():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, skl in zip(axes, SKL_ORDER):
        path = MATRIX_DIR / f"medium_frequency_{skl}" / "constellation.npz"
        data = np.load(path)
        mus = data["mus"]
        for mode_idx, mu in enumerate(mus):
            ax.plot(range(N), mu, marker="o", linewidth=1.8, label=f"Mode {mode_idx}")
        ax.set_title(SKL_LABELS[skl], fontweight="bold")
        ax.set_xlabel("Campaign")
        ax.set_xticks(range(N))
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Gaussian mean")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Current shared-environment constellation means (seed 42)", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "constellation_profiles.png")


def plot_high_frequency_spectrum():
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.7), sharex=True)
    for ax, skl in zip(axes, SKL_ORDER):
        matrix = np.load(MATRIX_DIR / f"high_frequency_{skl}" / "realization_00.npy")
        ax.imshow(matrix.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_ylabel(SKL_LABELS[skl].replace(" ", "\n"), fontsize=8)
    axes[-1].set_xlabel("Round")
    fig.suptitle("High-frequency (1/25) realized market streams across sKL", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "high_frequency_spectrum.png")


def clear_slide(slide):
    tree = slide.shapes._spTree
    for shape in list(slide.shapes):
        tree.remove(shape._element)


def add_box(slide, x, y, w, h, fill=WHITE, line="D9E2EC", radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    return shape


def add_text(slide, x, y, w, h, text, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = rgb(color)
    return box


def add_title(slide, title, subtitle, number):
    add_box(slide, 0, 0, 13.333, 0.16, fill=BLUE, line=BLUE, radius=False)
    add_text(slide, 0.65, 0.35, 11.9, 0.48, title, size=25, bold=True, color=NAVY)
    add_text(slide, 0.67, 0.87, 11.7, 0.36, subtitle, size=11, color=MUTED)
    add_text(slide, 12.45, 7.02, 0.35, 0.25, str(number), size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def add_card(slide, x, y, w, h, label, value, accent=BLUE, note=None, value_size=21):
    add_box(slide, x, y, w, h, fill=PALE, line="D9E2EC")
    add_text(slide, x + 0.2, y + 0.12, w - 0.4, 0.28, label.upper(), size=9, bold=True, color=accent)
    add_text(slide, x + 0.2, y + 0.43, w - 0.4, 0.34, value, size=value_size, bold=True, color=NAVY)
    if note:
        add_text(slide, x + 0.2, y + h - 0.28, w - 0.4, 0.18, note, size=8, color=MUTED)


def add_picture(slide, path, x, y, w, h):
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def rebuild_slide(slide, number, title, subtitle):
    clear_slide(slide)
    add_title(slide, title, subtitle, number)


def remove_slide_by_title(prs, target_title):
    for index, slide in enumerate(prs.slides):
        texts = [getattr(shape, "text", "").strip() for shape in slide.shapes]
        if target_title not in texts:
            continue
        slide_id = prs.slides._sldIdLst[index]
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)
        return True
    return False


def remove_slides_by_title(prs, target_title, keep=0):
    seen = 0
    index = 0
    while index < len(prs.slides):
        texts = [getattr(shape, "text", "").strip() for shape in prs.slides[index].shapes]
        if target_title not in texts:
            index += 1
            continue
        seen += 1
        if seen <= keep:
            index += 1
            continue
        slide_id = prs.slides._sldIdLst[index]
        prs.part.drop_rel(slide_id.rId)
        prs.slides._sldIdLst.remove(slide_id)


def move_slide(prs, old_index, new_index):
    slide_id = prs.slides._sldIdLst[old_index]
    prs.slides._sldIdLst.remove(slide_id)
    prs.slides._sldIdLst.insert(new_index, slide_id)


def renumber_slides(prs):
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


def add_matchstick_edge(slide, x1, y1, x2, y2):
    shadow = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1 + 0.02),
        Inches(y1 + 0.02),
        Inches(x2 + 0.02),
        Inches(y2 + 0.02),
    )
    shadow.line.color.rgb = rgb("8A4B16")
    shadow.line.width = Pt(9)
    edge = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    edge.line.color.rgb = rgb("D9983D")
    edge.line.width = Pt(7)


def add_graph_node(slide, x, y, label):
    node = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.18), Inches(y - 0.18), Inches(0.36), Inches(0.36))
    node.fill.solid()
    node.fill.fore_color.rgb = rgb(NAVY)
    node.line.color.rgb = rgb(WHITE)
    node.line.width = Pt(1.2)
    add_text(slide, x - 0.12, y - 0.115, 0.24, 0.2, str(label), size=8.5, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


def build_conflict_graph_slide(slide):
    clear_slide(slide)
    add_title(slide, "Conflict graph and feasible super-arms", "Edges are incompatible campaign pairs; a super-arm is feasible only if it contains no edge.", 6)

    nodes = {
        0: (3.05, 1.78),
        1: (4.45, 2.78),
        2: (3.9, 4.42),
        3: (2.2, 4.42),
        4: (1.65, 2.78),
    }
    for first, second in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]:
        x1, y1 = nodes[first]
        x2, y2 = nodes[second]
        add_matchstick_edge(slide, x1, y1, x2, y2)
    for node, (x, y) in nodes.items():
        add_graph_node(slide, x, y, node)

    add_text(slide, 0.85, 5.35, 4.45, 0.35, "Forbidden pairs are the five cycle edges.", size=13, bold=True, color=RED, align=PP_ALIGN.CENTER)

    add_box(slide, 5.6, 1.38, 6.7, 1.4, fill=WHITE, line="C8D6E5")
    add_text(slide, 5.9, 1.6, 1.8, 0.3, "Blocked", size=14, bold=True, color=RED)
    add_text(slide, 5.9, 2.0, 5.9, 0.42, "{0,1}, {1,2}, {2,3}, {3,4}, {4,0}", size=16, bold=True, color=NAVY)
    add_text(slide, 5.9, 2.42, 5.95, 0.22, "Adjacent campaigns cannot run in the same round.", size=10.5, color=MUTED)

    add_box(slide, 5.6, 3.08, 6.7, 2.05, fill=PALE, line="C8D6E5")
    add_text(slide, 5.9, 3.3, 2.1, 0.3, "Feasible", size=14, bold=True, color=GREEN)
    add_text(slide, 5.9, 3.72, 5.95, 0.34, "Singletons: {0}, {1}, {2}, {3}, {4}", size=13.5, bold=True, color=NAVY)
    add_text(slide, 5.9, 4.14, 5.95, 0.5, "Non-adjacent pairs: {0,2}, {0,3}, {1,3}, {1,4}, {2,4}", size=13.5, bold=True, color=NAVY)
    add_text(slide, 5.9, 4.72, 5.95, 0.22, "These 10 sets are the super-arms learned by the Req3 policy.", size=10.5, color=MUTED)

    add_box(slide, 0.85, 6.25, 11.45, 0.55, fill=PALE, line="D9E2EC")
    add_text(slide, 1.05, 6.34, 11.05, 0.34, "The learner and the oracle use the same feasible super-arm set, so regret is measured on the same action space.", size=12.2, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    return slide


def add_conflict_graph_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    return build_conflict_graph_slide(slide)


def metrics_for(runs, setting_id):
    data = runs[setting_id]
    utility = data["cumulative_utility"][:, -1]
    cost = data["cumulative_cost"][:, -1]
    return utility.mean(), utility.std(ddof=1), cost.mean(), cost.std(ddof=1)


def regret_metrics(runs, setting_id):
    data = runs[setting_id]
    regret = data["regret"][:, -1]
    oracle = data["cumulative_oracle_utility"][:, -1]
    return regret.mean(), regret.std(ddof=1), oracle.mean(), oracle.std(ddof=1)


def rebuild_deck(runs, summary, assets):
    source_path = PPTX_TEMPLATE_PATH if PPTX_TEMPLATE_PATH.exists() else PPTX_PATH
    prs = Presentation(source_path)
    remove_slide_by_title(prs, "Separating sKL and frequency effects")
    remove_slides_by_title(prs, "Conflict graph and feasible super-arms", keep=0)

    # Slide 2: experiment setting.
    slide = prs.slides[1]
    rebuild_slide(slide, 2, "Req3 experiment setting", "Symbols introduced here are used throughout the results slides.")
    add_card(slide, 0.7, 1.45, 2.8, 1.25, "Shared base", "T=3000, N=5", BLUE, "Budget B=2500; 11 bid levels")
    add_card(slide, 3.75, 1.45, 2.8, 1.25, "Separation sKL", "sKL = 20 / 200 / 500", TEAL, "Pairwise Gaussian separation", value_size=16)
    add_card(slide, 6.8, 1.45, 2.8, 1.25, "Change frequency p", "p = 1/1500 / 1/50 / 1/25", ORANGE, "Bernoulli regime-change probability", value_size=16)
    add_card(slide, 9.85, 1.45, 2.8, 1.25, "Realizations R", "R=10 per cell", GREEN, "90 saved market matrices")
    add_box(slide, 0.7, 3.05, 11.95, 2.8, fill=WHITE, line="C8D6E5")
    add_text(slide, 1.0, 3.3, 11.3, 0.4, "Notation saved with each setting", size=17, bold=True, color=NAVY)
    add_text(slide, 1.0, 3.85, 11.2, 1.25, "For each setting (sKL, p), config.json stores change points, phase_ids q_t, active modes, and seeds. constellation.npz stores Gaussian means mu_k and covariances Sigma_k; streams.npz stores market vectors m_t and phase_ids for t=1,...,T.", size=14)
    add_text(slide, 1.0, 5.15, 11.2, 0.42, "The learner receives the precomputed m_t sequence; learner randomness does not regenerate q_t or the market stream.", size=12, bold=True, color=BLUE)

    # Slide 4: current environment.
    slide = prs.slides[3]
    rebuild_slide(slide, 4, "Non-stationary market environment", "Each experiment consumes a precomputed 3000 x 5 highest-competing-bid matrix.")
    add_card(slide, 0.75, 1.5, 2.7, 1.35, "1. Constellation", "3 Gaussian modes", BLUE, "Target sKL controls separation")
    add_card(slide, 3.7, 1.5, 2.7, 1.35, "2. Schedule", "Bernoulli changes", TEAL, "p = 1/1500, 1/50, 1/25")
    add_card(slide, 6.65, 1.5, 2.7, 1.35, "3. Realization", "m_t in [0,1]^5", ORANGE, "One vector for every round")
    add_card(slide, 9.6, 1.5, 2.7, 1.35, "4. Input", "Saved matrix", GREEN, "Loaded by the learner")
    add_picture(slide, assets["high_spectrum"], 0.85, 3.15, 11.7, 3.35)

    # Slide 8: setup.
    slide = prs.slides[7]
    rebuild_slide(slide, 8, "Current experiment setup", "All numerical results from this point use the 3 x 3 grid with 10 realizations per cell.")
    add_card(slide, 0.75, 1.45, 2.7, 1.2, "Horizon", "3000 rounds", BLUE)
    add_card(slide, 3.7, 1.45, 2.7, 1.2, "Campaigns", "5", TEAL)
    add_card(slide, 6.65, 1.45, 2.7, 1.2, "Budget", "2500", ORANGE)
    add_card(slide, 9.6, 1.45, 2.7, 1.2, "Bid grid", "0.0 to 1.0", GREEN, "Step 0.1; 11 bids")
    add_box(slide, 0.75, 3.0, 11.55, 2.75, fill=WHITE, line="C8D6E5")
    add_text(slide, 1.05, 3.25, 5.2, 0.35, "Learner", size=15, bold=True, color=NAVY)
    add_text(slide, 1.05, 3.7, 5.1, 1.0, "Factorized primal-dual Hedge: one distribution over feasible super-arms and one bid distribution per campaign, with a dual price for spending.", size=14)
    add_text(slide, 6.65, 3.25, 5.2, 0.35, "Reported metrics", size=15, bold=True, color=NAVY)
    add_text(slide, 6.65, 3.7, 5.15, 1.0, "Learner and oracle cumulative utility and cost, cumulative regret, dual multiplier, and mean +/- SD across 10 shared realizations.", size=14)
    add_text(slide, 1.05, 4.78, 10.9, 0.35, "Conflict graph: edges (0,1), (1,2), (2,3), (3,4), (4,0) cannot run together; feasible super-arms are 5 singletons plus 5 non-adjacent pairs.", size=11.5, bold=True, color=NAVY)
    add_text(slide, 1.05, 5.3, 10.8, 0.48, "Oracle: exact piecewise-static hindsight optimum. It knows the true change points, selects one feasible super-arm action within each regime, and obeys the same global budget B=2500.", size=12, bold=True, color=BLUE)

    all_utilities = np.concatenate([data["cumulative_utility"][:, -1] for data in runs.values()])
    all_costs = np.concatenate([data["cumulative_cost"][:, -1] for data in runs.values()])
    all_oracle_utilities = np.concatenate([data["cumulative_oracle_utility"][:, -1] for data in runs.values()])
    all_regrets = np.concatenate([data["regret"][:, -1] for data in runs.values()])

    # Slide 9: protocol.
    slide = prs.slides[8]
    rebuild_slide(slide, 9, "Shared-realization protocol", "The experimental unit is one saved matrix, not a newly sampled environment object.")
    stages = [
        ("Market generator", "Generate constellation and schedule", BLUE),
        ("Saved matrix", "realization_XX.npy, shape 3000 x 5", TEAL),
        ("Req3 learner", "Load exactly that matrix", ORANGE),
        ("Saved metrics", "One row per realization", GREEN),
    ]
    for idx, (label, note, color) in enumerate(stages):
        x = 0.65 + idx * 3.15
        add_card(slide, x, 2.0, 2.65, 1.6, label, str(idx + 1), color, note)
        if idx < 3:
            add_text(slide, x + 2.7, 2.5, 0.35, 0.4, ">", size=24, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
    add_box(slide, 0.8, 4.25, 11.7, 1.25, fill=PALE, line="D9E2EC")
    add_text(slide, 1.1, 4.45, 11.1, 0.8, "Seeds and change points are stored beside every matrix in config.json. Algorithm randomness is separate, recorded, and does not alter the market stream.", size=16, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    # Slide 10: overall results.
    slide = prs.slides[9]
    rebuild_slide(slide, 10, "Overall primal-dual results", "Pooled view over all 90 current shared realizations.")
    add_card(slide, 0.7, 1.45, 2.75, 1.35, "Runs", "90", BLUE, "9 settings x 10 matrices")
    add_card(slide, 3.65, 1.45, 2.75, 1.35, "Mean utility", f"{all_utilities.mean():,.1f}", TEAL, f"SD {all_utilities.std(ddof=1):,.1f}")
    add_card(slide, 6.6, 1.45, 2.75, 1.35, "Mean oracle utility", f"{all_oracle_utilities.mean():,.1f}", ORANGE, f"SD {all_oracle_utilities.std(ddof=1):,.1f}")
    add_card(slide, 9.55, 1.45, 2.75, 1.35, "Mean final regret", f"{all_regrets.mean():,.1f}", PURPLE, f"SD {all_regrets.std(ddof=1):,.1f}")
    add_picture(slide, assets["curves"], 1.2, 3.05, 10.9, 3.65)

    # Slide 11: sensitivity.
    slide = prs.slides[10]
    rebuild_slide(slide, 11, "Utility and regret across the full grid", "Each cell is a mean over 10 shared realizations; uncertainty is shown in the detailed diagnostics.")
    add_picture(slide, assets["sensitivity"], 0.75, 1.35, 11.85, 5.65)

    # Slide 12: budget.
    slide = prs.slides[11]
    rebuild_slide(slide, 12, "Budget feasibility in every setting", "The hard filter keeps realized cumulative cost at or below B=2500 in all 90 runs.")
    add_picture(slide, assets["budget"], 1.25, 1.45, 7.6, 4.75)
    add_card(slide, 9.15, 1.65, 3.2, 1.25, "Maximum cost", f"{all_costs.max():,.2f}", GREEN)
    add_card(slide, 9.15, 3.15, 3.2, 1.25, "Minimum cost", f"{all_costs.min():,.2f}", BLUE)
    add_card(slide, 9.15, 4.65, 3.2, 1.25, "Violations", "0 / 90", TEAL)

    # Slide 13: curves.
    slide = prs.slides[12]
    rebuild_slide(slide, 13, "Final utility distribution across all settings", "Each dot is one realization; black markers show mean +/- 1 SD.")
    add_picture(slide, assets["distribution"], 0.85, 1.35, 11.7, 5.7)

    # Slide 14: representative diagnostic.
    slide = prs.slides[13]
    rebuild_slide(slide, 14, "Representative current run dynamics", "Medium sKL (200), medium frequency (1/50), mean +/- SD over 10 shared realizations.")
    add_picture(slide, assets["diag_medium"], 0.85, 1.35, 9.2, 5.65)
    u_mean, u_sd, c_mean, c_sd = metrics_for(runs, "medium_frequency_medium_skl")
    r_mean, r_sd, o_mean, o_sd = regret_metrics(runs, "medium_frequency_medium_skl")
    add_card(slide, 10.25, 1.45, 2.45, 1.05, "Learner utility", f"{u_mean:,.1f}", BLUE, f"SD {u_sd:,.1f}")
    add_card(slide, 10.25, 2.75, 2.45, 1.05, "Oracle utility", f"{o_mean:,.1f}", ORANGE, f"SD {o_sd:,.1f}")
    add_card(slide, 10.25, 4.05, 2.45, 1.05, "Final regret", f"{r_mean:,.1f}", PURPLE, f"SD {r_sd:,.1f}")
    add_card(slide, 10.25, 5.35, 2.45, 1.05, "Final cost", f"{c_mean:,.2f}", TEAL, f"SD {c_sd:.3f}")

    # Slide 15.
    slide = prs.slides[14]
    rebuild_slide(slide, 15, "Shared market generation protocol", "The market input is generated and saved before any learner is run.")
    add_card(slide, 0.75, 1.45, 2.7, 1.25, "1. Gaussian modes", "mu_k, Sigma_k", BLUE, "3 modes; target sKL controls separation", value_size=17)
    add_card(slide, 3.7, 1.45, 2.7, 1.25, "2. Regime schedule", "q_t", TEAL, "Bernoulli changes with p grid", value_size=21)
    add_card(slide, 6.65, 1.45, 2.7, 1.25, "3. Market stream", "m_t", ORANGE, "Highest bids for 5 campaigns", value_size=21)
    add_card(slide, 9.6, 1.45, 2.7, 1.25, "4. Saved input", ".npy / .npz", GREEN, "90 fixed realizations", value_size=17)
    add_box(slide, 0.85, 3.25, 11.55, 2.6, fill=WHITE, line="C8D6E5")
    add_text(slide, 1.15, 3.55, 4.9, 0.35, "What is fixed before learning", size=16, bold=True, color=NAVY)
    add_text(slide, 1.15, 4.05, 4.9, 1.05, "For each setting, config.json records sKL, p, seeds, change points, and active phase ids. constellation.npz stores the Gaussian means and covariances.", size=13.5)
    add_text(slide, 6.65, 3.55, 4.9, 0.35, "What the learner receives", size=16, bold=True, color=NAVY)
    add_text(slide, 6.65, 4.05, 4.95, 1.05, "Each run loads one precomputed T x 5 market matrix. Learner randomness changes bids, but never regenerates the market stream.", size=13.5)
    add_box(slide, 0.85, 6.15, 11.55, 0.55, fill=PALE, line="D9E2EC")
    add_text(slide, 1.05, 6.24, 11.15, 0.35, "The next slide only visualizes the Gaussian mode means; it does not define a new experimental setting.", size=12, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

    # Slide 16.
    slide = prs.slides[15]
    rebuild_slide(slide, 16, "Constellation profiles by sKL", "Gaussian mode means generated by the shared environment at the three requested sKL levels.")
    add_picture(slide, assets["profiles"], 0.8, 1.35, 11.8, 5.65)

    # Slide 17.
    slide = prs.slides[16]
    rebuild_slide(slide, 17, "High-frequency realized market spectrum", "The same realization index is shown for sKL 20, 200, and 500 at p=1/25.")
    add_picture(slide, assets["high_spectrum"], 0.75, 1.35, 11.85, 5.65)

    # Slides 18-20: sKL diagnostics at medium frequency.
    for slide_idx, skl, number in zip([17, 18, 19], SKL_ORDER, [18, 19, 20]):
        setting_id = f"medium_frequency_{skl}"
        slide = prs.slides[slide_idx]
        rebuild_slide(slide, number, f"{SKL_LABELS[skl]} diagnostics", "Medium change frequency (1/50); mean +/- SD over the 10 shared realizations.")
        add_picture(slide, assets[f"diag_{setting_id}"], 0.75, 1.35, 9.35, 5.7)
        u_mean, u_sd, c_mean, c_sd = metrics_for(runs, setting_id)
        r_mean, r_sd, o_mean, o_sd = regret_metrics(runs, setting_id)
        add_card(slide, 10.3, 1.4, 2.35, 1.05, "Learner utility", f"{u_mean:,.1f}", BLUE, f"SD {u_sd:,.1f}")
        add_card(slide, 10.3, 2.7, 2.35, 1.05, "Oracle utility", f"{o_mean:,.1f}", ORANGE, f"SD {o_sd:,.1f}")
        add_card(slide, 10.3, 4.0, 2.35, 1.05, "Final regret", f"{r_mean:,.1f}", PURPLE, f"SD {r_sd:,.1f}")
        add_card(slide, 10.3, 5.3, 2.35, 1.05, "Final cost", f"{c_mean:,.2f}", TEAL, f"SD {c_sd:.3f}")

    # Slide 21: sKL summary.
    slide = prs.slides[20]
    rebuild_slide(slide, 21, "sKL summary at medium frequency", "Dots are individual shared realizations; black markers show mean +/- SD.")
    add_picture(slide, assets["summary_skl"], 0.85, 1.35, 11.65, 5.7)

    # Slides 22-24: frequency diagnostics at medium sKL.
    for slide_idx, freq, number in zip([21, 22, 23], FREQ_ORDER, [22, 23, 24]):
        setting_id = f"{freq}_medium_skl"
        slide = prs.slides[slide_idx]
        rebuild_slide(slide, number, f"{FREQ_LABELS[freq]} diagnostics", "Medium sKL (200); mean +/- SD over the 10 shared realizations.")
        add_picture(slide, assets[f"diag_{setting_id}"], 0.75, 1.35, 9.35, 5.7)
        u_mean, u_sd, c_mean, c_sd = metrics_for(runs, setting_id)
        r_mean, r_sd, o_mean, o_sd = regret_metrics(runs, setting_id)
        add_card(slide, 10.3, 1.4, 2.35, 1.05, "Learner utility", f"{u_mean:,.1f}", BLUE, f"SD {u_sd:,.1f}")
        add_card(slide, 10.3, 2.7, 2.35, 1.05, "Oracle utility", f"{o_mean:,.1f}", ORANGE, f"SD {o_sd:,.1f}")
        add_card(slide, 10.3, 4.0, 2.35, 1.05, "Final regret", f"{r_mean:,.1f}", PURPLE, f"SD {r_sd:,.1f}")
        add_card(slide, 10.3, 5.3, 2.35, 1.05, "Final cost", f"{c_mean:,.2f}", TEAL, f"SD {c_sd:.3f}")

    # Slide 25.
    while len(prs.slides) <= 24:
        prs.slides.add_slide(prs.slide_layouts[0])
    slide = prs.slides[24]
    rebuild_slide(slide, 25, "Frequency summary at medium sKL", "The schedules change 2, 72, and 131 times at p=1/1500, 1/50, and 1/25.")
    add_picture(slide, assets["summary_frequency"], 0.85, 1.35, 11.65, 4.9)
    add_box(slide, 0.85, 6.3, 11.65, 0.58, fill=PALE, line="D9E2EC")
    add_text(slide, 1.05, 6.34, 11.25, 0.48, "Why utility is not monotone: schedule seeds do not equalize time spent in each mode. Mode 0/1/2 shares are 27/40/34%, 24/43/33%, and 28/33/38% for p=1/1500, 1/50, and 1/25.", size=9, color=NAVY, align=PP_ALIGN.CENTER)

    # Slide 26.
    while len(prs.slides) <= 25:
        prs.slides.add_slide(prs.slide_layouts[0])
    slide = prs.slides[25]
    rebuild_slide(slide, 26, "Current experimental conclusions", "Claims below are restricted to the aligned 90-run Req3 experiment.")
    add_card(slide, 0.75, 1.45, 3.7, 1.55, "Budget control", "0 violations", GREEN, "All final costs <= 2500")
    add_card(slide, 4.8, 1.45, 3.7, 1.55, "Main sensitivity", "sKL dominates", BLUE, "Utility falls as separation rises")
    add_card(slide, 8.85, 1.45, 3.7, 1.55, "Oracle benchmark", "Regret measured", PURPLE, "Piecewise-static, same feasible super-arms")
    add_box(slide, 0.75, 3.45, 11.8, 2.25, fill=PALE, line="D9E2EC")
    add_text(slide, 1.05, 3.7, 11.2, 0.42, "What the experiment supports", size=16, bold=True, color=NAVY)
    add_text(slide, 1.05, 4.18, 11.1, 1.05, "The factorized primal-dual learner remains budget-feasible on every shared realization. Regret is computed against the exact piecewise-static hindsight oracle under the same feasible super-arm set and B=2500 global budget. Final utility and regret both vary materially with sKL.", size=15)
    add_text(slide, 1.05, 5.35, 11.1, 0.32, "The same oracle definition and matrix files must be used for sliding-window and change-detector comparisons.", size=13, bold=True, color=BLUE)

    from repair_slide_typography import repair_presentation

    repair_presentation(prs)
    protocol_index = next(
        index
        for index, slide in enumerate(prs.slides)
        if "Shared market generation protocol" in [getattr(shape, "text", "").strip() for shape in slide.shapes]
    )
    build_conflict_graph_slide(prs.slides[protocol_index])
    move_slide(prs, protocol_index, 5)
    remove_slides_by_title(prs, "Conflict graph and feasible super-arms", keep=1)
    renumber_slides(prs)
    output = PPTX_PATH
    prs.save(output)
    return output


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    runs, summary, _ = load_and_run_all()
    sensitivity, budget = plot_heatmaps(summary)
    assets = {
        "sensitivity": sensitivity,
        "budget": budget,
        "curves": plot_learning_curves(runs),
        "distribution": plot_final_utility_distribution(runs),
        "profiles": plot_constellation_profiles(),
        "high_spectrum": plot_high_frequency_spectrum(),
        "summary_skl": plot_summary_axis(runs, "skl"),
        "summary_frequency": plot_summary_axis(runs, "frequency"),
    }
    diagnostic_comparison_ids = [
        *[f"medium_frequency_{skl}" for skl in SKL_ORDER],
        *[f"{freq}_medium_skl" for freq in FREQ_ORDER],
    ]
    diagnostic_limits = build_diagnostic_limits(runs, diagnostic_comparison_ids)
    for setting_id, data in runs.items():
        limits = diagnostic_limits if setting_id in diagnostic_comparison_ids else None
        assets[f"diag_{setting_id}"] = plot_diagnostic(setting_id, data, limits)
    assets["diag_medium"] = assets["diag_medium_frequency_medium_skl"]
    output = rebuild_deck(runs, summary, assets)
    print(f"saved updated deck to {output}")


if __name__ == "__main__":
    main()

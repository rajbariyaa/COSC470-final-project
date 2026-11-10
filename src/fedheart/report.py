"""Publication tables (CSV, Markdown, LaTeX/booktabs) and figures (PNG + PDF).

Everything here is computed from files saved by the pipeline; no number is typed by hand.

Colour: the three methods use the first three slots of a categorical palette validated for
colour-vision deficiency on every pairing (blue, orange, aqua); the methods also differ by
marker shape, every figure has a legend, and each figure has a companion table, so colour
never carries identity alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from fedheart.data import HOSPITAL_LABELS, HOSPITALS, PREDICTORS  # noqa: E402
from fedheart.metrics import METHOD_LABELS, METHODS, REPORTED_METRICS, SCOPES  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
METHOD_COLORS = {"centralized": "#2a78d6", "fedavg": "#eb6834", "fedavg_ft": "#1baf7a"}
METHOD_MARKERS = {"centralized": "o", "fedavg": "s", "fedavg_ft": "^"}
CLASS_COLORS = {"No disease": "#2a78d6", "Disease": "#eb6834"}
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SCOPE_LABELS = {**HOSPITAL_LABELS, "pooled": "Pooled", "equal_hospital_mean": "Equal-hospital mean"}
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "specificity": "Specificity", "f1": "F1-score", "roc_auc": "ROC-AUC",
}  # fmt: skip


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": BASELINE, "ytick.color": BASELINE,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 9,
        "font.family": "sans-serif", "lines.linewidth": 1.5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })  # fmt: skip


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


def _method_handles():
    return [
        plt.Line2D([], [], color=METHOD_COLORS[m], marker=METHOD_MARKERS[m], linestyle="-",
                   markersize=6, label=METHOD_LABELS[m])
        for m in METHODS
    ]  # fmt: skip


# --------------------------------------------------------------------------------------
# Table writers
# --------------------------------------------------------------------------------------


def fmt_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if pd.notna(sd) else f"{mean:.{digits}f}"


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("±", r"$\pm$"), ("−", "$-$"), ("≥", r"$\geq$")):  # fmt: skip
        s = s.replace(a, b)
    return s


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    align = "l" * min(2, len(df.columns)) + "r" * max(0, len(df.columns) - 2)
    body = [" & ".join(_tex_escape(str(c)) for c in df.columns) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        body.append(" & ".join("" if pd.isna(v) else _tex_escape(str(v)) for v in row) + r" \\")
    return "\n".join([
        r"\begin{table}[ht]", r"\centering", r"\small",
        rf"\caption{{{_tex_escape(caption)}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule", *body, r"\bottomrule",
        r"\end{tabular}", r"\end{table}", "",
    ])  # fmt: skip


def write_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(f"**{caption}**\n\n" + to_markdown(df))
    (out_dir / f"{name}.tex").write_text(to_latex(df, caption, f"tab:{name}"))


# --------------------------------------------------------------------------------------
# Data audit document
# --------------------------------------------------------------------------------------


def write_audit_markdown(audit: dict, feasibility: pd.DataFrame, manifest: dict, path: Path):
    fa = audit["feature_audit"]
    miss = fa.pivot(index="hospital", columns="feature", values="pct_missing")
    miss = miss.loc[list(HOSPITALS), list(PREDICTORS)].round(1).reset_index()
    files = pd.DataFrame([
        {"hospital": h, "file": v["path"], "records": v["records"], "sha256": v["sha256"]}
        for h, v in manifest["files"].items()
    ])  # fmt: skip
    numeric = fa[fa["type"] == "numeric"][
        ["hospital", "feature", "n_observed", "min", "max", "mean", "n_negative",
         "n_outside_plausible_range"]
    ].round(2)  # fmt: skip
    categorical = fa[fa["type"] == "categorical"][["hospital", "feature", "levels_observed"]]
    log = audit["cleaning_log"]
    zero_rules = log[log["rule"].isin(["C2", "C3", "C4"])]
    cross = int(audit["cross_hospital"].iloc[0, 0])
    text = [
        "# Data audit (generated)",
        "",
        "Generated by `scripts/01_download_audit.py` from the UCI files; do not edit by hand.",
        f"Source: <{manifest['source']}>. "
        f"Retrieved/verified: {manifest['retrieved_or_verified_at']}.",
        "",
        "## Files and checksums",
        "",
        to_markdown(files),
        "## Per-hospital summary",
        "",
        "`n_extra_duplicate_copies` counts rows that are exact copies of another row in the "
        "same hospital (kept, but always assigned to the same partition). "
        f"Identical records across hospitals: {cross}.",
        "",
        to_markdown(audit["hospital_summary"].round(3)),
        "## Percentage missing per predictor (after cleaning)",
        "",
        "Includes the `?` markers and the impossible zeros converted by rules C2/C3.",
        "",
        to_markdown(miss),
        "## Cleaning rules applied (rows affected)",
        "",
        to_markdown(zero_rules[["hospital", "rule", "column", "n_rows", "action", "rationale"]]),
        "## Exact duplicate records",
        "",
        to_markdown(audit["duplicates"]),
        "## Numeric predictors (observed values after cleaning)",
        "",
        to_markdown(numeric),
        "## Categorical predictors (observed levels: count)",
        "",
        to_markdown(categorical),
        "## Can each hospital support a stratified 60/20/20 split?",
        "",
        "Expected minority-class count per partition; feasible when every value is at least 1.",
        "",
        to_markdown(feasibility),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text))


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


def dataset_table(audit_summary: pd.DataFrame, feature_audit: pd.DataFrame,
                  split_summary: pd.DataFrame) -> pd.DataFrame:  # fmt: skip
    miss = feature_audit.pivot(index="hospital", columns="feature", values="pct_missing")
    sizes = split_summary.groupby(["hospital", "partition"])["n"].mean().unstack()
    rows = []
    for _, r in audit_summary.set_index("hospital").loc[list(HOSPITALS)].iterrows():
        h = r.name
        rows.append({
            "Hospital": HOSPITAL_LABELS[h],
            "Records": int(r["n_records"]),
            "Disease, n (%)": f"{int(r['n_positive'])} ({100 * r['prevalence']:.1f})",
            "No disease, n": int(r["n_negative"]),
            "Duplicate copies": int(r["n_extra_duplicate_copies"]),
            "chol missing %": f"{miss.loc[h, 'chol']:.0f}",
            "slope missing %": f"{miss.loc[h, 'slope']:.0f}",
            "ca missing %": f"{miss.loc[h, 'ca']:.0f}",
            "thal missing %": f"{miss.loc[h, 'thal']:.0f}",
            "Train/val/test rows": "/".join(
                f"{sizes.loc[h, p]:.0f}" for p in ("train", "val", "test")
            ),
        })  # fmt: skip
    return pd.DataFrame(rows)


def results_table(summary: pd.DataFrame, metrics=REPORTED_METRICS) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        for m in METHODS:
            r = summary[(summary["scope"] == scope) & (summary["method"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            row = {"Scope": SCOPE_LABELS[scope], "Method": METHOD_LABELS[m]}
            for k in metrics:
                cell = fmt_mean_sd(r[f"{k}_mean"], r[f"{k}_sd"])
                if r[f"{k}_n_defined"] < r["n_seeds"]:
                    cell += f" (n={int(r[f'{k}_n_defined'])})"
                row[METRIC_LABELS[k]] = cell
            rows.append(row)
    return pd.DataFrame(rows)


def paired_table(paired: pd.DataFrame, metrics=("accuracy", "f1", "roc_auc")) -> pd.DataFrame:
    rows = []
    for scope in SCOPES:
        for comp in paired["comparison"].unique():
            sub = paired[(paired["scope"] == scope) & (paired["comparison"] == comp)]
            if sub.empty:
                continue
            a, b = sub.iloc[0]["a"], sub.iloc[0]["b"]
            row = {"Scope": SCOPE_LABELS[scope],
                   "Comparison": f"{METHOD_LABELS[a]} − {METHOD_LABELS[b]}"}  # fmt: skip
            for k in metrics:
                r = sub[sub["metric"] == k].iloc[0]
                if pd.isna(r["mean_diff"]):
                    row[METRIC_LABELS[k]] = "n/a"
                    continue
                row[METRIC_LABELS[k]] = (
                    f"{r['mean_diff']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}] "
                    f"({int(r['wins'])}/{int(r['losses'])}/{int(r['ties'])})"
                )
            rows.append(row)
    return pd.DataFrame(rows)


def selection_table(validation: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for _, r in validation[m].iterrows():
            rows.append({
                "Method": METHOD_LABELS[m], "Learning rate": f"{r['lr']:g}",
                "Validation BCE (mean ± SD over seeds)":
                    fmt_mean_sd(r["val_loss_mean"], r["val_loss_sd"], 4),
                "Selected": "yes" if r["selected"] else "",
            })  # fmt: skip
    return pd.DataFrame(rows)


def budget_table(cfg: dict, summaries: dict[str, pd.DataFrame], selection: dict) -> pd.DataFrame:
    seeds = len(cfg["seeds"])
    c = summaries["centralized"]
    c_sel = c[c["lr"] == selection["centralized"]["lr"]]
    f = summaries["fedavg"]
    f_sel = f[f["lr"] == selection["fedavg"]["lr"]]
    t = summaries["fedavg_ft"]
    t_sel = t[t["lr"] == selection["fedavg_ft"]["lr"]]
    t_seed = t_sel.groupby("seed")[["budget_optimizer_steps", "budget_examples_processed"]].sum()

    def rng(s):
        return f"{s.median():.0f} [{s.min():.0f}–{s.max():.0f}]"

    fa = cfg["training"]["fedavg"]
    f_steps = f_sel["budget_optimizer_steps"].mean()
    f_mb = (f_sel["budget_bytes_training"] + f_sel["budget_bytes_evaluation"]).mean() / 1e6
    t_steps = t_seed["budget_optimizer_steps"].mean()

    return pd.DataFrame([
        {
            "Method": METHOD_LABELS["centralized"],
            "Learning rates tried": len(cfg["training"]["centralized"]["lr_grid"]),
            "Runs (all configs)": len(c),
            "Training per run": f"{cfg['training']['centralized']['max_epochs']} epochs",
            "Optimizer steps per run": f"{c_sel['budget_optimizer_steps'].mean():.0f}",
            "Examples processed per run": f"{c_sel['budget_examples_processed'].mean():.0f}",
            "Selected epoch/round, median [range]": rng(c_sel["best_epoch"]),
            "Model transfer per run (MB)": "0 (data pooled)",
        },
        {
            "Method": METHOD_LABELS["fedavg"],
            "Learning rates tried": len(cfg["training"]["fedavg"]["lr_grid"]),
            "Runs (all configs)": len(f),
            "Training per run": (f"{fa['num_rounds']} rounds × {fa['local_epochs']} local "
                                 "epoch, 4 clients"),
            "Optimizer steps per run": f"{f_steps:.0f} (all clients)",
            "Examples processed per run": f"{f_sel['budget_examples_processed'].mean():.0f}",
            "Selected epoch/round, median [range]": rng(f_sel["best_round"]),
            "Model transfer per run (MB)": f"{f_mb:.2f}",
        },
        {
            "Method": METHOD_LABELS["fedavg_ft"] + " (fine-tuning stage only)",
            "Learning rates tried": len(cfg["training"]["finetune"]["lr_grid"]),
            "Runs (all configs)": f"{len(t)} ({seeds} seeds × 4 hospitals × lr)",
            "Training per run": f"≤{cfg['training']['finetune']['max_epochs']} epochs per hospital",
            "Optimizer steps per run": f"{t_steps:.0f} (4 hospitals)",
            "Examples processed per run": f"{t_seed['budget_examples_processed'].mean():.0f}",
            "Selected epoch/round, median [range]": rng(t_sel["best_epoch"]),
            "Model transfer per run (MB)": "0 (local only)",
        },
    ])  # fmt: skip


def finetune_epoch_table(ft_selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HOSPITALS:
        s = ft_selected[ft_selected["hospital"] == h]
        rows.append({
            "Hospital": HOSPITAL_LABELS[h],
            "Seeds": len(s),
            "Kept global model (epoch 0)": int((s["best_epoch"] == 0).sum()),
            "Selected epoch, median [range]":
                f"{s['best_epoch'].median():.0f} [{s['best_epoch'].min()}–{s['best_epoch'].max()}]",
        })  # fmt: skip
    return pd.DataFrame(rows)


def calibration_table(cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HOSPITALS:
        sub = cal[cal["hospital"] == h].set_index("method")
        row = {"Hospital": HOSPITAL_LABELS[h],
               "Observed prevalence (test)": f"{sub['observed'].iloc[0]:.3f}"}  # fmt: skip
        for m in METHODS:
            row[f"Mean predicted: {METHOD_LABELS[m]}"] = f"{sub.loc[m, 'predicted']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_table(conf: pd.DataFrame) -> pd.DataFrame:
    out = conf.copy()
    out["method"] = out["method"].map(METHOD_LABELS)
    out["scope"] = out["scope"].map(SCOPE_LABELS)
    return out.rename(columns={"method": "Method", "scope": "Scope", "tn": "TN", "fp": "FP",
                               "fn": "FN", "tp": "TP"})  # fmt: skip


def undefined_table(metrics_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    per = metrics_long[metrics_long["scope"].isin(HOSPITALS + ("pooled",))]
    for (m, scope), g in per.groupby(["method", "scope"], sort=False):
        for k in ("precision", "recall", "specificity", "f1", "roc_auc"):
            n = int(g[k].isna().sum())
            if n:
                rows.append({"Method": METHOD_LABELS[m], "Scope": SCOPE_LABELS[scope],
                             "Metric": METRIC_LABELS[k], "Seeds undefined": n})  # fmt: skip
    return pd.DataFrame(rows, columns=["Method", "Scope", "Metric", "Seeds undefined"])


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_class_balance(audit_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    s = audit_summary.set_index("hospital").loc[list(HOSPITALS)[::-1]]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    y = np.arange(len(s))
    neg, pos = s["n_negative"].to_numpy(), s["n_positive"].to_numpy()
    ax.barh(y, neg, color=CLASS_COLORS["No disease"], height=0.6, edgecolor=SURFACE,
            linewidth=1.5, label="No disease (num = 0)")  # fmt: skip
    ax.barh(y, pos, left=neg, color=CLASS_COLORS["Disease"], height=0.6, edgecolor=SURFACE,
            linewidth=1.5, label="Disease (num 1–4)")  # fmt: skip
    for yi, n0, n1 in zip(y, neg, pos, strict=True):
        ax.text(n0 + n1 + 4, yi, f"{n1}/{n0 + n1} disease ({100 * n1 / (n0 + n1):.0f}%)",
                va="center", fontsize=8, color=INK2)  # fmt: skip
    ax.set_yticks(y, [HOSPITAL_LABELS[h] for h in s.index])
    ax.set_xlabel("Patients")
    ax.set_xlim(0, (neg + pos).max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax.set_title("Class balance differs sharply between hospitals", loc="left", pad=22)
    return save_figure(fig, out_dir, "fig1_class_balance")


def fig_missingness(feature_audit: pd.DataFrame, out_dir: Path) -> list[Path]:
    m = feature_audit.pivot(index="hospital", columns="feature", values="pct_missing")
    m = m.loc[list(HOSPITALS), list(PREDICTORS)]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    im = ax.imshow(m.to_numpy(), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v > 55 else INK)  # fmt: skip
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=0)
    ax.set_yticks(range(m.shape[0]), [HOSPITAL_LABELS[h] for h in m.index])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("% missing", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Missing values per predictor after cleaning (%)", loc="left")
    return save_figure(fig, out_dir, "fig2_missingness")


def _band(ax, x, values: np.ndarray, method: str, label: str) -> None:
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.plot(x, mean, color=METHOD_COLORS[method], label=label)
    ax.fill_between(x, mean - sd, mean + sd, color=METHOD_COLORS[method], alpha=0.15, linewidth=0)


def fig_validation_curves(histories: dict[str, list[pd.DataFrame]], out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    c = np.stack([h.loc[h["epoch"] >= 1, "val_loss"].to_numpy() for h in histories["centralized"]])
    f = np.stack([h["val_loss"].to_numpy() for h in histories["fedavg"]])
    _band(ax, np.arange(1, c.shape[1] + 1), c, "centralized", "Centralized (x = epoch)")
    _band(ax, np.arange(1, f.shape[1] + 1), f, "fedavg", "FedAvg (x = communication round)")
    ax.set_xlabel("Epoch (centralized) or round of 1 local epoch per hospital (FedAvg)")
    ax.set_ylabel("Pooled validation BCE")
    ax.legend(loc="upper right")
    ax.set_title("Validation loss at the selected learning rates (mean ± SD over seeds)",
                 loc="left")  # fmt: skip
    return save_figure(fig, out_dir, "fig3_validation_curves")


def fig_finetune_curves(ft_histories: dict[str, list[pd.DataFrame]], out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6), sharex=True)
    for ax, h in zip(axes, HOSPITALS, strict=True):
        v = np.stack([d["val_loss"].to_numpy() for d in ft_histories[h]])
        delta = v - v[:, :1]
        _band(ax, np.arange(v.shape[1]), delta, "fedavg_ft", "")
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_title(HOSPITAL_LABELS[h], loc="left", fontsize=9)
        ax.set_xlabel("Fine-tuning epoch")
    axes[0].set_ylabel("Δ validation BCE vs\nglobal FedAvg model")
    fig.suptitle("Local fine-tuning: change in each hospital's validation loss "
                 "(mean ± SD; y-axes differ)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)  # fmt: skip
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig4_finetune_curves")


def fig_test_metrics(metrics_long: pd.DataFrame, out_dir: Path, n_seeds: int) -> list[Path]:
    metrics = REPORTED_METRICS
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.5, 2.0 * len(metrics)), sharex=True)
    offsets = dict(zip(METHODS, (-0.24, 0.0, 0.24), strict=True))
    rng = np.random.default_rng(0)
    for ax, k in zip(axes, metrics, strict=True):
        for j, scope in enumerate(SCOPES):
            for m in METHODS:
                v = metrics_long.loc[(metrics_long["scope"] == scope)
                                     & (metrics_long["method"] == m), k].dropna()  # fmt: skip
                x = j + offsets[m]
                ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=9, color=METHOD_COLORS[m],
                           alpha=0.35, linewidths=0, marker=METHOD_MARKERS[m])  # fmt: skip
                if len(v):
                    sd = v.std(ddof=1) if len(v) > 1 else 0.0
                    ax.errorbar(x, v.mean(), yerr=sd, color=METHOD_COLORS[m],
                                marker=METHOD_MARKERS[m], markersize=6, linewidth=1.2,
                                markeredgecolor=SURFACE, markeredgewidth=0.8)  # fmt: skip
                if len(v) < n_seeds:
                    ax.text(x, 0.02, f"n={len(v)}", ha="center", fontsize=6.5, color=MUTED)
        ax.axvline(len(HOSPITALS) - 0.5, color=BASELINE, linewidth=1)
        # Dot plots need no zero baseline; start just below the lowest plotted value.
        lowest = metrics_long[k].min()
        ax.set_ylim(max(0.0, np.floor((lowest - 0.05) * 10) / 10) if pd.notna(lowest) else 0, 1.03)
        ax.set_ylabel(METRIC_LABELS[k])
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(range(len(SCOPES)), [SCOPE_LABELS[s].replace(" ", "\n", 1)
                                              if s == "equal_hospital_mean" else SCOPE_LABELS[s]
                                              for s in SCOPES])  # fmt: skip
    axes[0].legend(handles=_method_handles(), loc="lower center", bbox_to_anchor=(0.5, 1.02),
                   ncol=3)  # fmt: skip
    fig.suptitle(f"Test performance per hospital and overall ({n_seeds} seeds; "
                 "small points = seeds, large = mean ± SD)", x=0.01, ha="left", fontsize=10,
                 fontweight="bold", color=INK)  # fmt: skip
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig5_test_metrics")


def fig_paired_differences(paired: pd.DataFrame, out_dir: Path) -> list[Path]:
    metrics = ("accuracy", "f1", "roc_auc")
    comps = list(paired["comparison"].unique())
    fig, axes = plt.subplots(len(metrics), len(comps), figsize=(9, 7), sharey=True)
    y = np.arange(len(SCOPES))[::-1]
    for i, k in enumerate(metrics):
        bounds = paired.loc[paired["metric"] == k, ["ci95_low", "ci95_high", "mean_diff"]]
        lim = np.nanmax(np.abs(bounds.to_numpy())) if bounds.notna().any().any() else 0.0
        for j, comp in enumerate(comps):
            ax = axes[i, j]
            sub = paired[(paired["metric"] == k) & (paired["comparison"] == comp)]
            sub = sub.set_index("scope").loc[list(SCOPES)]
            ax.axvline(0, color=BASELINE, linewidth=1.2)
            ax.hlines(y, sub["ci95_low"], sub["ci95_high"], color=INK2, linewidth=1.5)
            ax.scatter(sub["mean_diff"], y, color=INK, s=22, zorder=3)
            ax.axhline(1.5, color=GRID, linewidth=0.8)
            ax.set_yticks(y, [SCOPE_LABELS[s] for s in SCOPES])
            ax.grid(axis="y", visible=False)
            if i == 0:
                a, b = comp.split(" - ")
                ax.set_title(f"{METHOD_LABELS[a]}\nminus {METHOD_LABELS[b]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[k])
            if np.isfinite(lim) and lim > 0:  # same symmetric range across a metric's row
                ax.set_xlim(-1.1 * lim, 1.1 * lim)
    fig.suptitle("Paired differences over seeds: mean and 95% CI (right of 0 favours the first)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)  # fmt: skip
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig6_paired_differences")


def fig_roc(predictions: pd.DataFrame, out_dir: Path) -> list[Path]:
    scopes = (*HOSPITALS, "pooled")
    fig, axes = plt.subplots(1, len(scopes), figsize=(12, 2.9), sharey=True)
    for ax, scope in zip(axes, scopes, strict=True):
        sub = predictions if scope == "pooled" else predictions[predictions["hospital"] == scope]
        ax.plot([0, 1], [0, 1], color=BASELINE, linewidth=1, linestyle="--")
        for m in METHODS:
            g = sub[sub["method"] == m]
            fpr, tpr, _ = roc_curve(g["y_true"], g["prob"])
            ax.plot(fpr, tpr, color=METHOD_COLORS[m], linewidth=1.5)
        n_neg = int((sub[sub["method"] == METHODS[0]]["y_true"] == 0).sum())
        ax.set_title(f"{SCOPE_LABELS[scope]}\n({n_neg} negatives, all seeds)", loc="left",
                     fontsize=9)  # fmt: skip
        ax.set_xlabel("False positive rate")
        ax.set_aspect("equal")
    axes[0].set_ylabel("True positive rate")
    fig.legend(handles=_method_handles(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig7_roc_curves")


def fig_confusion(conf: pd.DataFrame, out_dir: Path) -> list[Path]:
    scopes = (*HOSPITALS, "pooled")
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(len(METHODS), len(scopes), figsize=(11, 6.6))
    for i, m in enumerate(METHODS):
        for j, scope in enumerate(scopes):
            ax = axes[i, j]
            r = conf[(conf["method"] == m) & (conf["scope"] == scope)].iloc[0]
            mat = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
            rows = mat.sum(axis=1, keepdims=True)
            share = np.divide(mat, rows, out=np.zeros_like(mat), where=rows > 0)
            ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{int(mat[a, b])}\n({100 * share[a, b]:.0f}%)", ha="center",
                            va="center", fontsize=7.5,
                            color="#ffffff" if share[a, b] > 0.55 else INK)  # fmt: skip
            ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1], ["True 0", "True 1"])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(SCOPE_LABELS[scope], fontsize=9)
            if j == 0:
                ax.set_ylabel(METHOD_LABELS[m].replace(" with ", "\nwith "), fontsize=8)
    fig.suptitle("Confusion matrices summed over seeds (row % in brackets; threshold 0.5)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)  # fmt: skip
    fig.tight_layout()
    return save_figure(fig, out_dir, "fig8_confusion_matrices")

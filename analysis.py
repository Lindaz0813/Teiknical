"""
analysis.py
Runs Parts 2, 3, and 4 of the analysis pipeline.

Outputs:
  - outputs/part2_frequency_table.csv
  - outputs/part3_boxplot.png
  - outputs/part3_stats.csv
  - outputs/part4_subset_summary.txt

"""

import sqlite3
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

DB_PATH = "immune_trial.db"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Shared helper: load frequency table from DB
# ─────────────────────────────────────────────
def get_frequency_table(conn):
    """
    Part 2: For every sample, compute total cell count and
    the relative frequency (%) of each population.
    Returns a tidy DataFrame.
    """
    query = """
        SELECT
            s.sample_id                         AS sample,
            sub.subject_id,
            sub.condition,
            sub.treatment,
            sub.response,
            sub.sex,
            smp.sample_type,
            smp.time_from_treatment_start,
            cc.population,
            cc.count,
            SUM(cc.count) OVER (PARTITION BY s.sample_id) AS total_count
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        JOIN cell_counts cc ON cc.sample_id = s.sample_id
        -- alias to avoid ambiguity
        JOIN samples smp ON smp.sample_id = s.sample_id
    """
    df = pd.read_sql_query(query, conn)
    df["percentage"] = (df["count"] / df["total_count"] * 100).round(4)
    return df


# ─────────────────────────────────────────────
# Part 2
# ─────────────────────────────────────────────
def run_part2(conn):
    df = get_frequency_table(conn)
    out = df[["sample", "total_count", "population", "count", "percentage"]].copy()
    out = out.sort_values(["sample", "population"]).reset_index(drop=True)

    path = os.path.join(OUT_DIR, "part2_frequency_table.csv")
    out.to_csv(path, index=False)
    print(f"[Part 2] Frequency table saved → {path}")
    print(out.to_string(index=False))
    return df   # return full df for downstream use


# ─────────────────────────────────────────────
# Part 3
# ─────────────────────────────────────────────
def run_part3(df_full):
    """
    Compare relative frequencies between responders and non-responders
    for melanoma patients on miraclib, PBMC samples only.
    """
    mask = (
        (df_full["condition"] == "melanoma") &
        (df_full["treatment"] == "miraclib") &
        (df_full["sample_type"] == "PBMC")
    )
    df = df_full[mask].copy()

    populations = df["population"].unique()
    results = []

    fig, axes = plt.subplots(1, len(populations), figsize=(18, 6), sharey=False)
    fig.suptitle(
        "Cell Population Frequencies: Responders vs Non-Responders\n"
        "(Melanoma, Miraclib, PBMC)",
        fontsize=13, fontweight="bold"
    )

    colors = {"yes": "#4C9BE8", "no": "#E8714C"}

    for ax, pop in zip(axes, sorted(populations)):
        resp    = df[(df["population"] == pop) & (df["response"] == "yes")]["percentage"].values
        nonresp = df[(df["population"] == pop) & (df["response"] == "no" )]["percentage"].values

        # Mann-Whitney U (non-parametric, appropriate for small samples)
        if len(resp) >= 2 and len(nonresp) >= 2:
            stat, pval = stats.mannwhitneyu(resp, nonresp, alternative="two-sided")
        else:
            stat, pval = float("nan"), float("nan")

        results.append({
            "population": pop,
            "n_responders": len(resp),
            "n_non_responders": len(nonresp),
            "median_responders": round(float(pd.Series(resp).median()), 4),
            "median_non_responders": round(float(pd.Series(nonresp).median()), 4),
            "mannwhitney_stat": round(stat, 4) if not pd.isna(stat) else None,
            "p_value": round(pval, 4) if not pd.isna(pval) else None,
            "significant": "Yes" if (not pd.isna(pval) and pval < 0.05) else "No",
        })

        # Boxplot
        data_to_plot = [resp, nonresp]
        bp = ax.boxplot(
            data_to_plot,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor(colors["yes"])
        bp["boxes"][1].set_facecolor(colors["no"])

        ax.set_title(pop.replace("_", " ").title(), fontsize=10)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Responders", "Non-Resp."], fontsize=8)
        ax.set_ylabel("Frequency (%)" if pop == sorted(populations)[0] else "")

        sig_label = f"p={pval:.3f}" if not pd.isna(pval) else "p=N/A"
        ax.text(0.5, 0.97, sig_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=8,
                color="red" if (not pd.isna(pval) and pval < 0.05) else "gray")

    legend_patches = [
        mpatches.Patch(color=colors["yes"], label="Responders"),
        mpatches.Patch(color=colors["no"],  label="Non-Responders"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plot_path = os.path.join(OUT_DIR, "part3_boxplot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Part 3] Boxplot saved → {plot_path}")

    stats_df = pd.DataFrame(results)
    stats_path = os.path.join(OUT_DIR, "part3_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[Part 3] Stats table saved → {stats_path}")
    print("\nStatistical Results:")
    print(stats_df.to_string(index=False))

    sig = stats_df[stats_df["significant"] == "Yes"]["population"].tolist()
    if sig:
        print(f"\n  → Significant populations (p < 0.05): {', '.join(sig)}")
    else:
        print("\n  → No populations reached p < 0.05 significance.")

    return stats_df


# ─────────────────────────────────────────────
# Part 4
# ─────────────────────────────────────────────
def run_part4(conn, df_full):
    """
    Melanoma PBMC baseline (time=0) samples treated with miraclib.
    Aggregated breakdowns + average B cells for melanoma males at baseline.
    """
    # Query directly from DB for accuracy
    query = """
        SELECT
            sub.subject_id,
            sub.project,
            sub.sex,
            sub.response,
            sub.condition,
            sub.treatment,
            smp.sample_id,
            smp.sample_type,
            smp.time_from_treatment_start,
            cc.population,
            cc.count
        FROM subjects sub
        JOIN samples smp ON smp.subject_id = sub.subject_id
        JOIN cell_counts cc ON cc.sample_id = smp.sample_id
        WHERE
            sub.condition  = 'melanoma'
            AND smp.sample_type = 'PBMC'
            AND smp.time_from_treatment_start = 0
            AND sub.treatment = 'miraclib'
    """
    df = pd.read_sql_query(query, conn)

    lines = []
    lines.append("=" * 60)
    lines.append("PART 4 – Melanoma PBMC Baseline Miraclib Subset")
    lines.append("=" * 60)

    # Unique samples in subset
    samples_subset = df["sample_id"].unique()
    lines.append(f"\nTotal qualifying samples : {len(samples_subset)}")

    # Samples per project
    lines.append("\n--- Samples per project ---")
    per_project = (
        df[["sample_id", "project"]]
        .drop_duplicates()
        .groupby("project")["sample_id"]
        .count()
        .reset_index()
        .rename(columns={"sample_id": "n_samples"})
    )
    lines.append(per_project.to_string(index=False))

    # Subjects (unique) responder / non-responder
    lines.append("\n--- Subjects: responders vs non-responders ---")
    subj_resp = (
        df[["subject_id", "response"]]
        .drop_duplicates()
        .groupby("response")["subject_id"]
        .count()
        .reset_index()
        .rename(columns={"subject_id": "n_subjects"})
    )
    lines.append(subj_resp.to_string(index=False))

    # Subjects by sex
    lines.append("\n--- Subjects: males vs females ---")
    subj_sex = (
        df[["subject_id", "sex"]]
        .drop_duplicates()
        .groupby("sex")["subject_id"]
        .count()
        .reset_index()
        .rename(columns={"subject_id": "n_subjects"})
    )
    lines.append(subj_sex.to_string(index=False))

    # Average B cells — melanoma males, responders, time=0
    lines.append("\n--- Avg B cells: melanoma males, responders, time=0 ---")
    bcell_df = df[
        (df["sex"] == "M") &
        (df["response"] == "yes") &
        (df["population"] == "b_cell")
    ]
    if len(bcell_df) > 0:
        avg_bcell = bcell_df["count"].mean()
        lines.append(f"Average B cell count: {avg_bcell:.2f}")
    else:
        lines.append("No qualifying records found.")

    lines.append("\n" + "=" * 60)
    output = "\n".join(lines)
    print(f"\n[Part 4]\n{output}")

    out_path = os.path.join(OUT_DIR, "part4_subset_summary.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\n[Part 4] Summary saved → {out_path}")


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database '{DB_PATH}' not found. Run `python load_data.py` first."
        )

    conn = sqlite3.connect(DB_PATH)
    try:
        print("\n" + "=" * 60)
        print("PART 2 – Frequency Table")
        print("=" * 60)
        df_full = run_part2(conn)

        print("\n" + "=" * 60)
        print("PART 3 – Statistical Analysis")
        print("=" * 60)
        run_part3(df_full)

        print("\n" + "=" * 60)
        print("PART 4 – Subset Analysis")
        print("=" * 60)
        run_part4(conn, df_full)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

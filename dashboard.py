"""
dashboard.py  —  Teiko Technical Question
Run: python dashboard.py  ->  http://127.0.0.1:8050
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats

import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH     = "immune_trial.db"
POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
POP_LABELS  = {p: p.replace("_", " ").title() for p in POPULATIONS}
PALETTE       = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#76B7B2"]
PALETTE_LIGHT = ["rgba(78,121,167,0.15)", "rgba(242,142,43,0.15)",
                 "rgba(89,161,79,0.15)",  "rgba(176,122,161,0.15)",
                 "rgba(118,183,178,0.15)"]

BG          = "#F7F8FA"
WHITE       = "#FFFFFF"
BORDER      = "#E2E6EA"
TEXT_DARK   = "#1A1F36"
TEXT_MID    = "#4A5568"
TEXT_LIGHT  = "#8A94A6"
ACCENT      = "#2563EB"
ACCENT_SOFT = "#EBF0FD"
GREEN       = "#16A34A"
GREEN_SOFT  = "#F0FDF4"
GRAY_SOFT   = "#F1F3F5"

CHART_LAYOUT = dict(
    paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(family="Inter, Segoe UI, Arial", size=12, color=TEXT_DARK),
    margin=dict(t=48, b=64, l=52, r=24),
    xaxis=dict(showgrid=False, linecolor=BORDER, linewidth=1),
    yaxis=dict(gridcolor="#EAECEF", zeroline=False, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

# ── Layout helpers ─────────────────────────────────────────────────────────────
def card(children, border_color=None, extra=None):
    style = {"background": WHITE, "borderRadius": "8px", "padding": "20px 24px",
             "marginBottom": "16px", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
             "border": f"1px solid {BORDER}"}
    if border_color:
        style["borderLeft"] = f"3px solid {border_color}"
    if extra:
        style.update(extra)
    return html.Div(children, style=style)

def section_header(title, subtitle=None):
    els = [html.Div(title, style={"fontSize": "15px", "fontWeight": "600",
                                   "color": TEXT_DARK, "letterSpacing": "-0.2px",
                                   "marginBottom": "4px"})]
    if subtitle:
        els.append(html.Div(subtitle, style={"fontSize": "12px", "color": TEXT_LIGHT}))
    return html.Div(els, style={"marginBottom": "24px"})

def divider():
    return html.Hr(style={"border": "none", "borderTop": f"1px solid {BORDER}",
                           "margin": "28px 0"})

def lbl():
    return {"fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.5px",
            "color": TEXT_MID, "textTransform": "uppercase",
            "marginBottom": "6px", "display": "block"}

def table_style():
    return dict(
        style_table={"overflowX": "auto", "borderRadius": "8px",
                     "border": f"1px solid {BORDER}"},
        style_header={"backgroundColor": TEXT_DARK, "color": WHITE,
                      "fontWeight": "600", "fontSize": "12px",
                      "letterSpacing": "0.3px", "padding": "10px 14px"},
        style_cell={"padding": "9px 14px", "fontSize": "13px", "color": TEXT_DARK,
                    "border": f"1px solid {BORDER}",
                    "fontFamily": "Inter, Segoe UI, Arial"},
        style_data_conditional=[{"if": {"row_index": "odd"},
                                  "backgroundColor": GRAY_SOFT}],
    )

# ── Data helpers ───────────────────────────────────────────────────────────────
def load_frequency_table():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT smp.sample_id AS sample, sub.subject_id, sub.project,
               sub.condition, sub.treatment, sub.response, sub.sex,
               smp.sample_type, smp.time_from_treatment_start,
               cc.population, cc.count
        FROM samples smp
        JOIN subjects sub ON smp.subject_id = sub.subject_id
        JOIN cell_counts cc ON cc.sample_id = smp.sample_id
    """, conn)
    conn.close()
    totals = df.groupby("sample")["count"].transform("sum")
    df["total_count"] = totals
    df["percentage"]  = (df["count"] / df["total_count"] * 100).round(4)
    return df

def compute_stats(df):
    mask = ((df["condition"] == "melanoma") & (df["treatment"] == "miraclib") &
            (df["sample_type"] == "PBMC"))
    filtered = df[mask]
    results = []
    for pop in POPULATIONS:
        resp    = filtered[(filtered["population"] == pop) & (filtered["response"] == "yes")]["percentage"].values
        nonresp = filtered[(filtered["population"] == pop) & (filtered["response"] == "no")]["percentage"].values
        if len(resp) >= 2 and len(nonresp) >= 2:
            stat, pval = stats.mannwhitneyu(resp, nonresp, alternative="two-sided")
        else:
            stat, pval = np.nan, np.nan
        results.append({
            "population":              POP_LABELS[pop],
            "n_responders":            len(resp),
            "n_non_responders":        len(nonresp),
            "median_responders_%":     round(float(np.median(resp)),    2) if len(resp)    else np.nan,
            "median_non_responders_%": round(float(np.median(nonresp)), 2) if len(nonresp) else np.nan,
            "mannwhitney_U":           round(stat, 2) if not np.isnan(stat) else "N/A",
            "p_value":                 round(pval, 4) if not np.isnan(pval) else "N/A",
            "significant (p<0.05)":    "Yes" if (not np.isnan(pval) and pval < 0.05) else "No",
        })
    return pd.DataFrame(results)

def load_part4():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT sub.subject_id, sub.project, sub.sex, sub.response,
               smp.sample_id, cc.population, cc.count
        FROM subjects sub
        JOIN samples smp ON smp.subject_id = sub.subject_id
        JOIN cell_counts cc ON cc.sample_id = smp.sample_id
        WHERE sub.condition = 'melanoma'
          AND smp.sample_type = 'PBMC'
          AND smp.time_from_treatment_start = 0
          AND sub.treatment = 'miraclib'
    """, conn)
    conn.close()
    return df

# ── Pre-load ───────────────────────────────────────────────────────────────────
df_full  = load_frequency_table()
df_stats = compute_stats(df_full)
df_p4    = load_part4()

ALL_SAMPLES = sorted(df_full["sample"].unique().tolist())
ALL_POPS    = [{"label": POP_LABELS[p], "value": p} for p in sorted(POPULATIONS)]

# ── Chart helper ───────────────────────────────────────────────────────────────
def make_freq_chart(df):
    """Box plot per population — full distribution with quantiles, IQR, mean±SD."""
    n = df["sample"].nunique()
    if n == 0:
        return go.Figure()
    fig = go.Figure()
    for i, pop in enumerate(sorted(POPULATIONS)):
        vals = df[df["population"] == pop]["percentage"].dropna()
        if len(vals) == 0:
            continue
        # Downsample only for rendering speed; stats computed on full set
        y = vals.sample(min(1500, len(vals)), random_state=42).tolist()
        fig.add_trace(go.Box(
            y=y,
            name=POP_LABELS[pop],
            marker=dict(color=PALETTE[i], size=3, opacity=0.6),
            line=dict(color=PALETTE[i], width=1.5),
            fillcolor=PALETTE_LIGHT[i],
            boxmean="sd",       # shows mean ± SD diamond
            boxpoints="outliers",
        ))
    fig.update_layout(
        **CHART_LAYOUT,
        xaxis_title="Cell Population",
        yaxis_title="Relative Frequency (%)",
        height=440,
        showlegend=False,
        annotations=[dict(
            text=(f"{n:,} sample(s) aggregated  \u00b7  "
                  "box: Q1\u2013Q3 (IQR)  \u00b7  line: median  \u00b7  "
                  "whiskers: 1.5\u00d7IQR  \u00b7  diamond: mean \u00b1 SD  \u00b7  "
                  "points: outliers"),
            xref="paper", yref="paper", x=0, y=-0.15,
            showarrow=False,
            font=dict(size=11, color=TEXT_LIGHT),
            align="left",
        )],
    )
    return fig

# ── App ────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Teiko — Trial Dashboard")

TAB_BASE = {"padding": "12px 24px", "fontWeight": "500", "fontSize": "13px",
            "color": TEXT_MID, "borderBottom": "2px solid transparent", "background": WHITE}
TAB_ACTIVE = {**TAB_BASE, "color": ACCENT, "fontWeight": "600",
              "borderBottom": f"2px solid {ACCENT}"}

app.layout = html.Div([

    html.Div([
        html.Div("TEIKO", style={"fontSize": "10px", "fontWeight": "700",
                                  "letterSpacing": "2px", "color": TEXT_LIGHT,
                                  "marginBottom": "5px"}),
        html.Div("Technical Question  \u2014  Immune Cell Trial Analysis",
                 style={"fontSize": "17px", "fontWeight": "600", "color": TEXT_DARK,
                        "letterSpacing": "-0.3px"}),
    ], style={"background": WHITE, "padding": "20px 36px",
              "borderBottom": f"1px solid {BORDER}"}),

    html.Div([
        dcc.Tabs(id="tabs", value="tab-2", children=[
            dcc.Tab(label="Frequency Overview",   value="tab-2",
                    style=TAB_BASE, selected_style=TAB_ACTIVE),
            dcc.Tab(label="Statistical Analysis", value="tab-3",
                    style=TAB_BASE, selected_style=TAB_ACTIVE),
            dcc.Tab(label="Subset Analysis",      value="tab-4",
                    style=TAB_BASE, selected_style=TAB_ACTIVE),
        ], style={"border": "none"}),
    ], style={"background": WHITE, "paddingLeft": "28px",
              "borderBottom": f"1px solid {BORDER}"}),

    # ── Tab-2 panel (always in DOM) ────────────────────────────────────────────
    html.Div(id="tab2-panel", children=[

        section_header(
            "Cell Population Relative Frequencies",
            "Filter samples and populations below. "
            "The chart aggregates all selected samples regardless of population filter."
        ),

        # Filter bar
        html.Div([
            html.Div([
                html.Label("Sample", style=lbl()),
                dcc.Dropdown(
                    id="p2-sample",
                    options=[{"label": s, "value": s} for s in ALL_SAMPLES],
                    multi=True, placeholder="Type or select samples…",
                    style={"fontSize": "13px"},
                ),
            ], style={"flex": "3", "minWidth": "260px"}),

            html.Div([
                html.Label("Population", style=lbl()),
                dcc.Dropdown(
                    id="p2-pop",
                    options=ALL_POPS,
                    multi=True, placeholder="All populations",
                    style={"fontSize": "13px"},
                ),
            ], style={"flex": "2", "minWidth": "180px"}),

            html.Div([
                html.Label(id="p2-pct-label",
                           children="Percentage  0% \u2013 100%", style=lbl()),
                dcc.RangeSlider(
                    id="p2-pct", min=0, max=100, step=0.5, value=[0, 100],
                    marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"flex": "3", "minWidth": "240px", "paddingTop": "2px"}),

            html.Div([
                html.Label("\u00a0", style=lbl()),
                html.Button("Reset", id="p2-reset", n_clicks=0, style={
                    "padding": "7px 16px", "cursor": "pointer",
                    "borderRadius": "5px", "border": f"1px solid {BORDER}",
                    "background": WHITE, "fontSize": "12px",
                    "fontWeight": "500", "color": TEXT_MID,
                }),
            ], style={"flexShrink": "0"}),

        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                  "alignItems": "flex-start", "background": WHITE,
                  "border": f"1px solid {BORDER}", "borderRadius": "8px",
                  "padding": "16px 20px", "marginBottom": "16px"}),

        # Stats strip
        html.Div(id="p2-stats-strip", style={"marginBottom": "16px"}),

        # Table
        html.Div("Summary Table", style={"fontSize": "13px", "fontWeight": "600",
                                          "color": TEXT_DARK, "marginBottom": "6px"}),
        html.Div(id="p2-row-label", style={"fontSize": "12px", "color": TEXT_LIGHT,
                                            "marginBottom": "8px"}),
        dash_table.DataTable(
            id="p2-table",
            columns=[{"name": c, "id": c} for c in
                     ["sample", "total_count", "population", "count", "percentage"]],
            data=[],
            page_size=25,
            sort_action="native",
            filter_action="none",
            **table_style(),
        ),

        divider(),

        # Chart
        html.Div("Frequency Distribution  \u2014  All Selected Samples Aggregated",
                 style={"fontSize": "13px", "fontWeight": "600",
                        "color": TEXT_DARK, "marginBottom": "4px"}),
        dcc.Graph(id="p2-chart"),

    ], style={"padding": "32px 36px", "background": BG, "display": "block"}),

    # ── Tab-3 / Tab-4 dynamic panel ────────────────────────────────────────────
    html.Div(id="tab-content", style={"padding": "32px 36px", "background": BG,
                                       "minHeight": "90vh", "display": "none"}),

], style={"fontFamily": "'Inter', 'Segoe UI', Arial, sans-serif",
          "background": BG, "color": TEXT_DARK})


# ── Show/hide tab panels ───────────────────────────────────────────────────────
@app.callback(
    Output("tab2-panel",  "style"),
    Output("tab-content", "style"),
    Input("tabs", "value"),
)
def toggle_panels(tab):
    show2  = {"padding": "32px 36px", "background": BG, "display": "block"}
    show34 = {"padding": "32px 36px", "background": BG,
               "minHeight": "90vh", "display": "block"}
    hide   = {"display": "none"}
    return (show2, hide) if tab == "tab-2" else (hide, show34)


# ── Tab-3 / Tab-4 renderer ────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    if tab == "tab-3":
        mask = ((df_full["condition"] == "melanoma") &
                (df_full["treatment"] == "miraclib") &
                (df_full["sample_type"] == "PBMC"))
        df3 = df_full[mask].copy()
        df3["Response"] = df3["response"].map({"yes": "Responder", "no": "Non-Responder"})
        df3["pop_label"] = df3["population"].map(POP_LABELS)

        box_fig = go.Figure()
        for resp_label, color in [("Responder", ACCENT), ("Non-Responder", "#E05252")]:
            subset = df3[df3["Response"] == resp_label]
            box_fig.add_trace(go.Box(
                x=subset["pop_label"], y=subset["percentage"],
                name=resp_label, marker_color=color,
                boxmean=True, line=dict(width=1.5),
            ))
        box_fig.update_layout(**CHART_LAYOUT, boxmode="group",
                               legend_title_text="Response",
                               xaxis_title="Cell Population",
                               yaxis_title="Relative Frequency (%)", height=420)

        ts = table_style()
        ts["style_data_conditional"] = [
            {"if": {"filter_query": '{significant (p<0.05)} = "Yes"',
                    "column_id": "significant (p<0.05)"},
             "color": GREEN, "fontWeight": "600"},
            {"if": {"row_index": "odd"}, "backgroundColor": GRAY_SOFT},
        ]

        sig_rows    = df_stats[df_stats["significant (p<0.05)"] == "Yes"]
        nonsig_rows = df_stats[df_stats["significant (p<0.05)"] != "Yes"]

        def finding_bullets(rows):
            items = []
            for _, r in rows.iterrows():
                direction = ("higher in responders"
                             if r["median_responders_%"] > r["median_non_responders_%"]
                             else "higher in non-responders")
                items.append(html.Li([
                    html.Strong(r["population"]),
                    f": median {r['median_responders_%']}% vs"
                    f" {r['median_non_responders_%']}%  \u2014  {direction}"
                    f"  (U={r['mannwhitney_U']}, p={r['p_value']})"
                ], style={"marginBottom": "7px", "fontSize": "13px",
                          "lineHeight": "1.5", "color": TEXT_DARK}))
            return items

        return html.Div([
            section_header("Responder vs Non-Responder Analysis",
                           "Melanoma  \u00b7  miraclib  \u00b7  PBMC samples only"),
            dcc.Graph(figure=box_fig),
            divider(),
            html.Div("Statistical Summary", style={"fontSize": "13px", "fontWeight": "600",
                                                    "color": TEXT_DARK, "marginBottom": "12px"}),
            dash_table.DataTable(data=df_stats.to_dict("records"),
                                 columns=[{"name": c, "id": c} for c in df_stats.columns],
                                 **ts),
            divider(),
            html.Div("Key Findings", style={"fontSize": "13px", "fontWeight": "600",
                                             "color": TEXT_DARK, "marginBottom": "12px"}),
            html.Div([
                card([
                    html.Div("Significant  \u2014  p < 0.05",
                             style={"fontSize": "11px", "fontWeight": "700",
                                    "letterSpacing": "0.5px", "color": GREEN,
                                    "textTransform": "uppercase", "marginBottom": "12px"}),
                    html.Ul(finding_bullets(sig_rows) or [html.Li("None.")],
                            style={"margin": "0", "paddingLeft": "16px"}),
                ], border_color=GREEN, extra={"flex": "1", "minWidth": "280px",
                                              "background": GREEN_SOFT}),
                card([
                    html.Div("Not significant  \u2014  p \u2265 0.05",
                             style={"fontSize": "11px", "fontWeight": "700",
                                    "letterSpacing": "0.5px", "color": TEXT_LIGHT,
                                    "textTransform": "uppercase", "marginBottom": "12px"}),
                    html.Ul(finding_bullets(nonsig_rows) or [html.Li("All significant.")],
                            style={"margin": "0", "paddingLeft": "16px"}),
                ], extra={"flex": "1", "minWidth": "280px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            divider(),
            html.Div("Statistical Note", style={"fontSize": "13px", "fontWeight": "600",
                                                 "color": TEXT_DARK, "marginBottom": "12px"}),
            card(html.P([
                "The ", html.Strong("Mann-Whitney U test"),
                " (non-parametric, two-sided, \u03b1\u202f=\u202f0.05) was used. "
                "Cell population percentages are bounded and often skewed, making a "
                "rank-based test more appropriate than a t-test. With 5 populations "
                "tested simultaneously, a Bonferroni threshold of \u03b1/5\u202f=\u202f0.01 "
                "may be applied for stricter family-wise error control."
            ], style={"margin": "0", "fontSize": "13px", "color": TEXT_MID,
                      "lineHeight": "1.7"}),
            border_color=ACCENT, extra={"background": ACCENT_SOFT}),
        ])

    elif tab == "tab-4":
        spp = (df_p4[["sample_id", "project"]].drop_duplicates()
               .groupby("project")["sample_id"].count().reset_index()
               .rename(columns={"sample_id": "Samples"})
               .sort_values("project"))
        bar_colors = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#76B7B2"]
        proj_fig = go.Figure(go.Bar(
            x=spp["project"],
            y=spp["Samples"],
            text=spp["Samples"],
            textposition="outside",
            marker_color=[bar_colors[i % len(bar_colors)] for i in range(len(spp))],
            marker_line_width=0,
        ))
        proj_fig.update_layout(
            **CHART_LAYOUT,
            title_text="Samples per Project",
            xaxis_title="Project", yaxis_title="Samples",
            height=340,
        )
        proj_fig.update_yaxes(range=[0, spp["Samples"].max() * 1.22])

        subj_resp = (df_p4[["subject_id", "response"]].drop_duplicates()
                     .groupby("response")["subject_id"].count().reset_index()
                     .rename(columns={"subject_id": "Subjects", "response": "Response"}))
        subj_resp["Response"] = subj_resp["Response"].map(
            {"yes": "Responder", "no": "Non-Responder"})
        resp_fig = go.Figure(go.Pie(
            labels=subj_resp["Response"],
            values=subj_resp["Subjects"],
            hole=0.5,
            marker=dict(colors=[ACCENT, "#E05252"], line=dict(color=WHITE, width=2)),
            texttemplate="%{label}<br><b>%{value}</b> (%{percent})",
            textposition="outside",
            hovertemplate="%{label}: %{value} subjects (%{percent})<extra></extra>",
        ))
        resp_fig.update_layout(**CHART_LAYOUT, title_text="Response Distribution",
                                height=340, showlegend=True)

        subj_sex = (df_p4[["subject_id", "sex"]].drop_duplicates()
                    .groupby("sex")["subject_id"].count().reset_index()
                    .rename(columns={"subject_id": "Subjects", "sex": "Sex"}))
        subj_sex["Sex"] = subj_sex["Sex"].map({"M": "Male", "F": "Female"})
        sex_fig = go.Figure(go.Pie(
            labels=subj_sex["Sex"],
            values=subj_sex["Subjects"],
            hole=0.5,
            marker=dict(colors=["#4E79A7", "#B07AA1"], line=dict(color=WHITE, width=2)),
            texttemplate="%{label}<br><b>%{value}</b> (%{percent})",
            textposition="outside",
            hovertemplate="%{label}: %{value} subjects (%{percent})<extra></extra>",
        ))
        sex_fig.update_layout(**CHART_LAYOUT, title_text="Sex Distribution",
                               height=340, showlegend=True)

        bcell = df_p4[(df_p4["sex"] == "M") & (df_p4["response"] == "yes") &
                      (df_p4["population"] == "b_cell")]
        avg_b   = f"{bcell['count'].mean():.2f}" if len(bcell) > 0 else "N/A"
        n_bcell = len(bcell)

        def kpi(value, label, accent=ACCENT):
            return html.Div([
                html.Div(str(value), style={"fontSize": "22px", "fontWeight": "700",
                                             "color": TEXT_DARK, "letterSpacing": "-0.5px"}),
                html.Div(label, style={"fontSize": "11px", "color": TEXT_LIGHT,
                                        "marginTop": "4px", "lineHeight": "1.4"}),
            ], style={"background": WHITE, "borderRadius": "8px", "padding": "16px 20px",
                      "border": f"1px solid {BORDER}", "borderTop": f"3px solid {accent}",
                      "minWidth": "160px", "flex": "1"})

        return html.Div([
            section_header("Baseline Subset Analysis",
                           "Condition: melanoma  \u00b7  Sample type: PBMC  \u00b7"
                           "  Time from treatment start: 0  \u00b7  Treatment: miraclib"),
            html.Div([
                kpi(f"{df_p4['sample_id'].nunique():,}", "Qualifying samples"),
                kpi(f"{df_p4['subject_id'].nunique():,}", "Unique subjects"),
                kpi(avg_b, "Avg B cells  \u2014  melanoma males, responders, time=0", GREEN),
                kpi(f"{n_bcell:,}", "Samples in B cell average"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                      "marginBottom": "32px"}),
            html.Div([
                html.Div(dcc.Graph(figure=proj_fig), style={"flex": "1.2", "minWidth": "300px"}),
                html.Div(dcc.Graph(figure=resp_fig), style={"flex": "1",   "minWidth": "260px"}),
                html.Div(dcc.Graph(figure=sex_fig),  style={"flex": "1",   "minWidth": "260px"}),
            ], style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}),
        ])

    return html.Div()


# ── Part 2 callbacks ───────────────────────────────────────────────────────────

@app.callback(
    Output("p2-pct-label", "children"),
    Input("p2-pct", "value"),
)
def update_pct_label(rng):
    lo, hi = rng if rng else [0, 100]
    return f"Percentage  {lo}% \u2013 {hi}%"


@app.callback(
    Output("p2-sample", "value"),
    Output("p2-pop",    "value"),
    Output("p2-pct",    "value"),
    Input("p2-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_filters(_):
    return None, None, [0, 100]


@app.callback(
    Output("p2-chart",       "figure"),
    Output("p2-stats-strip", "children"),
    Input("p2-sample", "value"),
)
def update_p2_chart(sel_samples):
    df = df_full.copy()
    if sel_samples:
        df = df[df["sample"].isin(sel_samples)]

    fig = make_freq_chart(df)

    n_samples = df["sample"].nunique()
    n_total   = df_full["sample"].nunique()

    stats_strip = html.Div([
        html.Div([
            html.Span(f"{n_samples:,}", style={"fontSize": "18px", "fontWeight": "700",
                                                "color": TEXT_DARK}),
            html.Span(f"  of {n_total:,} samples selected",
                      style={"fontSize": "12px", "color": TEXT_LIGHT}),
        ], style={"marginRight": "28px"}),
    ], style={"display": "flex", "alignItems": "center", "background": WHITE,
               "border": f"1px solid {BORDER}", "borderRadius": "6px",
               "padding": "10px 18px", "width": "fit-content"})

    return fig, stats_strip


@app.callback(
    Output("p2-table",     "data"),
    Output("p2-row-label", "children"),
    Input("p2-sample", "value"),
    Input("p2-pop",    "value"),
    Input("p2-pct",    "value"),
)
def update_p2_table(sel_samples, sel_pops, pct_range):
    df = df_full.copy()

    if sel_samples:
        df = df[df["sample"].isin(sel_samples)]

    lo, hi = pct_range if pct_range else [0, 100]
    df = df[(df["percentage"] >= lo) & (df["percentage"] <= hi)]

    if sel_pops:
        df = df[df["population"].isin(sel_pops)]

    table_df = (
        df[["sample", "total_count", "population", "count", "percentage"]]
        .assign(population=df["population"].map(POP_LABELS))
        .sort_values(["sample", "population"])
        .reset_index(drop=True)
    )

    n_samples = df["sample"].nunique()
    row_label = f"Showing {len(table_df):,} rows  \u00b7  {n_samples:,} sample(s)"
    return table_df.to_dict("records"), row_label


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=8050)
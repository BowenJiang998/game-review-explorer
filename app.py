"""
Video Game Review Explorer - Streamlit website.

Run:   streamlit run app.py                         (uses the data/ folder)
       streamlit run app.py -- --data data_demo     (uses another folder)
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

POS_COLOR, NEG_COLOR, NEUTRAL = "#2E86AB", "#E4572E", "#8D99AE"

st.set_page_config(page_title="Video Game Review Explorer", page_icon="🎮", layout="wide")


def data_dir():
    args = sys.argv[1:]
    if "--data" in args and args.index("--data") + 1 < len(args):
        return Path(args[args.index("--data") + 1])
    return Path(__file__).parent / "data"


@st.cache_data
def load(folder: str):
    f = Path(folder)
    read = lambda name, **kw: pd.read_csv(f / name, **kw) if (f / name).exists() else pd.DataFrame()
    d = {
        "summary": read("summary.csv", parse_dates=["first_month", "last_month"]),
        "monthly": read("monthly.csv", parse_dates=["month"]),
        "topics": read("topics.csv"),
        "topics_monthly": read("topics_monthly.csv", parse_dates=["month"]),
        "keywords": read("keywords.csv"),
        "examples": read("examples.csv", parse_dates=["date"]),
        "validation": read("validation.csv"),
    }
    d["meta"] = json.loads((f / "meta.json").read_text()) if (f / "meta.json").exists() else {}
    return d


DATA = data_dir()
if not (DATA / "summary.csv").exists():
    st.title("🎮 Video Game Review Explorer")
    st.error(f"No processed data found in `{DATA}`.")
    st.markdown(
        "Run the data step first:\n\n"
        "```bash\npython prep_data.py --input steam_reviews.csv\n```\n"
        "Or try it with fake demo data:\n\n"
        "```bash\npython make_demo_data.py\npython prep_data.py --input demo_reviews.csv --out data_demo\n"
        "streamlit run app.py -- --data data_demo\n```"
    )
    st.stop()

D = load(str(DATA))
summary, monthly, topics, kw = D["summary"], D["monthly"], D["topics"], D["keywords"]
games = summary["game"].tolist()

# ---------------- sidebar ----------------
st.sidebar.title("🎮 Review Explorer")
game = st.sidebar.selectbox("Choose a game", games, help="Sorted by number of reviews")
g_month = monthly[monthly["game"] == game]
min_m, max_m = g_month["month"].min().to_pydatetime(), g_month["month"].max().to_pydatetime()
if min_m < max_m:
    date_range = st.sidebar.slider("Date range", min_value=min_m, max_value=max_m, value=(min_m, max_m), format="MMM YYYY")
else:
    date_range = (min_m, max_m)
smooth = st.sidebar.checkbox("Smooth trend lines (3-month average)", value=True)
min_reviews = st.sidebar.number_input("Hide months with fewer reviews than", 0, 10_000, 30, step=10)
st.sidebar.caption(
    f"Data: {D['meta'].get('rows_kept', summary['reviews'].sum()):,} cleaned reviews "
    f"across {len(games)} games."
)


def in_range(df, col="month"):
    return df[(df[col] >= pd.Timestamp(date_range[0])) & (df[col] <= pd.Timestamp(date_range[1]))]


def trend(df):
    df = df[df["reviews"] >= min_reviews].sort_values("month").copy()
    if smooth and len(df) >= 3:
        r = df[["reviews", "positive"]].rolling(3, min_periods=1).sum()
        df["pct_positive"] = r["positive"] / r["reviews"] * 100
    return df


g_month_r = in_range(g_month)
n = int(g_month_r["reviews"].sum())
pos = int(g_month_r["positive"].sum())
pct = pos / n * 100 if n else 0.0

# ---------------- header ----------------
st.title(game)
st.caption(f"What players like and dislike, based on {n:,} reviews between "
           f"{pd.Timestamp(date_range[0]):%b %Y} and {pd.Timestamp(date_range[1]):%b %Y}.")

# recent vs. overall
last12 = g_month_r[g_month_r["month"] > g_month_r["month"].max() - pd.DateOffset(months=12)]
recent_pct = last12["positive"].sum() / max(last12["reviews"].sum(), 1) * 100

g_topics = topics[topics["game"] == game]
top_complaint = (g_topics.sort_values("pct_of_negative_reviews", ascending=False)["topic"].iloc[0]
                 if not g_topics.empty else "–")
top_praise = (g_topics.sort_values("pct_of_positive_reviews", ascending=False)["topic"].iloc[0]
              if not g_topics.empty else "–")


def rating_label(p):
    for cut, label in [(95, "Overwhelmingly Positive"), (80, "Very Positive"), (70, "Mostly Positive"),
                       (40, "Mixed"), (20, "Mostly Negative"), (0, "Very Negative")]:
        if p >= cut:
            return label
    return "Very Negative"


c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{n:,}")
c2.metric("Positive", f"{pct:.1f}%", rating_label(pct), delta_color="off")
c3.metric("Last 12 months", f"{recent_pct:.1f}%", f"{recent_pct - pct:+.1f} pts vs. whole period")
c4.metric("Most-mentioned complaint", top_complaint, f"Most praised: {top_praise}", delta_color="off")

tab_overview, tab_feedback, tab_time, tab_compare, tab_reviews, tab_about = st.tabs(
    ["📊 Overview", "💬 What players say", "📈 Over time", "⚖️ Compare games", "📝 Example reviews", "ℹ️ About the data"])

# ---------------- overview ----------------
with tab_overview:
    left, right = st.columns([1, 2])
    with left:
        fig = go.Figure(go.Pie(labels=["Positive", "Negative"], values=[pos, n - pos], hole=0.6,
                               marker_colors=[POS_COLOR, NEG_COLOR], sort=False))
        fig.update_layout(title="Positive vs. negative reviews", showlegend=True, height=380,
                          annotations=[dict(text=f"{pct:.0f}%<br>positive", showarrow=False, font_size=18)])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        vol = g_month_r.assign(negative=lambda d: d["reviews"] - d["positive"])
        fig = go.Figure([
            go.Bar(x=vol["month"], y=vol["positive"], name="Positive", marker_color=POS_COLOR),
            go.Bar(x=vol["month"], y=vol["negative"], name="Negative", marker_color=NEG_COLOR),
        ])
        fig.update_layout(barmode="stack", title="Reviews per month", height=380,
                          yaxis_title="Reviews", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    row = summary[summary["game"] == game].iloc[0]
    if "median_playtime_pos_h" in summary and pd.notna(row.get("median_playtime_pos_h")):
        st.info(f"⏱️ Reviewers who recommend the game played a median of **{row['median_playtime_pos_h']:.0f} h**; "
                f"those who don't played **{row['median_playtime_neg_h']:.0f} h**.")

    rank = summary.sort_values("pct_positive", ascending=False).reset_index(drop=True)
    place = rank.index[rank["game"] == game][0] + 1
    st.write(f"Ranked **#{place} of {len(rank)}** games in this dataset by % positive (all-time).")

# ---------------- feedback ----------------
with tab_feedback:
    st.subheader("Which topics come up in positive vs. negative reviews")
    st.caption("Share of positive (or negative) reviews that mention each topic. Topics are matched with keyword lists in topics.py.")
    if not g_topics.empty:
        t = g_topics.melt(id_vars="topic", value_vars=["pct_of_positive_reviews", "pct_of_negative_reviews"],
                          var_name="side", value_name="pct")
        t["side"] = t["side"].map({"pct_of_positive_reviews": "Positive reviews", "pct_of_negative_reviews": "Negative reviews"})
        order = g_topics.sort_values("pct_of_negative_reviews")["topic"].tolist()
        fig = px.bar(t, y="topic", x="pct", color="side", barmode="group", orientation="h",
                     category_orders={"topic": order[::-1]},
                     color_discrete_map={"Positive reviews": POS_COLOR, "Negative reviews": NEG_COLOR},
                     labels={"pct": "% of reviews mentioning", "topic": "", "side": ""})
        fig.update_layout(height=450, legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Words and phrases that set positive and negative reviews apart")
    g_kw = kw[kw["game"] == game] if not kw.empty else kw
    if g_kw.empty:
        st.write("Not enough reviews for keyword analysis.")
    else:
        k1, k2 = st.columns(2)
        for col, side, color, title, field in [
            (k1, "praise", POS_COLOR, "👍 Typical of positive reviews", "pct_positive_reviews"),
            (k2, "complaint", NEG_COLOR, "👎 Typical of negative reviews", "pct_negative_reviews"),
        ]:
            d = g_kw[g_kw["side"] == side].head(15).iloc[::-1]
            fig = px.bar(d, x=field, y="term", orientation="h", color_discrete_sequence=[color],
                         labels={field: "% of reviews on that side using it", "term": ""},
                         hover_data={"pct_positive_reviews": True, "pct_negative_reviews": True})
            fig.update_layout(title=title, height=480)
            col.plotly_chart(fig, use_container_width=True)
        st.caption(f"Based on a random sample of {int(g_kw['sample_size'].iloc[0]):,} reviews. "
                   "Ranked by how much more often a term appears on one side (log-odds), weighted by frequency.")

# ---------------- over time ----------------
with tab_time:
    st.subheader("How opinion changed over time")
    tr = trend(g_month_r)
    fig = go.Figure()
    fig.add_bar(x=tr["month"], y=tr["reviews"], name="Reviews", marker_color=NEUTRAL, opacity=0.35, yaxis="y2")
    fig.add_scatter(x=tr["month"], y=tr["pct_positive"], name="% positive", mode="lines+markers",
                    line=dict(color=POS_COLOR, width=3))
    fig.add_hline(y=pct, line_dash="dot", line_color="gray", annotation_text=f"period average {pct:.0f}%")
    fig.update_layout(height=430, yaxis=dict(title="% positive", range=[0, 100]),
                      yaxis2=dict(title="Reviews", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    if len(tr) >= 2:
        best, worst = tr.loc[tr["pct_positive"].idxmax()], tr.loc[tr["pct_positive"].idxmin()]
        st.write(f"Best month: **{best['month']:%b %Y}** ({best['pct_positive']:.0f}% positive). "
                 f"Worst month: **{worst['month']:%b %Y}** ({worst['pct_positive']:.0f}% positive).")

    st.subheader("Complaints by topic over time")
    st.caption("Share of each month's negative reviews that mention the topic. Spikes often line up with patches or updates.")
    tm = D["topics_monthly"]
    if not tm.empty:
        tm_g = in_range(tm[tm["game"] == game]).merge(g_month[["month", "reviews", "positive"]], on="month")
        tm_g["negative"] = tm_g["reviews"] - tm_g["positive"]
        tm_g = tm_g[tm_g["negative"] >= max(min_reviews // 3, 5)]
        default = [t for t in [top_complaint, "Bugs & Performance"] if t in tm_g["topic"].unique()]
        chosen = st.multiselect("Topics", sorted(tm_g["topic"].unique()), default=list(dict.fromkeys(default)))
        tm_g = tm_g[tm_g["topic"].isin(chosen)].sort_values("month")
        tm_g["pct"] = tm_g["mentions_negative"] / tm_g["negative"] * 100
        if smooth:
            tm_g["pct"] = tm_g.groupby("topic")["pct"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        fig = px.line(tm_g, x="month", y="pct", color="topic", markers=True,
                      labels={"pct": "% of negative reviews mentioning", "month": ""})
        fig.update_layout(height=400, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

# ---------------- compare ----------------
with tab_compare:
    st.subheader("Compare games side by side")
    default_cmp = [game] + [g for g in games if g != game][:2]
    picks = st.multiselect("Games to compare", games, default=default_cmp, max_selections=5)
    if picks:
        s = summary[summary["game"].isin(picks)].sort_values("pct_positive")
        fig = px.bar(s, x="pct_positive", y="game", orientation="h", text="pct_positive",
                     color="pct_positive", color_continuous_scale=[NEG_COLOR, "#F6AE2D", POS_COLOR], range_color=[30, 100],
                     labels={"pct_positive": "% positive (all-time)", "game": ""})
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(height=120 + 60 * len(picks), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        m = pd.concat([trend(monthly[monthly["game"] == p]) for p in picks])
        fig = px.line(m, x="month", y="pct_positive", color="game", labels={"pct_positive": "% positive", "month": ""})
        fig.update_layout(title="% positive over time", height=400, yaxis_range=[0, 100], legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        heat = topics[topics["game"].isin(picks)].pivot(index="game", columns="topic", values="pct_of_negative_reviews")
        fig = px.imshow(heat, text_auto=".0f", color_continuous_scale="Reds", aspect="auto",
                        labels=dict(color="% of negative reviews"))
        fig.update_layout(title="What negative reviews complain about (% mentioning each topic)", height=150 + 50 * len(picks))
        st.plotly_chart(fig, use_container_width=True)

# ---------------- example reviews ----------------
with tab_reviews:
    st.subheader("Most helpful example reviews")
    st.caption("Highest 'helpful' votes from the random sample. Reviewer identities are not stored.")
    ex = D["examples"]
    ex = ex[ex["game"] == game] if not ex.empty else ex
    e1, e2 = st.columns(2)
    for col, flag, title in [(e1, True, "👍 Recommended"), (e2, False, "👎 Not recommended")]:
        col.markdown(f"#### {title}")
        sub = ex[ex["positive"] == flag] if not ex.empty else ex
        if sub.empty:
            col.write("No examples.")
        for _, r in sub.iterrows():
            col.markdown(f"> {r['text']}\n\n<small>{r['date']:%b %Y} · {int(r['helpful'])} found this helpful</small>",
                         unsafe_allow_html=True)

# ---------------- about ----------------
with tab_about:
    meta = D["meta"]
    st.subheader("About the data")
    if meta:
        rows = [
            ("Source file", meta.get("source_file")),
            ("Rows read", f"{meta.get('rows_read', 0):,}"),
            ("Other games / languages filtered out", f"{meta.get('rows_other_games_or_languages', 0):,}"),
            ("Empty reviews dropped", f"{meta.get('dropped_empty', 0):,}"),
            (f"Reviews under {meta.get('min_words')} words dropped", f"{meta.get('dropped_short', 0):,}"),
            ("Duplicate reviews dropped", f"{meta.get('dropped_duplicate', 0):,}"),
            ("Reviews analysed", f"{meta.get('rows_kept', 0):,}"),
            ("Games", meta.get("games")),
            ("Date range", " to ".join(meta.get("date_range", []))),
            ("English only", meta.get("english_only")),
            ("Processed (UTC)", meta.get("generated_utc")),
        ]
        st.table(pd.DataFrame(rows, columns=["", "Value"]).set_index(""))
    st.markdown(
        "**Method.** *Positive / negative* comes from each reviewer's own recommend / not-recommend choice on Steam. "
        "*Topics* are found by keyword matching (see `topics.py`). *Keywords* compare how often a word appears in "
        "negative vs. positive reviews (smoothed log-odds). Trends are grouped by the month the review was written."
    )
    val = D["validation"]
    if not val.empty:
        st.subheader("Check: does a sentiment model agree with the thumbs up / down?")
        st.caption("VADER sentiment scores on a sample of reviews, compared with the recommended label. "
                   "Disagreements are often sarcasm, jokes or mixed reviews.")
        st.dataframe(val.rename(columns={"agreement_pct": "agreement %", "steam_pct_positive": "Steam % positive",
                                         "vader_pct_positive": "VADER % positive"}),
                     hide_index=True, use_container_width=True)
        st.write(f"Average agreement across games: **{val['agreement_pct'].mean():.1f}%**")
    st.subheader("All games")
    st.dataframe(summary[["game", "reviews", "pct_positive", "first_month", "last_month"]],
                 hide_index=True, use_container_width=True)

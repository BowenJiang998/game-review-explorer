"""
Step 1 of the project: turn the huge raw Steam reviews CSV into small summary
files that the website loads instantly.

Usage (from the project folder):
    python prep_data.py --input steam_reviews.csv
    python prep_data.py --input steam_reviews.csv --top 30
    python prep_data.py --input steam_reviews.csv --games "Terraria" "Rust" "Dota 2"

The raw file is read in chunks, so it never has to fit in memory at once.
Outputs go to the data/ folder (all small CSV/JSON files):
    summary.csv        one row per game: review count, % positive, dates
    monthly.csv        reviews and % positive per game per month
    topics.csv         how often positive vs negative reviews mention each topic
    topics_monthly.csv topic mentions per game per month
    keywords.csv       words/phrases most typical of positive vs negative reviews
    examples.csv       a few highly-voted example reviews per game
    validation.csv     VADER sentiment vs. Steam's "recommended" label (optional)
    meta.json          dataset statistics (rows read, rows dropped, etc.)
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from topics import TOPIC_REGEX

# Column names differ slightly between Steam review datasets, so accept a few aliases.
COLUMN_ALIASES = {
    "game": ["app_name", "game", "title", "app_title"],
    "text": ["review", "review_text", "text"],
    "recommended": ["recommended", "voted_up", "is_positive", "review_score"],
    "timestamp": ["timestamp_created", "timestamp", "date", "created"],
    "language": ["language", "lang"],
    "helpful": ["votes_helpful", "votes_up", "helpful"],
    "playtime": ["author.playtime_forever", "author_playtime_forever", "playtime_forever"],
}

# Gaming-review filler words that would otherwise dominate the keyword lists.
EXTRA_STOPWORDS = {
    "game", "games", "play", "played", "playing", "just", "like", "really", "dont",
    "don", "t", "s", "ve", "ll", "m", "d", "re", "im", "ive", "10", "hours", "hour",
    "good", "bad", "great", "best", "better", "make", "makes", "thing", "things",
    "way", "lot", "time", "got", "get", "gets", "does", "did", "doesn", "didn",
    "isn", "wasn", "know", "think", "want", "say", "pretty", "people", "recommend",
    "buy", "bought", "steam", "review", "reviews", "yes", "lol", "would", "one",
}
STOPWORDS = list(ENGLISH_STOP_WORDS | EXTRA_STOPWORDS)


def resolve_columns(csv_path):
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    lower = {c.lower(): c for c in header}
    found = {}
    for key, options in COLUMN_ALIASES.items():
        for opt in options:
            if opt.lower() in lower:
                found[key] = lower[opt.lower()]
                break
    missing = [k for k in ("game", "text", "recommended", "timestamp") if k not in found]
    if missing:
        raise SystemExit(
            f"Could not find columns for {missing}.\nColumns in your file: {header}\n"
            "Add the right name to COLUMN_ALIASES at the top of prep_data.py."
        )
    return found


def to_bool(series):
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "positive", "recommended"])


def to_datetime(series):
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.9:  # Unix seconds
        return pd.to_datetime(num, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def pick_top_games(csv_path, cols, top_n, chunksize, english_only):
    print(f"Pass 1/2: counting reviews per game to find the top {top_n} ...")
    use = [cols["game"]] + ([cols["language"]] if english_only and "language" in cols else [])
    counts = pd.Series(dtype="int64")
    for chunk in pd.read_csv(csv_path, usecols=use, chunksize=chunksize):
        if english_only and "language" in cols:
            chunk = chunk[chunk[cols["language"]] == "english"]
        counts = counts.add(chunk[cols["game"]].value_counts(), fill_value=0)
    top = counts.sort_values(ascending=False).head(top_n)
    print("  Selected games:", ", ".join(top.index.astype(str)))
    return list(top.index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="path to the raw reviews CSV")
    ap.add_argument("--out", default="data", help="output folder (default: data)")
    ap.add_argument("--top", type=int, default=25, help="number of most-reviewed games to keep")
    ap.add_argument("--games", nargs="*", help="exact game names to keep (overrides --top)")
    ap.add_argument("--all-languages", action="store_true", help="keep non-English reviews")
    ap.add_argument("--min-words", type=int, default=5, help="drop reviews shorter than this")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--keyword-sample", type=int, default=40_000, help="reviews per game sampled for keyword analysis")
    ap.add_argument("--vader-sample", type=int, default=2_000, help="reviews per game checked with VADER (0 = skip)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    english_only = not args.all_languages

    cols = resolve_columns(args.input)
    print("Using columns:", cols)
    if english_only and "language" not in cols:
        print("  (no language column found, so keeping all reviews)")

    games = args.games or pick_top_games(args.input, cols, args.top, args.chunksize, english_only)
    game_set = set(games)

    usecols = list(dict.fromkeys(cols.values()))
    stats = {"rows_read": 0, "rows_other_games_or_languages": 0, "dropped_empty": 0,
             "dropped_short": 0, "dropped_duplicate": 0, "dropped_bad_date": 0, "rows_kept": 0}
    seen = set()
    monthly_parts, topic_parts, topic_month_parts = [], [], []
    sample = None

    print("Pass 2/2: cleaning and analysing reviews ...")
    for i, chunk in enumerate(pd.read_csv(args.input, usecols=usecols, chunksize=args.chunksize)):
        stats["rows_read"] += len(chunk)
        df = pd.DataFrame({
            "game": chunk[cols["game"]],
            "text": chunk[cols["text"]],
            "positive": to_bool(chunk[cols["recommended"]]),
            "date": to_datetime(chunk[cols["timestamp"]]),
        })
        df["helpful"] = pd.to_numeric(chunk[cols["helpful"]], errors="coerce").fillna(0) if "helpful" in cols else 0
        if "playtime" in cols:
            df["playtime_h"] = pd.to_numeric(chunk[cols["playtime"]], errors="coerce") / 60

        keep = df["game"].isin(game_set)
        if english_only and "language" in cols:
            keep &= chunk[cols["language"]].eq("english")
        stats["rows_other_games_or_languages"] += int((~keep).sum())
        df = df[keep]

        # --- cleaning ---
        df["text"] = df["text"].astype("string").str.strip()
        empty = df["text"].isna() | (df["text"] == "")
        stats["dropped_empty"] += int(empty.sum())
        df = df[~empty]

        words = df["text"].str.split().str.len()
        short = words < args.min_words
        stats["dropped_short"] += int(short.sum())
        df = df[~short]

        bad_date = df["date"].isna()
        stats["dropped_bad_date"] += int(bad_date.sum())
        df = df[~bad_date]

        keys = pd.util.hash_pandas_object(df[["game", "text"]], index=False).tolist()
        in_seen = np.fromiter((k in seen for k in keys), dtype=bool, count=len(keys))
        dup = pd.Series(keys).duplicated().to_numpy() | in_seen
        seen.update(keys)
        stats["dropped_duplicate"] += int(dup.sum())
        df = df[~dup].copy()
        stats["rows_kept"] += len(df)
        if df.empty:
            continue

        df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
        lower = df["text"].str.lower()
        for topic, rx in TOPIC_REGEX.items():
            df[topic] = lower.str.contains(rx, na=False)
        topic_names = list(TOPIC_REGEX)

        # --- aggregate this chunk ---
        monthly_parts.append(df.groupby(["game", "month"]).agg(reviews=("positive", "size"), positive=("positive", "sum")))
        topic_parts.append(df.groupby(["game", "positive"])[topic_names].sum())
        topic_month_parts.append(df.groupby(["game", "month", "positive"])[topic_names].sum())

        # --- keep a random sample per game for keyword analysis (reservoir style) ---
        df["_r"] = rng.random(len(df))
        part = df[["game", "text", "positive", "helpful", "date", "_r"] + (["playtime_h"] if "playtime_h" in df else [])]
        sample = part if sample is None else pd.concat([sample, part])
        sample = sample.sort_values("_r").groupby("game", sort=False).head(args.keyword_sample)

        print(f"  chunk {i + 1}: read {stats['rows_read']:,} rows, kept {stats['rows_kept']:,} ({time.time() - t0:.0f}s)")

    if stats["rows_kept"] == 0:
        raise SystemExit("No reviews left after filtering. Check --games names / language column.")

    # ---------- combine ----------
    monthly = pd.concat(monthly_parts).groupby(level=[0, 1]).sum().reset_index()
    monthly["pct_positive"] = (monthly["positive"] / monthly["reviews"] * 100).round(2)
    monthly.to_csv(out / "monthly.csv", index=False)

    summary = monthly.groupby("game").agg(
        reviews=("reviews", "sum"), positive=("positive", "sum"),
        first_month=("month", "min"), last_month=("month", "max"))
    summary["negative"] = summary["reviews"] - summary["positive"]
    summary["pct_positive"] = (summary["positive"] / summary["reviews"] * 100).round(2)
    if "playtime_h" in sample:  # median hours played by reviewers (from the random sample)
        med = sample.groupby(["game", "positive"])["playtime_h"].median().unstack()
        summary["median_playtime_pos_h"] = med.get(True, pd.Series(dtype=float)).round(1)
        summary["median_playtime_neg_h"] = med.get(False, pd.Series(dtype=float)).round(1)
    summary = summary.sort_values("reviews", ascending=False).reset_index()
    summary.to_csv(out / "summary.csv", index=False)

    topics = pd.concat(topic_parts).groupby(level=[0, 1]).sum()
    topics = topics.reset_index().melt(id_vars=["game", "positive"], var_name="topic", value_name="mentions")
    topics = topics.pivot_table(index=["game", "topic"], columns="positive", values="mentions", fill_value=0).reset_index()
    topics.columns.name = None
    topics = topics.rename(columns={True: "mentions_positive", False: "mentions_negative"})
    for c in ("mentions_positive", "mentions_negative"):
        if c not in topics:
            topics[c] = 0
    topics = topics.merge(summary[["game", "positive", "negative"]], on="game")
    topics["pct_of_positive_reviews"] = (topics["mentions_positive"] / topics["positive"].clip(lower=1) * 100).round(2)
    topics["pct_of_negative_reviews"] = (topics["mentions_negative"] / topics["negative"].clip(lower=1) * 100).round(2)
    total = topics["mentions_positive"] + topics["mentions_negative"]
    topics["pct_mentions_negative"] = (topics["mentions_negative"] / total.clip(lower=1) * 100).round(2)
    topics = topics.drop(columns=["positive", "negative"])
    topics.to_csv(out / "topics.csv", index=False)

    tm = pd.concat(topic_month_parts).groupby(level=[0, 1, 2]).sum().reset_index()
    tm = tm.melt(id_vars=["game", "month", "positive"], var_name="topic", value_name="mentions")
    tm = tm.pivot_table(index=["game", "month", "topic"], columns="positive", values="mentions", fill_value=0).reset_index()
    tm.columns.name = None
    tm = tm.rename(columns={True: "mentions_positive", False: "mentions_negative"})
    for c in ("mentions_positive", "mentions_negative"):
        if c not in tm:
            tm[c] = 0
    tm.to_csv(out / "topics_monthly.csv", index=False)

    # ---------- keywords: words typical of negative vs positive reviews ----------
    print("Finding keywords ...")
    kw_rows = []
    for game, g in sample.groupby("game"):
        if len(g) < 50:
            continue
        vec = CountVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2), min_df=max(5, len(g) // 2000),
                              binary=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b", max_features=20_000)
        X = vec.fit_transform(g["text"].str.lower())
        terms = vec.get_feature_names_out()
        pos_mask = g["positive"].to_numpy()
        n_pos, n_neg = max(pos_mask.sum(), 1), max((~pos_mask).sum(), 1)
        c_pos = np.asarray(X[pos_mask].sum(axis=0)).ravel()
        c_neg = np.asarray(X[~pos_mask].sum(axis=0)).ravel()
        # smoothed log-odds: >0 means the term is more typical of negative reviews
        log_odds = np.log((c_neg + 1) / (n_neg + 1)) - np.log((c_pos + 1) / (n_pos + 1))
        frame = pd.DataFrame({"term": terms, "count_positive": c_pos, "count_negative": c_neg, "log_odds_negative": log_odds})
        frame["pct_positive_reviews"] = (frame["count_positive"] / n_pos * 100).round(2)
        frame["pct_negative_reviews"] = (frame["count_negative"] / n_neg * 100).round(2)
        min_count = max(10, int(0.005 * len(g)))
        # rank by "distinctive AND common": log-odds weighted by how often the term appears
        frame["score_neg"] = frame["log_odds_negative"] * np.log1p(frame["count_negative"])
        frame["score_pos"] = -frame["log_odds_negative"] * np.log1p(frame["count_positive"])
        neg = frame[(frame["count_negative"] >= min_count) & (frame["log_odds_negative"] > 0)].nlargest(25, "score_neg").assign(side="complaint")
        pos = frame[(frame["count_positive"] >= min_count) & (frame["log_odds_negative"] < 0)].nlargest(25, "score_pos").assign(side="praise")
        kw = pd.concat([neg, pos]).drop(columns=["score_neg", "score_pos"])
        kw_rows.append(kw.assign(game=game, sample_size=len(g)))
    keywords = pd.concat(kw_rows) if kw_rows else pd.DataFrame()
    keywords["log_odds_negative"] = keywords["log_odds_negative"].round(3)
    keywords.to_csv(out / "keywords.csv", index=False)

    # ---------- example reviews ----------
    ex = sample.copy()
    n_words = ex["text"].str.split().str.len()
    ex = ex[(n_words >= 15) & (n_words <= 150)]
    ex = ex.sort_values("helpful", ascending=False).groupby(["game", "positive"]).head(5)
    ex["text"] = ex["text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 700)
    ex[["game", "positive", "helpful", "date", "text"]].to_csv(out / "examples.csv", index=False)

    # ---------- validation: does VADER agree with Steam's thumbs up / down? ----------
    vader_note = "skipped"
    if args.vader_sample > 0:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            print("Running VADER validation ...")
            v = sample.groupby("game").head(args.vader_sample).copy()
            v["compound"] = v["text"].str.slice(0, 2000).map(lambda t: sia.polarity_scores(t)["compound"])
            v["vader_positive"] = v["compound"] >= 0.05
            val = v.groupby("game").apply(lambda d: pd.Series({
                "checked": len(d),
                "agreement_pct": round((d["vader_positive"] == d["positive"]).mean() * 100, 2),
                "steam_pct_positive": round(d["positive"].mean() * 100, 2),
                "vader_pct_positive": round(d["vader_positive"].mean() * 100, 2),
                "avg_compound": round(d["compound"].mean(), 3),
            }), include_groups=False).reset_index()
            val["checked"] = val["checked"].astype(int)
            val.to_csv(out / "validation.csv", index=False)
            vader_note = f"VADER checked {len(v):,} reviews"
        except ImportError:
            vader_note = "skipped (pip install vaderSentiment to enable)"
            print("  vaderSentiment not installed, skipping validation.")

    meta = {
        "source_file": Path(args.input).name,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "english_only": english_only and "language" in cols,
        "min_words": args.min_words,
        "games": len(summary),
        "date_range": [str(summary["first_month"].min())[:7], str(summary["last_month"].max())[:7]],
        "keyword_sample_per_game": args.keyword_sample,
        "validation": vader_note,
        "seconds": round(time.time() - t0, 1),
        **stats,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Done. Files written to {out.resolve()}")


if __name__ == "__main__":
    main()

"""
Creates a FAKE reviews file (demo_reviews.csv) with the same columns as the
Kaggle "Steam Reviews Dataset 2021", so you can test the whole project before
the real multi-GB download finishes. The games and reviews are made up.

    python make_demo_data.py            # ~150,000 fake reviews
    python prep_data.py --input demo_reviews.csv --out data_demo
    streamlit run app.py -- --data data_demo
"""
import argparse

import numpy as np
import pandas as pd

GAMES = {
    # name: (base % positive, launch year, event month index with a bad patch or None)
    "Starforge Odyssey": (0.86, 2016, 40),
    "Pixel Farm Valley": (0.95, 2017, None),
    "Iron Siege Online": (0.58, 2018, 20),
    "Neon Drift Racers": (0.74, 2016, None),
    "Crypt of Echoes": (0.90, 2019, 10),
    "Battle Royale X": (0.48, 2017, 30),
    "Tiny Kingdom Builder": (0.82, 2018, None),
    "Deep Ocean Survival": (0.69, 2016, 50),
}

POS = {
    "Gameplay": ["the combat feels amazing and the controls are tight", "gameplay loop is addictive and every mission is different",
                 "level design is clever and rewards exploration", "great replay value, I keep coming back"],
    "Graphics & Visuals": ["the art style is gorgeous", "beautiful visuals and smooth animations", "textures and lighting look stunning"],
    "Story & Characters": ["the story made me care about every character", "excellent writing and a satisfying ending", "deep lore and a great narrative"],
    "Price & Value": ["totally worth the price", "got it on sale and it is worth every cent", "huge amount of content for the money"],
    "Sound & Music": ["the soundtrack is incredible", "amazing music and voice acting"],
    "Multiplayer & Servers": ["co-op with friends is a blast", "matchmaking is fast and the online community is friendly"],
    "Updates & Developers": ["the devs keep adding free updates", "developers listen to the community and patch quickly"],
}
NEG = {
    "Bugs & Performance": ["constant crashes to desktop", "terrible fps drops and stutter even on a good pc", "so many bugs and glitches it feels broken",
                           "the latest patch made performance worse, it freezes every hour", "poorly optimized, lag everywhere"],
    "Price & Value": ["way overpriced for what you get", "full of microtransactions and pay to win dlc", "not worth the money, I asked for a refund"],
    "Gameplay": ["the grind is boring and repetitive", "controls feel clunky and combat is unbalanced", "missions are all the same"],
    "Multiplayer & Servers": ["servers are always down", "full of cheaters and hackers", "matchmaking takes forever and players are toxic"],
    "Updates & Developers": ["the devs abandoned this game", "stuck in early access forever with no roadmap", "every update breaks something"],
    "Story & Characters": ["the story is shallow and the ending makes no sense", "characters are forgettable"],
    "Graphics & Visuals": ["graphics look outdated and ugly", "animations are stiff"],
}
FILLER_POS = ["Highly recommended.", "10/10 would play again.", "One of my favorites this year.", "Easily hundreds of hours of fun."]
FILLER_NEG = ["Can't recommend it right now.", "Wait for a sale or skip it.", "Really disappointed.", "Uninstalled."]


def make_review(rng, positive, bug_boost):
    bank, filler = (POS, FILLER_POS) if positive else (NEG, FILLER_NEG)
    topics = list(bank)
    weights = np.ones(len(topics))
    if not positive:
        weights[topics.index("Bugs & Performance")] += bug_boost
    weights /= weights.sum()
    picks = rng.choice(len(topics), size=rng.integers(1, 4), replace=False, p=weights)
    sentences = [rng.choice(bank[topics[i]]) for i in picks]
    if positive and rng.random() < 0.15:
        sentences.append(rng.choice(NEG["Bugs & Performance"]) + " but it is still fun")
    if not positive and rng.random() < 0.15:
        sentences.append("the " + rng.choice(["art style is nice", "music is fine", "idea is good"]) + " though")
    text = ". ".join(s.capitalize() for s in sentences) + ". " + rng.choice(filler)
    text += f" ({int(rng.integers(2, 900))} hours in)"  # makes most fake reviews unique
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=150_000)
    ap.add_argument("--out", default="demo_reviews.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    names = list(GAMES)
    popularity = rng.dirichlet(np.ones(len(names)) * 2)
    rows = []
    for i in range(args.rows):
        g = names[rng.choice(len(names), p=popularity)]
        base, year, event = GAMES[g]
        start = pd.Timestamp(f"{year}-01-01")
        month_idx = int(min(rng.exponential(18), 60))
        date = start + pd.DateOffset(months=month_idx) + pd.Timedelta(days=int(rng.integers(0, 28)))
        p = base + 0.002 * month_idx  # slowly improving opinion
        bug_boost = 0.0
        if event is not None and 0 <= month_idx - event < 4:  # a bad patch: reviews dip and bug complaints spike
            p -= 0.30
            bug_boost = 6.0
        positive = rng.random() < min(max(p, 0.05), 0.99)
        lang = "english" if rng.random() < 0.75 else rng.choice(["schinese", "russian", "brazilian", "german"])
        text = make_review(rng, positive, bug_boost)
        r = rng.random()
        if r < 0.02:
            text = rng.choice(["good", "bad game", "yes", "10/10", ""])  # junk the cleaner should drop
        rows.append((g, lang, text, int(date.timestamp()), positive,
                     int(rng.poisson(3) * (8 if rng.random() < 0.02 else 1)), int(rng.gamma(2, 3000))))

    df = pd.DataFrame(rows, columns=["app_name", "language", "review", "timestamp_created", "recommended",
                                     "votes_helpful", "author.playtime_forever"])
    df = pd.concat([df, df.sample(frac=0.01, random_state=1)])  # duplicated reviews
    df.insert(0, "app_id", df["app_name"].map({n: 100000 + i for i, n in enumerate(names)}))
    df.insert(2, "review_id", np.arange(len(df)) + 10_000_000)
    df["timestamp_updated"] = df["timestamp_created"]
    df["votes_funny"] = 0
    df.to_csv(args.out)
    print(f"Wrote {len(df):,} fake reviews to {args.out}")


if __name__ == "__main__":
    main()

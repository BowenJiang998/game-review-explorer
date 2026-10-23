# Video Game Review Explorer

A big-data analytics website that analyzes millions of Steam player reviews to show what people
like and dislike about different games. Pick a game to see its positive / negative split, the
topics players praise and complain about (gameplay, bugs, price, servers…), the words that set
good and bad reviews apart, and how opinion changed over time. You can also compare games side by side.

Built for AAI 627 (Option 1: Big Data Analytics App using vibe coding).

## How it works

```
steam_reviews.csv (≈21M reviews, several GB)
        │  prep_data.py  – reads in chunks, filters, cleans, analyzes
        ▼
data/  (small CSV/JSON summaries, a few MB)
        │  app.py  – Streamlit website
        ▼
interactive dashboard in the browser
```

The heavy work happens once in `prep_data.py`, so the website loads instantly and can be
deployed for free on Streamlit Community Cloud without uploading the raw data.

| File | Purpose |
|---|---|
| `prep_data.py` | Big-data step: chunked loading, cleaning, topic detection, keyword analysis, monthly trends, VADER validation |
| `topics.py` | Keyword lists that define each feedback topic (edit to tune) |
| `app.py` | The Streamlit website |
| `make_demo_data.py` | Makes a fake dataset with the same columns, for testing |
| `requirements.txt` | Python packages |

## 1. Install

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## 2. Try it with demo data (2 minutes)

```bash
python make_demo_data.py
python prep_data.py --input demo_reviews.csv --out data_demo
streamlit run app.py -- --data data_demo
```

The demo games and reviews are made up. They're only for checking that everything works.

## 3. Run it on the real data

1. Download **Steam Reviews Dataset 2021** from Kaggle:
   https://www.kaggle.com/datasets/najzeko/steam-reviews-2021
2. Unzip it and put `steam_reviews.csv` in this folder.
3. Process it (expect roughly 10–30 minutes on a laptop):

   ```bash
   python prep_data.py --input steam_reviews.csv --top 25
   ```

   Options:
   - `--top 30` – number of most-reviewed games to keep
   - `--games "Terraria" "Rust"` – pick exact games instead
   - `--chunksize 200000` – use this if your computer runs low on memory
   - `--vader-sample 0` – skip the VADER check to save time
4. Start the website:

   ```bash
   streamlit run app.py
   ```

When it finishes, `data/meta.json` has the real numbers (rows read, duplicates removed,
reviews analyzed, etc.) for the dataset status report.

If the script says it can't find a column, it prints the columns in your file. Add the right
name to `COLUMN_ALIASES` at the top of `prep_data.py`.

## 4. Deploy (optional)

1. Push this folder to a GitHub repo, **including the `data/` folder** (the raw CSV is excluded by `.gitignore`).
2. Go to https://share.streamlit.io, connect the repo, and choose `app.py`.

## Methods

- **Positive / negative:** each reviewer's own Steam "Recommended / Not Recommended" choice.
- **Cleaning:** English only; removes empty reviews, reviews under 5 words, and exact duplicates; converts Unix timestamps to dates.
- **Topics:** a review mentions a topic if it matches that topic's keyword patterns in `topics.py`.
  The site shows the % of positive and of negative reviews that mention each topic.
- **Keywords:** on a random sample (up to 40,000 reviews per game), single words and two-word phrases
  are compared between negative and positive reviews with smoothed log-odds, weighted by frequency.
- **Trends:** reviews grouped by month, with optional 3-month smoothing.
- **Validation:** VADER sentiment on a sample of reviews per game, compared with the recommended label.
- **Privacy:** reviewer IDs are never loaded; only aggregated results and anonymous example texts are shown.

## Limitations

- Keyword topics miss sarcasm and context ("not buggy at all" still counts as a bugs mention).
- Very popular games dominate the dataset, while smaller games have noisier monthly trends.
- The 2021 dataset doesn't include reviews written after it was collected.

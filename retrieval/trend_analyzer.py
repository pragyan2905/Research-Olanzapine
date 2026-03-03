import pandas as pd
import re
from collections import Counter


def compute_yearly_volume(df):
    volume = df["year"].value_counts().sort_index()
    return volume


def compute_yearly_improvement(df, extract_numeric_metrics):
    yearly_data = []

    for _, row in df.iterrows():
        metrics, _ = extract_numeric_metrics(row["abstract"])
        for m in metrics:
            yearly_data.append({
                "year": row["year"],
                "improvement": m["value"]
            })

    if not yearly_data:
        return None

    yearly_df = pd.DataFrame(yearly_data)
    return yearly_df.groupby("year")["improvement"].mean().sort_index()


def compute_keyword_trend(df, top_k=5):
    words_by_year = {}

    for _, row in df.iterrows():
        year = row["year"]
        text = row["abstract"].lower()
        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text)

        if year not in words_by_year:
            words_by_year[year] = []

        words_by_year[year].extend(tokens)

    trend_summary = {}

    for year, words in words_by_year.items():
        counter = Counter(words)
        trend_summary[year] = counter.most_common(top_k)

    return trend_summary
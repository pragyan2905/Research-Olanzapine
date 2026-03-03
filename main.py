import pandas as pd
import glob
import re
from datetime import datetime
import arxiv

from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import ReRanker
from retrieval.contribution_extractor import extract_key_sentences


def load_latest_dataset():
    files = glob.glob("data/raw/arxiv_cs.LG_*.csv")
    if not files:
        raise FileNotFoundError("No arXiv dataset found in data/raw/")
    latest_file = sorted(files)[-1]
    print(f"Loading dataset: {latest_file}")
    return pd.read_csv(latest_file)

def build_index(df):
    embedder = Embedder()
    embeddings = embedder.encode(df["abstract"].tolist())
    dimension = embeddings.shape[1]
    store = VectorStore(dimension)
    store.add(embeddings)
    return embedder, store

def research_query(df, embedder, store, query, top_k=5):
    query_embedding = embedder.encode([query])
    scores, indices = store.search(query_embedding, top_k=50)

    candidates = df.iloc[indices[0]].copy()
    candidate_texts = candidates["abstract"].tolist()

    reranker = ReRanker()
    rerank_scores = reranker.rerank(query, candidate_texts)
    candidates["rerank_score"] = rerank_scores

    def keyword_boost(query, text):
        query_terms = query.lower().split()
        text_lower = text.lower()
        return sum(1 for term in query_terms if term in text_lower)

    candidates["keyword_score"] = [
        keyword_boost(query, abstract)
        for abstract in candidate_texts
    ]

    candidates["final_score"] = (
        0.8 * candidates["rerank_score"]
        + 0.2 * candidates["keyword_score"]
    )

    return candidates.sort_values(
        by="final_score",
        ascending=False
    ).head(top_k)


def fetch_dynamic_arxiv(query, max_results=50):
    print("\nFetching dynamic papers from arXiv...")

    search = arxiv.Search(
        query=f'all:"{query}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in search.results():
        papers.append({
            "paper_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "abstract": result.summary,
            "authors": ", ".join([a.name for a in result.authors]),
            "year": result.published.year,
        })

    return pd.DataFrame(papers)


def extract_year_constraint(query):
    query_lower = query.lower()

    match = re.search(r"(after|since)\s+(\d{4})", query_lower)
    if match:
        return int(match.group(2))

    match = re.search(r"last\s+(\d+)\s+years", query_lower)
    if match:
        n_years = int(match.group(1))
        return datetime.now().year - n_years

    if "recent" in query_lower:
        return datetime.now().year - 2

    return None

def clean_query(query):
    query = re.sub(r"(after|since)\s+\d{4}", "", query.lower())
    query = re.sub(r"last\s+\d+\s+years", "", query)
    query = query.replace("recent", "")
    return query.strip()


def main():
    static_df = load_latest_dataset()

    user_query = input("\nEnter research query: ")

    min_year = extract_year_constraint(user_query)
    if min_year:
        print(f"Applying year filter: {min_year}+")
        static_df = static_df[static_df["year"] >= min_year]

    dynamic_df = fetch_dynamic_arxiv(user_query, max_results=50)

    df = pd.concat([static_df, dynamic_df], ignore_index=True)
    df = df.drop_duplicates(subset=["title"])

    print(f"\nTotal papers after merge: {len(df)}")

    print("\nBuilding index...")
    embedder, store = build_index(df)

    cleaned_query = clean_query(user_query)
    results = research_query(df, embedder, store, cleaned_query, top_k=5)

    print("\nStructured Research Summary:\n")

    for _, row in results.iterrows():
        print("=" * 80)
        print(f"Title: {row['title']}")
        print(f"Year: {row['year']}")
        print("\nKey Contribution:")

        key_points = extract_key_sentences(row["abstract"])

        if key_points:
            for point in key_points:
                print(f"- {point}")
        else:
            print("- No explicit contribution sentence detected.")

        print("\n")


if __name__ == "__main__":
    main()
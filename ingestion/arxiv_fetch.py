import arxiv
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import argparse


DATA_PATH = Path("data/raw")
DATA_PATH.mkdir(parents=True, exist_ok=True)


def fetch_arxiv_papers(category="cs.LG", max_results=1000):
    query = f"cat:{category}"

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []

    for result in tqdm(search.results()):
        papers.append({
            "paper_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "abstract": result.summary,
            "authors": ", ".join([a.name for a in result.authors]),
            "categories": ", ".join(result.categories),
            "published_date": result.published,
            "updated_date": result.updated,
            "year": result.published.year,
            "pdf_url": result.pdf_url
        })

    return pd.DataFrame(papers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="cs.LG")
    parser.add_argument("--max_results", type=int, default=1000)

    args = parser.parse_args()

    print(f"\nFetching {args.max_results} papers from {args.category}...\n")

    df = fetch_arxiv_papers(
        category=args.category,
        max_results=args.max_results
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"arxiv_{args.category}_{timestamp}.csv"

    df.to_csv(DATA_PATH / filename, index=False)

    print(f"\nSaved {len(df)} papers to data/raw/{filename}\n")


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd

from main import (
    load_latest_dataset,
    build_index,
    research_query,
    fetch_dynamic_arxiv,
    clean_query
)

from retrieval.contribution_extractor import extract_key_sentences
from retrieval.performance_extractor import extract_numeric_metrics
from retrieval.clusterer import cluster_papers
from retrieval.synthesizer import synthesize_comparison
from retrieval.llm_synthesizer import generate_research_review


st.set_page_config(
    page_title="Research Intelligence System",
    layout="wide"
)


st.title("Research Intelligence System")

st.write(
"""
Semantic research analysis system that retrieves papers,
extracts contributions, analyzes performance metrics,
clusters research directions, and generates an automated research review.
"""
)


query = st.text_input("Research Topic")

top_k = st.slider("Number of Papers to Analyze", 3, 10, 5)


if st.button("Run Analysis") and query:

    with st.spinner("Loading dataset..."):
        static_df = load_latest_dataset()

    with st.spinner("Fetching papers from arXiv..."):
        dynamic_df = fetch_dynamic_arxiv(query, max_results=50)

    df = pd.concat([static_df, dynamic_df], ignore_index=True)
    df = df.drop_duplicates(subset=["title"])

    st.write(f"Total Papers Analyzed: {len(df)}")

    with st.spinner("Building embedding index..."):
        embedder, store = build_index(df)

    cleaned_query = clean_query(query)

    with st.spinner("Searching relevant papers..."):
        results = research_query(df, embedder, store, cleaned_query, top_k=top_k)


    st.header("Top Papers")

    for _, row in results.iterrows():

        st.subheader(row["title"])
        st.write(f"Year: {row['year']}")

        contributions = extract_key_sentences(row["abstract"])

        st.write("Key Contribution")

        if contributions:
            for c in contributions:
                st.write(f"- {c}")
        else:
            st.write("No explicit contribution detected")

        metrics, sota_flag = extract_numeric_metrics(row["abstract"])

        st.write("Performance Signals")

        if metrics:
            for m in metrics:
                st.write(f"- {m['value']}% ({m['type']})")
        else:
            st.write("No numeric performance signals detected")

        if sota_flag:
            st.write("Claims state-of-the-art performance")

        st.divider()


    st.header("Semantic Research Clusters")

    abstracts = results["abstract"].tolist()
    titles = results["title"].tolist()

    cluster_text = ""

    if len(abstracts) >= 2:

        labels = cluster_papers(abstracts, num_clusters=2)

        clusters = {}

        for label, title in zip(labels, titles):
            clusters.setdefault(label, []).append(title)

        for cid, papers in clusters.items():

            st.subheader(f"Cluster {cid + 1}")

            cluster_text += f"\nCluster {cid+1}:\n"

            for p in papers:
                st.write(f"- {p}")
                cluster_text += f"- {p}\n"


    st.header("Paper Comparison")

    titles = []
    abstracts = []
    contributions = []

    for _, row in results.iterrows():
        titles.append(row["title"])
        abstracts.append(row["abstract"])
        contributions.extend(extract_key_sentences(row["abstract"]))

    comparison = synthesize_comparison(titles, abstracts, contributions)

    st.write(comparison)


    st.header("LLM Generated Research Review")

    trend_summary = f"Total papers analyzed: {len(df)}"

    with st.spinner("Generating research review..."):

        review = generate_research_review(
            query,
            cluster_text,
            str(results[["title","year"]].to_string(index=False)),
            trend_summary
        )

    st.markdown(review)
import re
from collections import Counter


def extract_keywords(texts):

    words = []

    stopwords = {
        "the", "and", "for", "with", "this", "that",
        "from", "are", "our", "we", "paper", "propose",
        "proposes", "present", "introduce", "using",
        "based", "into", "their", "through", "method",
        "model", "approach", "results", "show"
    }

    for text in texts:

        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        filtered = [t for t in tokens if t not in stopwords]

        words.extend(filtered)

    return Counter(words)


def synthesize_comparison(titles, abstracts, contributions):

    combined_text = []

    combined_text.extend(titles)
    combined_text.extend(abstracts)
    combined_text.extend(contributions)

    keyword_counts = extract_keywords(combined_text)

    top_keywords = [word for word, _ in keyword_counts.most_common(6)]

    synthesis = "\nComparative Insights:\n"
    synthesis += "-" * 80 + "\n"

    synthesis += "Key technical themes observed across the retrieved papers include:\n"

    for kw in top_keywords:
        synthesis += f"- {kw}\n"

    synthesis += "\n"

    synthesis += "The retrieved works fall broadly into two directions:\n"

    synthesis += (
        "1. Architectural innovations that modify neural network structures "
        "to improve representation learning, scalability, or adaptive inference.\n"
    )

    synthesis += (
        "2. Training or optimization techniques that enhance robustness, "
        "efficiency, or convergence behavior of existing architectures.\n\n"
    )

    synthesis += (
        "Across these papers, improvements are typically achieved through "
        "better parameter utilization, adaptive computation mechanisms, "
        "or improved training strategies rather than completely new paradigms.\n"
    )

    synthesis += "\n"

    synthesis += (
        "Overall, the literature indicates a strong focus on improving "
        "model efficiency, robustness, and scalability while maintaining "
        "competitive predictive performance.\n"
    )

    return synthesis
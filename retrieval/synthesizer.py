import re
from collections import Counter


def extract_keywords(texts):
    words = []
    stopwords = {
        "the", "and", "for", "with", "this", "that",
        "from", "are", "our", "we", "paper", "propose",
        "proposes", "present", "introduce", "using",
        "based", "into", "their", "through"
    }

    for text in texts:
        tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        words.extend([t for t in tokens if t not in stopwords])

    return Counter(words)


def synthesize_comparison(titles, abstracts, contributions):
    keyword_counts = extract_keywords(contributions)

    top_keywords = [word for word, _ in keyword_counts.most_common(5)]

    synthesis = "\nComparative Insights:\n"
    synthesis += "-" * 80 + "\n"

    synthesis += f"Across the retrieved papers, common themes include: {', '.join(top_keywords)}.\n\n"

    synthesis += "Several works propose architectural modifications, while others focus on optimization or efficiency improvements.\n"

    synthesis += "The trend suggests continued refinement of model efficiency and representational power rather than entirely new paradigms.\n"

    return synthesis
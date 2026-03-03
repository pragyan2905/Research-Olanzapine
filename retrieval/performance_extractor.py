import re


def extract_performance_metrics(text):
    metrics = []

    text_lower = text.lower()

    # Percentage improvements
    percent_patterns = re.findall(r'(\d+\.?\d*)\s*%', text_lower)
    for p in percent_patterns:
        metrics.append(f"{p}% improvement/metric mentioned")

    # Improvement phrases
    improvement_patterns = re.findall(
        r'(improves?|improvement|outperforms?|reduces?|achieves?)[^.]*\.',
        text_lower
    )
    for match in improvement_patterns:
        metrics.append(match.strip())

    # SOTA claims
    if "state-of-the-art" in text_lower or "sota" in text_lower:
        metrics.append("Claims state-of-the-art performance")

    # Benchmark mentions
    benchmark_patterns = re.findall(
        r'on\s+([A-Z][A-Za-z0-9\-]+)',
        text
    )
    for bench in benchmark_patterns:
        metrics.append(f"Evaluated on {bench}")

    return list(set(metrics))
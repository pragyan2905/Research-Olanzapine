import re


def extract_numeric_metrics(text):
    text_lower = text.lower()
    metrics = []

    percent_matches = re.findall(r'(\d+\.?\d*)\s*%', text_lower)

    for value in percent_matches:
        value_float = float(value)

        if "accuracy" in text_lower:
            metric_type = "Accuracy"
        elif "f1" in text_lower:
            metric_type = "F1 Score"
        elif "bleu" in text_lower:
            metric_type = "BLEU"
        elif "compute" in text_lower or "flops" in text_lower:
            metric_type = "Compute Reduction"
        elif "memory" in text_lower:
            metric_type = "Memory Reduction"
        elif "speed" in text_lower or "faster" in text_lower:
            metric_type = "Speed Improvement"
        else:
            metric_type = "General Improvement"

        metrics.append({
            "value": value_float,
            "type": metric_type
        })

    sota_flag = "state-of-the-art" in text_lower or "sota" in text_lower

    return metrics, sota_flag
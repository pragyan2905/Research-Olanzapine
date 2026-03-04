import re


def extract_method_sentences(text):

    method_patterns = [
        r"we propose .*?\.",
        r"we introduce .*?\.",
        r"our method .*?\.",
        r"the model .*?\.",
        r"the framework .*?\.",
        r"we design .*?\."
    ]

    sentences = re.split(r'(?<=[.!?]) +', text)

    results = []

    for s in sentences:
        for p in method_patterns:
            if re.search(p, s.lower()):
                results.append(s)
                break

    return results[:3]
import re


def extract_key_sentences(abstract):
    sentences = re.split(r'(?<=[.!?])\s+', abstract)

    key_patterns = [
        "we propose",
        "we introduce",
        "we present",
        "this paper proposes",
        "this work proposes",
        "we develop",
    ]

    key_sentences = []

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(pattern in sentence_lower for pattern in key_patterns):
            key_sentences.append(sentence.strip())

    return key_sentences[:2]  
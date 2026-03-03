import requests
import json

BASE_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def get_citation_score(arxiv_id):
    params = {
        "query": arxiv_id,
        "limit": 1,
        "fields": "title,citationCount,influentialCitationCount,year"
    }

    response = requests.get(BASE_SEARCH_URL, params=params)

    print("Status code:", response.status_code)
    print("Response text:", response.text)

    if response.status_code != 200:
        return 0

    data = response.json()

    if "data" not in data or len(data["data"]) == 0:
        return 0

    paper = data["data"][0]

    citation_count = paper.get("citationCount", 0)
    influential = paper.get("influentialCitationCount", 0)

    return citation_count + 2 * influential
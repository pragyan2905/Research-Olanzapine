from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash" 

def generate_research_review(query, clusters, performance_table, trend_summary):

    prompt = f"""
You are an expert AI research analyst.

Research Query:
{query}

Clustered Research Directions:
{clusters}

Performance Comparison Table:
{performance_table}

Trend Analysis:
{trend_summary}

Generate a detailed structured literature review with:

1. Overview of Research Landscape
2. Major Methodological Directions
3. Performance Comparison Insights
4. Emerging Trends
5. Open Challenges and Future Directions

Be analytical and structured.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text
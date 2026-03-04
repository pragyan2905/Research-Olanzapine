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

You are an expert machine learning researcher.

Analyze the retrieved papers with focus on:

1. Mathematical intuition behind the proposed methods
2. Model architectures (neural networks, transformers, probabilistic models)
3. Objective functions and optimization strategies
4. Uncertainty estimation techniques
5. Limitations of current methods

Explain the reasoning behind each method, not just the applications.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text
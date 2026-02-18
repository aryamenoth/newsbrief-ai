# --------------------------------------------------------------
# 📰 NEWSBRIEF AI — DYNAMIC REAL-TIME VERSION (PRODUCTION SAFE)
# - Fetches real-time news from NewsAPI
# - Summarizes articles using Hugging Face (BART)
# - Adds a "Why it matters" explanation
# --------------------------------------------------------------

# ---------------- IMPORTS ----------------
import os
import requests
from dotenv import load_dotenv
from transformers import pipeline

# ---------------- ENV SETUP ----------------
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    raise RuntimeError("NEWS_API_KEY not found. Set it in .env or Streamlit secrets.")

MAX_ARTICLES = 5

# ---------------- MODEL (LAZY LOAD) ----------------
_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline(
            task="summarization",
            model="facebook/bart-large-cnn",
            framework="pt"
        )
    return _summarizer

# ---------------- NEWS FETCHING ----------------
def fetch_articles(query, max_results=MAX_ARTICLES):
    """
    Fetch latest news articles from NewsAPI.
    Returns a list of structured article dictionaries.
    """
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={query}&pageSize={max_results}&language=en&apiKey={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception:
        return []

    if data.get("status") != "ok":
        return []

    articles = []
    for a in data.get("articles", []):
        text = a.get("description") or a.get("content")

        articles.append({
            "title": a.get("title"),
            "text": text,
            "source": a.get("source", {}).get("name"),
            "date": (a.get("publishedAt") or "")[:10],
            "url": a.get("url"),
        })

    return articles

# ---------------- SUMMARIZATION ----------------
def summarize_article(text):
    """
    Summarize article text into 3–5 sentences.
    Handles short, missing, or overly long text safely.
    """
    if not text or len(text.split()) < 40:
        return "Summary unavailable due to insufficient article content."

    # BART safety limit
    text = text[:3000]

    summarizer = get_summarizer()

    try:
        summary = summarizer(
            text,
            max_length=120,
            min_length=40,
            do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception:
        return "Summary could not be generated due to processing limits."

# ---------------- WHY IT MATTERS ----------------
def explain_importance(title):
    """
    Generates a short explanation of why the article matters.
    """
    return (
        f"This development around '{title}' may influence public opinion, "
        "market trends, or upcoming decisions related to this topic."
    )

# ---------------- MAIN PIPELINE ----------------
def generate_news_digest(user_query):
    """
    Full pipeline:
    - Fetch articles
    - Summarize each
    - Add 'Why it matters'
    """
    articles = fetch_articles(user_query)
    digest = []

    for art in articles:
        summary = summarize_article(art["text"])
        why_it_matters = explain_importance(art["title"])

        digest.append({
            "title": art["title"],
            "source": art["source"],
            "date": art["date"],
            "summary": summary,
            "why_it_matters": why_it_matters,
            "url": art["url"],
        })

    return digest

# ---------------- LOCAL TEST ----------------
if __name__ == "__main__":
    query = "iPhone 17 features"
    print(f"\nFetching news for: {query}\n")

    digest = generate_news_digest(query)

    for i, d in enumerate(digest, 1):
        print(f"{i}. {d['title']} ({d['source']}, {d['date']})")
        print(f"   Summary: {d['summary']}")
        print(f"   Why it matters: {d['why_it_matters']}")
        print(f"   Link: {d['url']}\n")
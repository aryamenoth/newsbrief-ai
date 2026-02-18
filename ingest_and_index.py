# --------------------------------------------------------------
# 📰 NEWSBRIEF AI — STEP 1
# This script prepares the data for your app:
# 1. Creates a few sample news articles (mock dataset)
# 2. Converts them into "embeddings" using AI (numerical meaning)
# 3. Saves those embeddings inside FAISS for quick searching
# 4. Stores article info in a small SQLite database
# --------------------------------------------------------------

# ----------- IMPORTS -----------
import os
import json
import sqlite3        # lightweight database built into Python
from tqdm import tqdm  # shows progress bars in terminal
from sentence_transformers import SentenceTransformer  # creates embeddings (vector form of text)
import faiss           # Facebook AI Similarity Search library for searching by meaning

# ----------- CONFIGURATION -----------
# These are file paths and model names used later
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast embedding model from Hugging Face
DB_PATH = "news.db"                # where we’ll store article info
FAISS_INDEX_PATH = "faiss.index"   # where FAISS vector index will be saved
META_JSONL = "meta.jsonl"          # a simple metadata file for reference


# ----------- STEP 1: SAMPLE ARTICLES -----------
def fetch_sample_articles():
    """
    This is a placeholder dataset.
    In a real version, you’d fetch news from APIs or the GDELT dataset.
    """
    articles = [
        {
            "id": "1",
            "title": "AI Regulation Bill Passed in Europe",
            "text": (
                "The European Parliament has passed a comprehensive AI regulation framework "
                "aiming to ensure transparency, fairness, and accountability in artificial intelligence systems. "
                "The law affects tech companies and startups operating across the EU."
            ),
            "source": "BBC News",
            "date": "2025-10-01",
            "url": "https://example.com/ai-regulation",
            "tags": ["politics", "tech"]
        },
        {
            "id": "2",
            "title": "New Breakthrough in Battery Technology",
            "text": (
                "Scientists have developed a new solid-state battery that promises faster charging "
                "and longer lifespan compared to current lithium-ion batteries. "
                "The innovation could reshape the electric vehicle industry."
            ),
            "source": "TechCrunch",
            "date": "2025-10-02",
            "url": "https://example.com/battery",
            "tags": ["science", "technology"]
        },
        {
            "id": "3",
            "title": "Stock Markets Hit Record Highs",
            "text": (
                "Global stock markets reached record highs following strong quarterly earnings "
                "and investor optimism about AI-driven growth. Analysts warn, however, "
                "that inflation concerns may cause short-term volatility."
            ),
            "source": "Reuters",
            "date": "2025-10-03",
            "url": "https://example.com/stocks",
            "tags": ["finance", "business"]
        }
    ]
    return articles


# ----------- STEP 2: DATABASE SETUP -----------
def init_db():
    """
    Creates a simple SQLite database if it doesn’t already exist.
    This database will just store article metadata (title, source, date, etc.).
    """
    conn = sqlite3.connect(DB_PATH)  # connects or creates the DB file
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS articles
                 (id TEXT PRIMARY KEY, title TEXT, source TEXT, date TEXT, url TEXT, tags TEXT)''')
    conn.commit()
    return conn


def save_meta(conn, item):
    """
    Saves one article’s metadata into the SQLite database.
    If the article already exists, it replaces it (to avoid duplicates).
    """
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO articles VALUES (?, ?, ?, ?, ?, ?)",
              (item["id"], item["title"], item["source"], item["date"], item["url"], ",".join(item["tags"])))
    conn.commit()


# ----------- STEP 3: CREATE FAISS INDEX -----------
def build_faiss_index(docs):
    """
    Turns all article texts into embeddings (vectors)
    and stores them in a FAISS index for fast similarity search.
    """
    print("⚙️ Creating embeddings using SentenceTransformer...")
    model = SentenceTransformer(EMBED_MODEL)

    # Combine title + text for better context during embedding
    texts = [d["title"] + "\n\n" + d["text"] for d in docs]

    # Convert text into numerical embeddings (vectors)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize the embeddings for cosine similarity search
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    dim = embeddings.shape[1]              # number of dimensions per vector
    index = faiss.IndexFlatIP(dim)         # IP = inner product (used for cosine similarity)
    index.add(embeddings)                  # add all article vectors to index

    # Save FAISS index to file so it can be loaded later
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("💾 FAISS index saved!")

    # Also save metadata (titles, URLs, tags) in a JSON Lines file
    with open(META_JSONL, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({
                "id": d["id"],
                "title": d["title"],
                "url": d["url"],
                "source": d["source"],
                "date": d["date"],
                "tags": d["tags"]
            }) + "\n")
    print("🗂️ Metadata file created!")


# ----------- STEP 4: RUN EVERYTHING -----------
if __name__ == "__main__":
    print("📥 Building local news index...")

    # 1️⃣ Load our sample articles
    docs = fetch_sample_articles()

    # 2️⃣ Initialize the database
    conn = init_db()

    # 3️⃣ Save metadata to database
    for d in tqdm(docs, desc="Saving to database"):
        save_meta(conn, d)

    # 4️⃣ Build FAISS index
    build_faiss_index(docs)

    print("✅ Done! Index and database created successfully.")

# --------------------------------------------------------------
# 🌐 NEWSBRIEF AI — DYNAMIC REAL-TIME STREAMLIT APP
# --------------------------------------------------------------

import streamlit as st
from datetime import datetime
from pipeline import generate_news_digest

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NewsBrief AI 📰",
    page_icon="🧠",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------- HEADER ----------------
st.title("🧠 NewsBrief AI — Real-Time News Summarizer")

st.markdown("""
**What is this?**  
NewsBrief AI fetches the latest real-time news articles on any topic you choose,  
summarizes them into short, easy-to-read insights, and explains *why they matter*.

Perfect for staying informed — fast.
""")

st.markdown("---")

# ---------------- USER INPUT ----------------
query = st.text_input(
    "🔍 Enter a news topic:",
    placeholder="e.g., iPhone 17 features, AI regulation, electric vehicles"
)

# ---------------- CACHE WRAPPER ----------------
@st.cache_data(show_spinner=False, ttl=900)
def cached_news_digest(user_query):
    return generate_news_digest(user_query)

# ---------------- BUTTONS (SIDE BY SIDE) ----------------
col1, col2 = st.columns(2)

with col1:
    get_news = st.button("📰 Get News Summary")

with col2:
    reset_app = st.button("🔄 Reset")

# ---------------- RESET LOGIC ----------------
if reset_app:
    st.session_state.results = None
    st.rerun()

# ---------------- FETCH LOGIC ----------------
if get_news:

    if not query.strip():
        st.warning("⚠️ Please enter a topic to search for news.")
    else:
        with st.spinner(f"Fetching and summarizing news about **{query}**..."):
            try:
                results = cached_news_digest(query)
                st.session_state.results = results

            except Exception as e:
                st.error("Something went wrong while fetching the news.")
                st.exception(e)

# ---------------- DISPLAY RESULTS ----------------
if st.session_state.results:

    results = st.session_state.results

    if not results:
        st.error("No articles found. Try a different topic.")
    else:
        st.success(f"Found {len(results)} articles!")

        for i, r in enumerate(results, 1):
            with st.expander(
                f"{i}. {r['title']} ({r['source']}, {r['date']})"
            ):
                st.markdown(f"**Summary**  \n{r['summary']}")
                st.markdown(f"**Why it matters**  \n{r['why_it_matters']}")
                st.markdown(f"[🔗 Read Full Article]({r['url']})")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Hugging Face, and NewsAPI")
st.caption(f"© {datetime.now().year} NewsBrief AI — Personal Project by Arya M A")
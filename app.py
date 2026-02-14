import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI, OpenAIError

APP_TITLE = "Market Research Assistant"
DEFAULT_LLM = "gpt-5.2-pro"


def clean_industry(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

# ------------------- ADD: industry validation helpers -------------------
INDUSTRY_HINTS = [
    "industry", "sector", "market", "markets",
    "banking", "finance", "financial",
    "manufacturing", "services", "business", "trade",
    "retail", "supply", "economy", "economic",
    "companies", "corporation", "firm", "firms"
]

BAD_INPUTS = {"hi", "hello", "hey", "yo", "sup", "test", "testing", "ok", "okay", "thanks"}

def build_industry_query(industry: str) -> str:
    lower = industry.lower()
    if any(w in lower for w in ["industry", "sector", "market"]):
        return industry
    return f"{industry} industry"

def looks_like_industry_from_results(user_input: str, docs) -> bool:
    """
    Accept if results look industry/sector-related.
    """
    titles = [(d.metadata.get("title") or "").lower() for d in docs]

    # Strong signal: titles include industry hints
    if any(any(h in t for h in INDUSTRY_HINTS) for t in titles):
        return True

    # Medium signal: user input appears in at least 2 titles
    q = (user_input or "").lower()
    q_tokens = [tok for tok in re.split(r"\W+", q) if tok]
    if q_tokens:
        hits = sum(1 for t in titles if any(tok in t for tok in q_tokens))
        if hits >= 2:
            return True

    return False
# ------------------------------------------------------------------------


def get_wikipedia_pages(industry: str, k: int = 5):
    retriever = WikipediaRetriever(top_k_results=k, doc_content_chars_max=2000)
    docs = retriever.invoke(industry)
    return docs[:k]


def doc_to_url(doc) -> str:
    title = doc.metadata.get("title") or ""
    if not title:
        return ""
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def generate_report(industry: str, docs, model: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    sources = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", f"Source {i}")
        content = doc.page_content
        sources.append(f"Source {i}: {title}\n{content}")
    sources_text = "\n\n".join(sources)

    system = (
        "You are a market research assistant. "
        "Write concise, factual industry reports for business analysts."
    )
    user = (
        f"Industry: {industry}\n\n"
        "Using ONLY the information from the sources below, write an industry report under 500 words.\n"
        "Format as plain text with exactly 6 sections. Each section must be on its own line and separated by a blank line.\n"
        "Use this exact structure and headings:\n"
        "Market Definition:\n"
        "Major Segments:\n"
        "Key Players/Types:\n"
        "Value Chain:\n"
        "Demand Drivers:\n"
        "Risks/Trends:\n"
        "Do NOT use markdown symbols or bullets.\n\n"
        f"Sources:\n{sources_text}"
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    text = resp.output_text.strip()
    words = text.split()
    if len(words) > 500:
        text = " ".join(words[:500]) + "..."
    return text


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    llm_choice = st.selectbox("LLM", options=[DEFAULT_LLM])
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        st.error("Missing OPENAI_API_KEY. Set it in Streamlit Secrets or .streamlit/secrets.toml")
        st.stop()

st.markdown("Step 1: Enter an industry")
industry_input = st.text_input("Industry", placeholder="e.g., Electric vehicles")
industry = clean_industry(industry_input)

if "wiki_docs" not in st.session_state:
    st.session_state.wiki_docs = []

if st.button("Find Wikipedia pages"):
    lower = industry.lower()

    # Basic invalid inputs
    if (not industry) or (lower in BAD_INPUTS) or (not re.search(r"[A-Za-z]", industry)):
        st.warning("Please enter a valid industry (e.g., 'fast fashion', 'electric vehicles', 'retail banking').")
        st.session_state.wiki_docs = []
    else:
        query = build_industry_query(industry)
        docs = get_wikipedia_pages(query, k=5)

        # Reject if results don't look like an industry
        if (not docs) or (not looks_like_industry_from_results(industry, docs)):
            st.warning("Please enter a valid industry (e.g., 'fast fashion', 'electric vehicles', 'retail banking').")
            st.session_state.wiki_docs = []
        else:
            st.session_state.wiki_docs = docs


if st.session_state.wiki_docs:
    st.markdown("Step 2: Top 5 Wikipedia pages")
    urls = [doc_to_url(doc) for doc in st.session_state.wiki_docs]
    for url in urls:
        if url:
            st.write(url)

  st.markdown("Step 3: Generate industry report")
if st.button("Generate report"):
    if (not api_key) or (not api_key.startswith("sk-")):
        st.warning("API key ไม่ถูกต้องหรือยังไม่ได้ตั้งค่า (ควรขึ้นต้นด้วย sk-)")
        st.stop()

    if not llm_choice:
        st.warning("Please set the LLM model name in the sidebar.")
        st.stop()

    with st.spinner("Generating report..."):
        try:
            report = generate_report(industry, st.session_state.wiki_docs, llm_choice, api_key)
            st.markdown("**Industry report**")
            st.text(report)
        except OpenAIError as e:
            st.error(f"LLM request failed: {e}")

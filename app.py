import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI, OpenAIError

APP_TITLE = "Market Research Assistant"
DEFAULT_LLM = "gpt-5.2-pro"  # keep as you had it; change to your available model if needed

# Default ranking behavior (can be overridden from sidebar)
DEFAULT_K_CANDIDATES = 25
DEFAULT_MIN_RELEVANCE_SCORE = 0.35


def clean_industry(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


# ------------------- Industry query + validation -------------------
BAD_INPUTS = {"hi", "hello", "hey", "yo", "sup", "test", "testing", "ok", "okay", "thanks"}

def build_industry_query(industry: str) -> str:
    lower = (industry or "").lower()
    if any(w in lower for w in ["industry", "sector", "market"]):
        return industry
    return f"{industry} industry"
# --------------------------------------------------------------


# ------------------- Wiki filtering & ranking -------------------
INDUSTRY_HINTS = [
    "industry", "sector", "market", "markets", "value chain", "supply chain",
    "manufacturing", "services", "retail", "wholesale", "distribution",
    "revenue", "sales", "demand", "customers", "suppliers", "competition",
    "companies", "company", "firms", "business", "economy", "economic",
]

NOISE_TITLE_HINTS = [
    "disambiguation", "may refer to", "list of", "outline of", "glossary of",
    "index of", "template:", "wikipedia:", "category:"
]

def tokenize(text: str):
    # simple unicode-friendly tokenization
    text = (text or "").lower()
    toks = re.split(r"[^\w]+", text, flags=re.UNICODE)
    return [t for t in toks if t and len(t) > 1]

def is_noise_page(title: str, content: str) -> bool:
    t = (title or "").lower()
    c = (content or "").lower()

    if any(h in t for h in NOISE_TITLE_HINTS):
        return True
    if "may refer to" in c and "disambiguation" in c:
        return True

    return False

def overlap_score(query_tokens, text_tokens) -> float:
    if not query_tokens:
        return 0.0
    qset = set(query_tokens)
    tset = set(text_tokens)
    inter = len(qset & tset)
    return inter / max(1, len(qset))

def phrase_match_boost(query: str, title: str, content: str) -> float:
    q = (query or "").strip().lower()
    if not q:
        return 0.0
    t = (title or "").lower()
    c = (content or "").lower()

    boost = 0.0
    if q in t:
        boost += 1.5
    if q in c:
        boost += 0.7
    return boost

def industry_hint_boost(title: str, content: str) -> float:
    t = (title or "").lower()
    c = (content or "").lower()

    boost = 0.0
    boost += 0.4 * sum(1 for h in INDUSTRY_HINTS if h in t)
    boost += 0.1 * sum(1 for h in INDUSTRY_HINTS if h in c)
    return boost

def relevance_score(industry: str, doc) -> float:
    title = doc.metadata.get("title") or ""
    content = doc.page_content or ""

    if is_noise_page(title, content):
        return -1.0

    q_tokens = tokenize(industry)
    t_tokens = tokenize(title)
    c_tokens = tokenize(content)

    # Title is weighted heavier than content
    score = 0.0
    score += 2.2 * overlap_score(q_tokens, t_tokens)
    score += 1.0 * overlap_score(q_tokens, c_tokens)

    # Exact-phrase boost
    score += phrase_match_boost(industry, title, content)

    # Industry-context boost (helps avoid random pages with same keyword but not market/sector context)
    score += industry_hint_boost(title, content)

    # Light penalty for broad list pages
    lt = title.lower()
    if lt.startswith("list of ") or lt.startswith("outline of "):
        score -= 0.3

    return score

def filter_and_rank_docs(industry: str, docs, top_n: int = 5, min_score: float = 0.35):
    scored = []
    for d in docs:
        s = relevance_score(industry, d)
        if s >= 0:  # drop noise (-1)
            scored.append((s, d))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Prefer docs above threshold; fallback to best available if none meet threshold
    strong = [(s, d) for (s, d) in scored if s >= min_score]
    chosen = strong[:top_n] if strong else scored[:top_n]

    return [d for (s, d) in chosen], chosen  # docs, plus scored pairs for debug
# --------------------------------------------------------------


def get_wikipedia_pages(query: str, k_candidates: int = 25):
    retriever = WikipediaRetriever(top_k_results=k_candidates, doc_content_chars_max=2000)
    docs = retriever.invoke(query)
    return docs[:k_candidates]


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


# ------------------- Streamlit UI -------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    llm_choice = st.selectbox("LLM", options=[DEFAULT_LLM])
    api_key = st.text_input("API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

    st.divider()
    st.subheader("Wikipedia ranking")
    k_candidates = st.slider("Candidate pages to fetch", min_value=10, max_value=60, value=DEFAULT_K_CANDIDATES, step=5)
    min_score = st.slider("Minimum relevance threshold", min_value=0.0, max_value=1.5, value=float(DEFAULT_MIN_RELEVANCE_SCORE), step=0.05)
    show_debug = st.checkbox("Show debug scores", value=False)

st.markdown("Step 1: Enter an industry")
industry_input = st.text_input("Industry", placeholder="e.g., Electric vehicles")
industry = clean_industry(industry_input)

if "wiki_docs" not in st.session_state:
    st.session_state.wiki_docs = []
if "wiki_scored" not in st.session_state:
    st.session_state.wiki_scored = []  # store scored pairs for debug


if st.button("Find Wikipedia pages"):
    lower = industry.lower()

    # Basic invalid inputs (unicode-friendly: require at least one word char)
    if (not industry) or (lower in BAD_INPUTS) or (not re.search(r"\w", industry, flags=re.UNICODE)):
        st.warning("Please enter a valid industry (e.g., 'fast fashion', 'electric vehicles', 'retail banking').")
        st.session_state.wiki_docs = []
        st.session_state.wiki_scored = []
    else:
        query = build_industry_query(industry)

        try:
            with st.spinner("Searching Wikipedia..."):
                candidates = get_wikipedia_pages(query, k_candidates=int(k_candidates))
                top_docs, scored_pairs = filter_and_rank_docs(
                    industry=industry,
                    docs=candidates,
                    top_n=5,
                    min_score=float(min_score),
                )
        except Exception as e:
            st.warning(f"Wikipedia search failed. Try again or adjust the query. Error: {e}")
            top_docs, scored_pairs = [], []

        if not top_docs:
            st.warning("Couldn't find relevant Wikipedia pages for that industry. Try a more specific industry name.")
            st.session_state.wiki_docs = []
            st.session_state.wiki_scored = []
        else:
            st.session_state.wiki_docs = top_docs
            st.session_state.wiki_scored = scored_pairs


if st.session_state.wiki_docs:
    st.markdown("Step 2: Top 5 Wikipedia pages (ranked by relevance)")
    for idx, doc in enumerate(st.session_state.wiki_docs, start=1):
        title = doc.metadata.get("title", f"Result {idx}")
        url = doc_to_url(doc)
        if url:
            st.write(f"{idx}. {title} — {url}")
        else:
            st.write(f"{idx}. {title}")

    if show_debug and st.session_state.wiki_scored:
        with st.expander("Debug: relevance scores (higher = more relevant)"):
            for score, doc in st.session_state.wiki_scored:
                title = doc.metadata.get("title", "")
                st.write(f"{score:.2f}  -  {title}")

    st.markdown("Step 3: Generate industry report")
    if st.button("Generate report"):
        if not api_key:
            st.warning("Please enter an API key to generate the report.")
        elif not llm_choice:
            st.warning("Please set the LLM model name in the sidebar.")
        else:
            with st.spinner("Generating report..."):
                try:
                    report = generate_report(industry, st.session_state.wiki_docs, llm_choice, api_key)
                    st.markdown("**Industry report**")
                    st.text(report)
                except OpenAIError as e:
                    st.error(f"LLM request failed: {e}")

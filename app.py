import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI, OpenAIError

# =========================
# <<< EDIT HERE >>>
# Choose your final model name (only ONE model in dropdown for submission)
APP_TITLE = "Market Research Assistant"        # <<< EDIT HERE (optional) >>>
DEFAULT_LLM = "gpt-5.2-pro"                    # <<< EDIT HERE (important) >>>
# =========================


def clean_industry(text: str) -> str:
    # Cleans whitespace so "   " becomes "" (empty) and fails validation for Q1
    return re.sub(r"\s+", " ", (text or "").strip())


def get_wikipedia_pages(industry: str, k: int = 5):
    # Retrieves top k Wikipedia pages for the industry
    retriever = WikipediaRetriever(top_k_results=k, doc_content_chars_max=2000)
    docs = retriever.invoke(industry)
    return docs[:k]


def doc_to_url(doc) -> str:
    # Converts a retrieved doc into a Wikipedia URL using its title metadata
    title = doc.metadata.get("title") or ""
    if not title:
        return ""
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def generate_report(industry: str, docs, model: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    # Build a "sources" block from Wikipedia content
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

    # Safety cap in case the model exceeds 500 words
    words = text.split()
    if len(words) > 500:
        text = " ".join(words[:500]) + "..."
    return text


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")

    # Q0: dropdown with ONLY ONE option in final version
    llm_choice = st.selectbox("LLM", options=[DEFAULT_LLM])  # <<< EDIT HERE if you change DEFAULT_LLM >>>

    # Q0: API key field
    api_key = st.text_input(
        "API key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )


# Step 1 (Q1): user enters industry
st.markdown("Step 1: Enter an industry")
industry_input = st.text_input("Industry", placeholder="e.g., Electric vehicles")
industry = clean_industry(industry_input)

if "wiki_docs" not in st.session_state:
    st.session_state.wiki_docs = []


# Step 2 (Q2): retrieve top 5 wiki pages
if st.button("Find Wikipedia pages"):
    # Q1: check industry is provided
    if not industry:
        st.warning("Please enter an industry to continue.")  # <<< EDIT HERE (optional message tweak) >>>
    else:
        st.session_state.wiki_docs = get_wikipedia_pages(industry, k=5)  # <<< EDIT HERE (optional explicit k=5) >>>


# Display results and enable Step 3 if docs exist
if st.session_state.wiki_docs:
    st.markdown("Step 2: Top 5 Wikipedia pages")
    urls = [doc_to_url(doc) for doc in st.session_state.wiki_docs]
    for url in urls:
        if url:
            st.write(url)

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

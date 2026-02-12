import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI

APP_TITLE = "Market Research Assistant"
DEFAULT_LLM = "gpt-5.2-pro"  # Placeholder; update as needed.


def clean_industry(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


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
        "Using ONLY the information from the sources below, write an industry report "
        "under 500 words. Cover: market definition, major segments, key players/types, "
        "value chain, demand drivers, and risks/trends. If information is missing, say so.\n\n"
        f"Sources:\n{sources_text}"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    text = resp.choices[0].message.content.strip()

    words = text.split()
    if len(words) > 500:
        text = " ".join(words[:500]) + "..."
    return text


st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    llm_choice = st.selectbox("LLM", options=[DEFAULT_LLM])
    api_key = st.text_input("API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))

st.markdown("Step 1: Enter an industry")
industry_input = st.text_input("Industry", placeholder="e.g., Electric vehicles")

industry = clean_industry(industry_input)

if "wiki_docs" not in st.session_state:
    st.session_state.wiki_docs = []

if st.button("Find Wikipedia pages"):
    if not industry:
        st.warning("Please enter an industry to continue.")
    else:
        st.session_state.wiki_docs = get_wikipedia_pages(industry)

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
        else:
            with st.spinner("Generating report..."):
                report = generate_report(industry, st.session_state.wiki_docs, llm_choice, api_key)
            st.subheader("Industry report")
            st.write(report)

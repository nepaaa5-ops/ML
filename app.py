import os
from typing import List

import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document


st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="wide",
)

st.title("Market Research Assistant")
st.caption("Generate a concise industry report based on Wikipedia sources.")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("LLM model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    st.divider()
    st.markdown("**Environment**")
    st.code("OPENAI_API_KEY", language="text")


def get_wikipedia_docs(industry: str, k: int = 5) -> List[Document]:
    retriever = WikipediaRetriever(top_k_results=k, lang="en")
    return retriever.get_relevant_documents(industry)


def extract_urls(docs: List[Document]) -> List[str]:
    urls = []
    for d in docs:
        url = d.metadata.get("source") or d.metadata.get("url")
        if url and url not in urls:
            urls.append(url)
    return urls


def build_context(docs: List[Document], max_chars: int = 1200) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        title = d.metadata.get("title", f"Source {i}")
        content = d.page_content.strip().replace("\n", " ")
        blocks.append(f"[{i}] {title}: {content[:max_chars]}")
    return "\n\n".join(blocks)


def count_words(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def generate_report(industry: str, docs: List[Document], model: str, temp: float) -> str:
    context = build_context(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a market research assistant for corporate business analysts. "
                "Use only the provided Wikipedia excerpts. Be precise and professional.",
            ),
            (
                "human",
                "Industry: {industry}\n\n"
                "Sources:\n{context}\n\n"
                "Write an industry report under 500 words with these sections: "
                "Market overview, Value chain, Key segments, Demand drivers, Risks, Major players, Recent trends. "
                "If any section lacks evidence from sources, say so briefly in that section.",
            ),
        ]
    )

    llm = ChatOpenAI(model=model, temperature=temp)
    report = llm.invoke(prompt.format_messages(industry=industry, context=context)).content

    if count_words(report) > 500:
        tighten_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a concise editor. Reduce word count without losing factual accuracy.",
                ),
                (
                    "human",
                    "Shorten the report to under 500 words. Preserve the section headings.\n\n{report}",
                ),
            ]
        )
        report = llm.invoke(tighten_prompt.format_messages(report=report)).content

    return report


industry = st.text_input("Enter an industry (e.g., renewable energy, fintech, automotive)")
run = st.button("Generate Report", type="primary")

if run:
    if not industry.strip():
        st.warning("Please provide an industry to continue.")
        st.stop()

    with st.spinner("Retrieving Wikipedia sources..."):
        try:
            docs = get_wikipedia_docs(industry, k=5)
        except Exception as exc:
            st.error(f"Failed to retrieve Wikipedia sources: {exc}")
            st.stop()

    if not docs:
        st.error("No Wikipedia pages found for that industry. Try a different term.")
        st.stop()

    urls = extract_urls(docs)[:5]

    st.subheader("Step 2: Top Wikipedia Pages")
    if urls:
        for url in urls:
            st.write(url)
    else:
        st.info("No URLs found in Wikipedia metadata, but sources were retrieved.")

    st.subheader("Step 3: Industry Report")
    with st.spinner("Generating report..."):
        try:
            report = generate_report(industry, docs[:5], model_name, temperature)
        except Exception as exc:
            st.error(
                "Failed to generate report. Ensure OPENAI_API_KEY is set and the model name is valid. "
                f"Details: {exc}"
            )
            st.stop()

    st.write(report)
    st.caption(f"Word count: {count_words(report)}")

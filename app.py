import json
import re
from dataclasses import dataclass
from typing import List, Dict

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from openai import OpenAI


st.set_page_config(
    page_title="ResearchGraph AI",
    page_icon="🧠",
    layout="wide",
)


@dataclass
class Entity:
    name: str
    entity_type: str
    description: str


@dataclass
class Relationship:
    source: str
    target: str
    relation_type: str
    description: str


def clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def extract_graph_from_document(client: OpenAI, document: str) -> Dict:
    prompt = f"""
You are ResearchGraph AI.

Extract a lightweight knowledge graph from the following document.

Return ONLY valid JSON with this structure:

{{
  "summary": "short summary of the document",
  "entities": [
    {{
      "name": "entity name",
      "type": "PERSON | ORGANIZATION | TECHNOLOGY | CONCEPT | EVENT | LOCATION | METHOD",
      "description": "short description"
    }}
  ],
  "relationships": [
    {{
      "source": "source entity",
      "target": "target entity",
      "type": "relationship type",
      "description": "short relationship explanation"
    }}
  ],
  "key_claims": [
    {{
      "claim": "important claim",
      "source_text": "exact supporting sentence or short passage from the document"
    }}
  ]
}}

Document:
{document}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    return json.loads(clean_json(content))


def answer_question_with_citations(client: OpenAI, question: str, graph_data: Dict, document: str) -> Dict:
    prompt = f"""
You are ResearchGraph AI.

Answer the user's question using the extracted knowledge graph and the original document.

You must provide:
- a clear answer
- citations from the document
- a reasoning trace
- related entities used to answer

Return ONLY valid JSON with this structure:

{{
  "answer": "answer with citation markers like [1], [2]",
  "citations": [
    {{
      "id": 1,
      "source_text": "exact supporting text from the document",
      "explanation": "why this citation supports the answer"
    }}
  ],
  "reasoning_trace": [
    "step 1",
    "step 2",
    "step 3"
  ],
  "related_entities": ["entity 1", "entity 2"]
}}

Question:
{question}

Knowledge Graph:
{json.dumps(graph_data, ensure_ascii=False, indent=2)}

Original Document:
{document}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    return json.loads(clean_json(content))


def build_networkx_graph(graph_data: Dict) -> nx.DiGraph:
    graph = nx.DiGraph()

    for entity in graph_data.get("entities", []):
        graph.add_node(
            entity["name"],
            label=entity["name"],
            entity_type=entity.get("type", "UNKNOWN"),
        )

    for rel in graph_data.get("relationships", []):
        source = rel.get("source")
        target = rel.get("target")
        rel_type = rel.get("type", "RELATED_TO")

        if source and target:
            graph.add_edge(source, target, label=rel_type)

    return graph


def display_graph(graph_data: Dict):
    graph = build_networkx_graph(graph_data)

    if graph.number_of_nodes() == 0:
        st.info("No graph to display yet.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    pos = nx.spring_layout(graph, seed=42, k=0.8)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=2500,
        font_size=9,
        arrows=True,
        ax=ax,
    )

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        ax=ax,
    )

    ax.set_axis_off()
    st.pyplot(fig)


def display_extracted_graph(graph_data: Dict):
    st.subheader("📌 Document Summary")
    st.write(graph_data.get("summary", ""))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧩 Extracted Entities")
        entities = graph_data.get("entities", [])
        if entities:
            for entity in entities:
                st.markdown(
                    f"**{entity.get('name', '')}** "
                    f"({entity.get('type', '')}) — {entity.get('description', '')}"
                )
        else:
            st.info("No entities extracted.")

    with col2:
        st.subheader("🔗 Extracted Relationships")
        relationships = graph_data.get("relationships", [])
        if relationships:
            for rel in relationships:
                st.markdown(
                    f"**{rel.get('source', '')}** → "
                    f"*{rel.get('type', '')}* → "
                    f"**{rel.get('target', '')}**"
                )
                st.caption(rel.get("description", ""))
        else:
            st.info("No relationships extracted.")

    st.subheader("🧠 Knowledge Graph Visualization")
    display_graph(graph_data)

    st.subheader("📚 Key Claims with Source Text")
    for i, claim in enumerate(graph_data.get("key_claims", []), start=1):
        with st.expander(f"Claim {i}: {claim.get('claim', '')}"):
            st.write(claim.get("source_text", ""))


def display_answer(answer_data: Dict):
    st.subheader("💬 Answer")
    st.markdown(answer_data.get("answer", ""))

    st.subheader("📚 Verifiable Citations")
    citations = answer_data.get("citations", [])
    if citations:
        for citation in citations:
            with st.expander(f"Citation [{citation.get('id', '')}]"):
                st.write("**Source text:**")
                st.write(citation.get("source_text", ""))
                st.write("**Why it supports the answer:**")
                st.write(citation.get("explanation", ""))
    else:
        st.info("No citations returned.")

    st.subheader("🧭 Reasoning Trace")
    for step in answer_data.get("reasoning_trace", []):
        st.write(f"- {step}")

    st.subheader("🔎 Related Entities")
    st.write(", ".join(answer_data.get("related_entities", [])))


def main():
    st.title("🧠 ResearchGraph AI")
    st.write(
        "A lightweight Knowledge Graph RAG assistant that extracts entities, "
        "relationships and verifiable citations from research documents."
    )

    if "document" not in st.session_state:
        st.session_state.document = ""
    if "graph_data" not in st.session_state:
        st.session_state.graph_data = None
    if "answer_data" not in st.session_state:
        st.session_state.answer_data = None

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        st.markdown("---")
        st.write("Built with Streamlit, OpenAI, NetworkX and Matplotlib.")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    client = get_client(api_key)

    tab1, tab2, tab3 = st.tabs(
        ["📄 Add Document", "❓ Ask Questions", "🧠 Graph Explorer"]
    )

    with tab1:
        st.header("1. Add a research document")

        sample_text = """
GraphRAG is a retrieval-augmented generation approach that combines language models
with knowledge graphs. It was introduced by Microsoft Research to improve reasoning
over large document collections. Unlike traditional vector-based RAG, GraphRAG builds
a graph of entities and relationships, allowing the system to answer complex questions
that require connecting information across multiple parts of a document.
"""

        document = st.text_area(
            "Paste a research paper excerpt, technical document or article",
            value=st.session_state.document or sample_text,
            height=300,
        )

        if st.button("Extract Knowledge Graph"):
            if not document.strip():
                st.error("Please paste a document first.")
                return

            with st.spinner("Extracting entities, relationships and claims..."):
                try:
                    st.session_state.document = document
                    st.session_state.graph_data = extract_graph_from_document(
                        client, document
                    )
                    st.session_state.answer_data = None
                    st.success("Knowledge graph extracted successfully.")
                except json.JSONDecodeError:
                    st.error("The AI response was not valid JSON. Please try again.")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.graph_data:
            display_extracted_graph(st.session_state.graph_data)

    with tab2:
        st.header("2. Ask questions with citations")

        if not st.session_state.graph_data:
            st.warning("Please extract a knowledge graph first.")
            return

        question = st.text_input(
            "Ask a question about the document",
            value="What is GraphRAG and why is it different from traditional RAG?",
        )

        if st.button("Ask ResearchGraph AI"):
            if not question.strip():
                st.error("Please enter a question.")
                return

            with st.spinner("Reasoning over the knowledge graph..."):
                try:
                    st.session_state.answer_data = answer_question_with_citations(
                        client,
                        question,
                        st.session_state.graph_data,
                        st.session_state.document,
                    )
                    st.success("Answer generated.")
                except json.JSONDecodeError:
                    st.error("The AI response was not valid JSON. Please try again.")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.answer_data:
            display_answer(st.session_state.answer_data)

    with tab3:
        st.header("3. Graph Explorer")

        if not st.session_state.graph_data:
            st.warning("Please extract a knowledge graph first.")
            return

        display_graph(st.session_state.graph_data)

        st.subheader("Raw Knowledge Graph JSON")
        st.json(st.session_state.graph_data)


if __name__ == "__main__":
    main()

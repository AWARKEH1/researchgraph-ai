# 🧠 ResearchGraph AI

AI-powered Knowledge Graph RAG system with explainable reasoning and verifiable citations.

ResearchGraph AI transforms unstructured documents into a lightweight knowledge graph, extracts entities and relationships, and answers user questions with source-backed citations.

---

## 🚀 Features

- 📄 Document ingestion
- 🧩 Automatic entity extraction
- 🔗 Relationship discovery
- 🧠 Knowledge Graph generation
- 📚 Verifiable citations
- 🔍 Explainable reasoning trace
- 💬 Question answering over documents
- 🌐 Interactive Streamlit interface

---

## 🏗 Architecture

```text
Document
   │
   ▼
Entity & Relationship Extraction
   │
   ▼
Knowledge Graph Construction
   │
   ▼
Question Answering Engine
   │
   ▼
Answer + Citations + Reasoning Trace
```

---

## 📸 Application Workflow

### 1. Upload or Paste a Document

The system analyzes technical documents, research papers, articles, and reports.

### 2. Extract a Knowledge Graph

ResearchGraph AI automatically identifies:

- People
- Organizations
- Technologies
- Concepts
- Methods
- Events

and creates relationships between them.

### 3. Ask Questions

Example:

```text
What is GraphRAG and why is it different from traditional RAG?
```

### 4. Receive Verified Answers

The system returns:

- Answer
- Supporting citations
- Reasoning path
- Related entities

---

## 🛠 Tech Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Graph Processing | NetworkX |
| Visualization | Matplotlib |
| Language | Python |

---

## 📂 Project Structure

```text
researchgraph-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/AWARKEH1/researchgraph-ai.git
cd researchgraph-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔑 Configuration

Create an OpenAI API Key:

https://platform.openai.com/api-keys

Enter the key in the Streamlit sidebar.

---

## 💡 Example Use Cases

### Research Assistant

Analyze research papers and retrieve source-backed answers.

### Technical Documentation Search

Explore internal documentation through a graph-based interface.

### Knowledge Discovery

Identify hidden relationships between concepts and technologies.

### AI Engineering

Experiment with explainable Retrieval-Augmented Generation systems.

---

## 🎯 Future Improvements

- PDF Upload Support
- Multi-document Graphs
- Neo4j Integration
- Graph Analytics Dashboard
- Local LLM Support (Ollama)
- Hybrid Search
- Graph Persistence

---

## 📄 License

MIT License

---

## 👨‍💻 Author

Abbas Awarkeh

AI Engineer | Data Scientist | Machine Learning Enthusiast

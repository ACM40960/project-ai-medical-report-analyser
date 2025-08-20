# 🧠 Medical Report Analyser (RAG-Powered)

**Streamlit • LangChain • Pinecone • HuggingFace Embeddings • Google Gemini • DuckDuckGo fallback • Session Metrics**

A Retrieval-Augmented Generation (RAG) assistant for analysing patient medical reports. Upload patient PDFs/TXTs and an optional helpbook, and the system retrieves the most relevant chunks to generate **source-grounded, transparent summaries and answers**.  
Built with LangChain, Pinecone, HuggingFace embeddings, and Gemini (via Google GenAI API).  

---

## 🚀 Live Features
- 📂 Upload patient PDF/TXT files and helpbook references
- 🔎 Similarity search over Pinecone (per-session patient index + persistent helpbook index)
- 🧠 Gemini-powered RAG answering with inline `[patient]` / `[helpbook]` citations
- 🌐 DuckDuckGo web search fallback for up-to-date information
- 📝 Automatic summaries of patient reports and lab test interpretation
- 📊 Metrics logging (faithfulness, helpfulness, latency, hallucination rate)
- 🔐 .env-based config for API keys and Pinecone setup
- 🖥️ Clean Streamlit chat UI with session memory and index reset

---

## 📂 Project Structure
```
medical-rag-assistant/
├── app.py              # Streamlit entrypoint (upload, chat, purge sessions)
├── agent.py            # LangChain agent assembly with tools
├── embeddings.py       # HuggingFace MiniLM embeddings
├── ingest.py           # Ingestion of patient/helpbook docs → Pinecone
├── llm.py              # Gemini LLM wrapper
├── metrics.py          # Session summarisation & CSV logging
├── rag.py              # Dual-retriever pipeline (patient + helpbook)
├── rag_tools.py        # Wrappers for summarisation & lab interpretation
├── vectorstore.py      # Pinecone vectorstore setup & management
├── web_tools.py        # DuckDuckGo search fallback tools
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Pinecone account + API key
- Google API key (for Gemini)
- (Optional) Docker for deployment
- Curated patient reports and helpbook PDFs

### Clone repo & setup environment
```bash
git clone <your-repo-url>
cd medical-rag-assistant
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

---

## 🔑 Configuration

Create a `.env` file in the project root:
```ini
# Pinecone
PINECONE_API_KEY=****
PINECONE_REGION=us-east-1
GENERAL_INDEX_NAME=medical-helpbook
PATIENT_INDEX_NAME=patient-reports

# Gemini (Google GenAI)
GOOGLE_API_KEY=****

# Streamlit
FLASK_ENV=development
```

---

## ▶️ Usage

### Run locally
```bash
streamlit run app.py
```
App will be available at `http://localhost:8501`.

### Workflow
1. Upload **patient report(s)** in sidebar → embed into Pinecone (session-specific).
2. Optionally upload a **helpbook PDF** → embedded once, persistent across sessions.
3. Chat with the assistant:
   - `RAG_QA`: answer based on patient + helpbook docs
   - `Summarise_Patient_Report`: concise overview of lab values & next steps
   - `Interpret_Lab`: explain one lab test with ranges/status
   - `WEB_SEARCH_*`: DuckDuckGo tools if info not in KB
4. Reset session: purge patient index, rotate session ID, metrics logged to `session_metrics.csv`.

---

## 🧩 RAG Pipeline

**Offline ingestion**
- Split PDFs/TXTs into chunks (1000 chars, 150 overlap)
- Embed with HuggingFace MiniLM (384-dim)
- Upsert into Pinecone (`general_index`, `patient_index`)

**Online inference**
- Query embedded → retrieve top docs (MMR search)
- Construct context with `[patient]` and `[helpbook]` tags
- Generate structured, citation-rich answer with Gemini
- If no relevant docs → fallback to DuckDuckGo search

---

## 📊 Evaluation & Metrics
Each turn logs:
- Faithfulness / helpfulness (auto-evaluated by LLM judge)
- Latencies (retrieval, LLM, total)
- Docs retrieved (patient/helpbook counts)
- Grounding rates & hallucination rate
- Answer/context lengths

At session end:
- Summary appended to `session_metrics.csv`

---

## 🔐 Safety & Compliance
- ✅ Transparent citations `[patient]`, `[helpbook]`, `[web]`
- ✅ No diagnostic claims (educational use only)
- ✅ Session-specific patient index purged on reset
- ✅ GDPR/HIPAA principles respected (no personal data persisted)

---

## 🌐 Deployment
- **Local**: Streamlit (`streamlit run app.py`)
- **Cloud (recommended)**:
  - Dockerize the app
  - Deploy to AWS EC2
  - Restrict inbound ports & rotate API keys
  - Use GitHub Actions CI/CD for auto-deployment

---

## 📌 Future Work
- Multilingual support & voice interaction
- Physician co-pilot integrations
- More robust retrieval evaluation dashboard
- Active learning: improve embeddings with user feedback

---

## 🧑‍💻 Authors
- **Sushmitha B (24209228)**
- **Kritheshwar (24233914)**

---

## 📜 License
MIT License. See [LICENSE](LICENSE) for details.

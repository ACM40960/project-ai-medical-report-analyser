# 🧠 AI Medical Report Analyser – RAG-powered Clinical Assistant  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![LangChain](https://img.shields.io/badge/Framework-LangChain-green)](https://www.langchain.com/)  
[![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-orange)](https://www.pinecone.io/)  
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini--1.5--Flash-red)](https://ai.google.dev/gemini-api)  
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-pink)](https://streamlit.io/)  
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)  

---

## 📖 Abstract  

The **AI Medical Report Analyser** is a Retrieval-Augmented Generation (RAG) chatbot designed to **interpret patient medical reports**.  
- It embeds **helper documents** (reference handbook) and **patient PDFs/TXTs** into a Pinecone vector database.  
- Queries are answered using **Google Gemini** via LangChain, grounded in the retrieved evidence.  
- If knowledge is missing, the app falls back to **DuckDuckGo search**.  
- A **Streamlit UI** provides easy uploads, chat, and automatic metrics tracking.  

⚠️ **Note**: This application is **educational only** — it does not provide medical diagnoses.  

---

## 🎯 System Goals  

- ✅ Grounded answers with citations  
- ✅ Minimise hallucinations with helper docs + patient data  
- ✅ Transparent provenance: `[patient]`, `[helpbook]`, `[web]`  
- ✅ Private: patient reports stored per session only  

---

## ✨ Features  

- 📘 Upload and embed **Helper Docs** (medical reference handbooks in `/helperDocs`)  
- 📑 Upload **Patient Reports** (PDF/TXT files) for session-based analysis  
- 🧠 RAG-powered chat with **Gemini** (faithfulness + helpfulness scoring)  
- 📝 Specialised tools:  
  - **Summarise Patient Report**  
  - **Interpret Lab Test**  
- 🌐 **DuckDuckGo fallback** for up-to-date external info  
- 📊 Evaluation metrics (faithfulness, helpfulness, latency, hallucination rate)  
- 🖥️ Hosted entirely in **Streamlit**  

---

## 📂 Project Structure  

```
project-ai-medical-report-analyser/
├── app.py             # Streamlit app (UI + workflow)
├── agent.py           # LangChain agent & tools
├── rag.py             # RAG pipeline (retrieval + prompt + memory)
├── rag_tools.py       # Summarise + lab interpretation tools
├── ingest.py          # Load & chunk PDFs/TXTs
├── embeddings.py      # HuggingFace MiniLM embeddings
├── vectorstore.py     # Pinecone vectorstore setup
├── web_tools.py       # DuckDuckGo fallback tools
├── metrics.py         # Evaluation + CSV logging
├── helperDocs/        # 📘 Place helper docs here
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation  

### Prerequisites  
- Python 3.10+  
- Pinecone API key  
- Google API key (Gemini)  

### Setup  
```bash
git clone https://github.com/ACM40960/project-ai-medical-report-analyser.git
cd project-ai-medical-report-analyser

# create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate.ps1  # Windows PowerShell

# install dependencies
pip install -r requirements.txt
```

### Configure `.env`  
```ini
PINECONE_API_KEY=...
PINECONE_REGION=us-east-1
GENERAL_INDEX_NAME=medical-helpbook
PATIENT_INDEX_NAME=patient-reports

GOOGLE_API_KEY=...
```

---

## ▶️ Usage  

### Workflow  

1️⃣ **Upload Helper Docs**  
- Go to sidebar → *Upload Helpbook*  
- Use the files from `/helperDocs` (provided in repo)  
- Click **Process file** → embedded into Pinecone  

2️⃣ **Upload Patient Reports**  
- Upload patient PDF/TXT files  
- Click **Process File** → embedded for this session only  

3️⃣ **Chat**  
- Ask: *“Summarise the report”*, *“Explain Hemoglobin”*, etc.  
- Responses cite `[patient]`, `[helpbook]`, `[web]`  
- Reset anytime with *Process new report* (purges patient index)  

```bash
streamlit run app.py
```
App runs at: `http://localhost:8501`  

---

## 🔄 Workflow Overview  

> **Important:** On first run, upload and process the **helper docs** in `/helperDocs` (once).  
> Then upload **patient reports** for the current session. Only after that, start chatting.

### Diagram  

![RAG Flow](docs/README_flow.png)

### What happens under the hood
1. **Ingestion:** PDFs/TXTs are parsed (PyPDFLoader/TXT), split into 1000‑char chunks with 150 overlap.  
2. **Embeddings:** Chunks are embedded using **MiniLM‑L6‑v2 (384‑d)**.  
3. **Storage:** Vectors go to Pinecone — a persistent **GENERAL_INDEX** for helpbook and a session‑scoped **PATIENT_INDEX**.  
4. **Retrieval:** Dual MMR retrievers (helpbook k=6, λ=0.2; patient k=10, λ=0.35 with `session_id` filter) fetch relevant context.  
5. **RAG Generation:** A prompt template with **chat memory** feeds context to **Gemini 1.5 Flash**. The answer enforces inline citations `[patient]` / `[helpbook]` / `[web]`.  
6. **Fallback:** If context is insufficient, the agent uses **DuckDuckGo** tools and clearly tags results as `[web]`.  
7. **Metrics:** Each turn logs faithfulness, helpfulness, latency, retrieval counts, grounding %, and hallucination rate to `session_metrics.csv`.  

---

## 📊 Evaluation Metrics  

The app tracks:  
- Faithfulness & helpfulness (auto-scored)  
- Latency (retrieval, LLM, total)  
- Docs retrieved (patient/helpbook)  
- Grounding % & hallucination rate  

📈 Example chart:  
![Metrics](docs/evaluation_metrics.png)  

---

## 🌐 Deployment  

- **Local run:** `streamlit run app.py`  
- **Public hosting:** Deploy via [Streamlit Cloud](https://streamlit.io/cloud) (recommended)  

---

## 📌 Future Work  

- 🌍 Multilingual + voice support  
- 🧑‍⚕️ Clinician co-pilot integration  
- 📈 More analytics dashboards  
- 🔄 User feedback loops for prompt tuning  

---

## 🧑‍💻 Authors  

- Sushmitha B (24209228)  
- Kritheshwar (24233914)  
- *Projects in Maths Modelling*  

---

## 📜 License  

MIT License. See [LICENSE](LICENSE).  

---

⚠️ **Disclaimer:** This tool is for **educational purposes only** and not for clinical use.  

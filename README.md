# AI Medical Report Analyzer  

## 📌 Project Overview  
This project is part of the **ACM40960 – Project in Mathematical Modelling** module.  
The goal is to design an **AI-powered medical report analyzer** that simplifies patient reports using a **Retrieval-Augmented Generation (RAG) pipeline** and a **Large Language Model (LLM)**.  

Patients often struggle to interpret medical reports because of complex terms and unstructured formats.  
Our system processes uploaded reports, retrieves relevant context, and generates simplified explanations to improve patient understanding.  

---

## ⚙️ Technologies Used  
- **Pinecone** → Vector storage for patient reports  
- **Hugging Face Embeddings** → Converts text into embeddings  
- **LangChain** → Manages retrieval flow (report, handbook, or web search)  
- **Google Gemini 1.5 Flash** → Generates simplified answers  
- **Streamlit** → User interface for uploading and querying reports  
- **Fallback Search** → Web/handbook retrieval for missing context  

---

## 🛠️ Methodology (Workflow)  
1. **User Uploads** → Patient report + optional handbook  
2. **Preprocessing** → Read, chunk, and embed text  
3. **Vector Storage** → Save embeddings in Pinecone  
4. **Context Retrieval** → Retrieve relevant chunks (semantic search + MMR)  
5. **Answer Generation** → Gemini 1.5 Flash produces simplified output  
6. **Memory & Follow-ups** → Buffer to maintain conversation history  
7. **Final Answer** → Structured, tagged response  

---

## 📊 Evaluation Metrics  
The system was tested on **3 patient reports** with ~30 sample queries.  

**Report-Level Metrics**  
- Grounding Accuracy: **94%**  
- Retrieval Hit@10: **97%**  
- Source-Tag Accuracy: **95%**  
- Response Clarity: **94%**  
- Safety Compliance: **100%**  

**System-Level Metrics**  
- Faithfulness: **50%**  
- Helpfulness: **100%**  
- Hallucination Rate: **50%**  
- Retrieval Success Rate: **100%**  

---

## 📌 Limitations  
- Small evaluation sample (3 reports, ~30 queries)  
- Works best with structured PDFs/TXT (format sensitivity)  
- Only supports English text at present  

---

## 🚀 Future Work  
- Multilingual support  
- Structured outputs for medical systems  
- Deployment and EHR integration  
- Improved hallucination control  

---

## 👨‍🎓 Authors  
- **Sushmitha B (24209298)**  
- **Kritheshvar KRV (24231949)**  


## 📂 Repository Structure  

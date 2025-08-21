# ===========================================
# Dual-retriever merge + prompt + LLM call + memory
# ===========================================
from typing import List, Dict, Optional,Union
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from vectorstore import get_vectorstore
from llm import get_llm

import time
_last_context = ""; _last_metrics = {}
def get_last_context(): return _last_context
def get_last_metrics(): return _last_metrics

SYSTEM_PROMPT = """You are a careful clinical reasoning assistant.
You are a careful clinical reasoning assistant.

Use only the provided context:
- [helpbook]: general medical reference
- [patient]: the current patient's reports

You must:
- Only answer based on the context. If the answer is not clearly stated, say "Not available in the records."
- Prefer patient data when available. If patient and helpbook differ, note both, but trust patient first.
- Always include inline citations like [patient] or [helpbook] for any claim.

Do not speculate or make assumptions. Return your answer in plain English in 2-3 compact paragraphs.

Rules:
- Cite brief inline tags like [helpbook] or [patient] in your answer when you use them.
- If data conflicts, clearly state both and prefer patient data.
- Be concise, structured, and avoid speculation.
- If missing info, state what else you'd need.

STYLE (clean narrative, visually appealing):
- Write as 2 to 3 short paragraphs, conversational but neutral.
- Use light inline emphasis:
  • Bold the test names and key numbers (e.g., **Hemoglobin 11.1 g/dL**).
  • Add status as a brief italic chip right after a value (e.g., *low*).
  • Show reference ranges inline in parentheses (e.g., (12 to 15 g/dL)).
  • Tag source right after the fact (e.g., [patient], [helpbook]).
- Use soft separators instead of headings/lists:
  • Start with a one-line takeaway: “In brief: …”
  • For grouped facts, place them on one line separated by “ · ” (e.g., **Hb** … · **MCV** … · **Ferritin** …).
  • You may use a single horizontal rule “---” once to break paragraphs.
- Keep sentences compact. Prefer concrete numbers over adjectives.
- No prescribing or firm diagnoses. Frame as education and suggest discussing with a clinician.
- If a fact is not in patient docs, say “Not found in patient docs” before adding brief context [helpbook].

Return: short summary, key findings, differentials (if any), and next steps.
"""

# Create a qn prompt
QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        # IMPORTANT: placeholder for chat memory
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)

# In-memory store
_memory_store: dict[str, ChatMessageHistory] = {}

# get chat history
def _get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _memory_store:
        _memory_store[session_id] = ChatMessageHistory()
    return _memory_store[session_id]

def _history_resolver(cfg: Union[str, Dict]) -> ChatMessageHistory:
    """
    LangChain sometimes passes just the session_id (str),
    other times a dict like {'configurable': {'session_id': '...'}}.
    Handle both to avoid 'string indices must be integers'.
    """
    if isinstance(cfg, str):
        sid = cfg
    elif isinstance(cfg, dict):
        sid = (cfg.get("configurable") or {}).get("session_id") or cfg.get("session_id")
    else:
        sid = None
    return _get_history(sid or "default")

# format medical document
def _format_docs(tag: str, docs: List[Document]) -> str:
    out = []
    for d in docs:
        chunk = d.page_content.strip().replace("\n", " ")
        out.append(f"[{tag}] {chunk}")
    return "\n".join(out)

# Answer qn using content retrival and RAG
def answer_question(
    question: str,
    general_index_name: str,
    patient_index_name: str,
    session_id: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    t0 = time.perf_counter()
    # Build retrievers
    general_vs = get_vectorstore(general_index_name)
    patient_vs = get_vectorstore(patient_index_name)

    # Retrieve
    general_ret = general_vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 100, "lambda_mult": 0.2},
    )
    patient_ret = patient_vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 50,
            "lambda_mult": 0.35,
            "filter": {"session_id": {"$eq": session_id}},
        },
    )

    patient_query = (
        question
        + " include exact units, reference ranges, and any symptoms/advisory sections"
    )

    # Fetch relevant documents based on the query
    general_docs = general_ret.get_relevant_documents(question)
    patient_docs = patient_ret.get_relevant_documents(patient_query)

    # --- Fallback if filter produced 0 docs (likely session_id mismatch) ---
    fallback_used=False
    if not patient_docs:
        broad_hits = patient_vs.similarity_search(patient_query, k=8)  # no filter
        if broad_hits:
            fallback_used=True
            # Use the session_id we actually find in the index (helps if your UI rotated IDs)
            fallback_sid = broad_hits[0].metadata.get("session_id")
            patient_docs = [d for d in broad_hits if d.metadata.get("session_id") == fallback_sid][:10]

    # Merge contexts
    ctx = []
    if general_docs:
        ctx.append(_format_docs("helpbook", general_docs))
    if patient_docs:
        ctx.append(_format_docs("patient", patient_docs))
    context = "\n".join(ctx) if ctx else "No retrieved context."
    #print("Patient context: ", patient_docs)
    #print("Context: ",context)

    global _last_context
    _last_context = context

    t_ret = time.perf_counter()

    # LLM + memory
    llm = get_llm()
    core_chain = QUESTION_PROMPT | llm

    chain_with_memory = RunnableWithMessageHistory(
        core_chain,
        _history_resolver,                 # <-- robust resolver
        input_messages_key="question",
        history_messages_key="history",
    )

    # get response based on the context retrieved
    resp = chain_with_memory.invoke(
        {"question": question, "context": context},
        config={"configurable": {"session_id": session_id}},
    )

    t_end = time.perf_counter()
    answer_text = getattr(resp, "content", str(resp))

    # Save performance metrics
    global _last_metrics
    _last_metrics = {
        "latency_ms_total": round((t_end - t0) * 1000, 1),
        "latency_ms_retrieval": round((t_ret - t0) * 1000, 1),
        "latency_ms_llm": round((t_end - t_ret) * 1000, 1),
        "retrieved_docs_patient": len(patient_docs or []),
        "retrieved_docs_helpbook": len(general_docs or []),
        "used_patient_in_answer": "[patient]" in answer_text,
        "used_helpbook_in_answer": "[helpbook]" in answer_text,
        "fallback_used": fallback_used,
        "context_chars": len(context),
        "answer_chars": len(answer_text),
    }
    return answer_text

def clear_session_memory(session_id: str):
    _memory_store.pop(session_id, None)

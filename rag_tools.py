# rag_tools.py
from rag import answer_question

# This directive gets prepended to every query the agent sends to RAG.
PATIENT_FIRST_PREFIX = (
    "Patient-first grounding:\n"
    "- Prioritise the patient's uploaded documents for this session.\n"
    "- If the needed value or detail is not present in patient docs, say 'Not found in patient docs' "
    "and then (optionally) add brief general guidance tagged [helpbook].\n"
    "- Do NOT ask the user to upload the report; it is already ingested for this session.\n\n"
)

# Answers the patient's qn
def _rag(question: str, general_index: str, patient_index: str, session_id: str) -> str:
    return answer_question(
        question=PATIENT_FIRST_PREFIX + question,
        general_index_name=general_index,
        patient_index_name=patient_index,
        session_id=session_id,
        chat_history=[],  # agent holds dialog memory
    )

def rag_tool(question: str, general_index: str, patient_index: str, session_id: str) -> str:
    return _rag(question, general_index, patient_index, session_id)

# Summarising tool
def summarise_patient_report(general_index: str, patient_index: str, session_id: str) -> str:
    prompt = (
        "Summarise the current patient's uploaded lab/clinical report. "
        "Pull concrete values with reference ranges, flag out-of-range items, "
        "give a short clinical interpretation and next steps. Be concise and structured."
    )
    return _rag(prompt, general_index, patient_index, session_id)

# Interpret lab results
def interpret_lab(test_name: str, general_index: str, patient_index: str, session_id: str) -> str:
    prompt = (
        f"Interpret the patient's {test_name}. If present, cite the exact value and reference range "
        f"from the patient's documents and say if it is low/normal/high. "
        f"Add a brief, non-diagnostic explanation and what to discuss with a clinician."
    )
    return _rag(prompt, general_index, patient_index, session_id)

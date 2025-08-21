# metrics.py
# Summarize one session (avg faith/help + avg latencies + useful rates) and append to CSV.

import csv, time, pathlib, statistics
from functools import lru_cache
from typing import Dict, List, Optional
from langchain.evaluation import load_evaluator
from llm import get_llm

def _to_float(x, default=0.0):
    try: return float(x)
    except Exception:
        s = str(x).strip().lower()
        if s in {"y","yes","true"}: return 1.0
        if s in {"n","no","false"}: return 0.0
        return default

@lru_cache(maxsize=1)
def _get_judges():
    llm = get_llm()  # keep temperature low in llm.py for stable scoring
    faith = load_evaluator(
        "labeled_criteria", llm=llm,
        criteria={"faithfulness": "Is the answer supported by the provided context?"}
    )
    helpf = load_evaluator(
        "criteria", llm=llm,  # no reference required
        criteria={"helpfulness": "Is it directly useful to the user?"}
    )
    # return helpfulness and faithfulness metrics
    return faith, helpf

def summarize_session(turns: List[Dict], faith_threshold: float = 0.5) -> Dict[str, float]:
    """
    Each turn: {"q":str, "answer":str, "context":str, "ts":..., "metrics":{...}}
      where metrics (set in rag.py) may include:
        latency_ms_total, latency_ms_retrieval, latency_ms_llm,
        retrieved_docs_patient, retrieved_docs_helpbook,
        used_patient_in_answer, used_helpbook_in_answer, fallback_used,
        context_chars, answer_chars
    Returns a flat dict ready to write to CSV.
    """
    if not turns:
        return {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "avg_faithfulness": 0.0, "avg_helpfulness": 0.0, "turns_scored": 0,
            "avg_latency_ms_total": 0.0, "avg_latency_ms_retrieval": 0.0, "avg_latency_ms_llm": 0.0,
            "retrieval_success_rate_pct": 0.0, "grounded_in_patient_rate_pct": 0.0,
            "helpbook_cite_rate_pct": 0.0, "fallback_rate_pct": 0.0,
            "empty_context_rate_pct": 0.0, "avg_context_chars": 0.0, "avg_answer_chars": 0.0,
            "avg_retrieved_docs_patient": 0.0, "avg_retrieved_docs_helpbook": 0.0,
            "hallucination_rate_pct": 0.0,
        }

    faith_eval, help_eval = _get_judges()
    f_scores, h_scores = [], []

    lat_total=[]; lat_ret=[]; lat_llm=[]
    used_patient=[]; used_helpbook=[]; fallback=[]
    ctx_chars=[]; ans_chars=[]; empty_ctx=[]
    ret_succ=[]; rdp=[]; rdh=[]

    for t in turns:
        q = (t.get("q") or "").strip()
        a = (t.get("answer") or "").strip()
        c = (t.get("context") or "").strip()
        if q and a and c:
            # Evaluate faithfulness and helpfulness
            f = faith_eval.evaluate_strings(prediction=a, input=q, reference=c).get("score")
            h = help_eval.evaluate_strings(prediction=a, input=q).get("score")
            f_scores.append(_to_float(f)); h_scores.append(_to_float(h))

        m = t.get("metrics") or {}
        def num(k):
            v = m.get(k)
            return float(v) if isinstance(v, (int, float)) else None
        # Get latency
        x = num("latency_ms_total");      lat_total += [x] if x is not None else []
        x = num("latency_ms_retrieval");  lat_ret   += [x] if x is not None else []
        x = num("latency_ms_llm");        lat_llm   += [x] if x is not None else []

        ctx_chars.append(len(c))
        ans_chars.append(len(a))
        empty_ctx.append(1 if not c or c.strip().lower().startswith("no retrieved context") else 0)

        used_patient.append(1 if m.get("used_patient_in_answer") else 0)
        used_helpbook.append(1 if m.get("used_helpbook_in_answer") else 0)
        fallback.append(1 if m.get("fallback_used") else 0)
        #Get number of retrieved docs
        rp = m.get("retrieved_docs_patient"); rdp += [float(rp)] if isinstance(rp,(int,float)) else []
        rh = m.get("retrieved_docs_helpbook"); rdh += [float(rh)] if isinstance(rh,(int,float)) else []
        ret_succ.append(1 if (isinstance(rp,(int,float)) and rp>0) else 0)

    n = len(f_scores)
    mean = lambda xs: round(sum(xs)/len(xs), 2) if xs else 0.0
    pct  = lambda xs: round(100.0*sum(xs)/len(xs), 1) if xs else 0.0

    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "avg_faithfulness": round(statistics.mean(f_scores), 4) if n else 0.0,
        "avg_helpfulness": round(statistics.mean(h_scores), 4) if n else 0.0,
        "turns_scored": n,
        "avg_latency_ms_total": mean(lat_total),
        "avg_latency_ms_retrieval": mean(lat_ret),
        "avg_latency_ms_llm": mean(lat_llm),
        "retrieval_success_rate_pct": pct(ret_succ),        # “Internal Query Accuracy”
        "grounded_in_patient_rate_pct": pct(used_patient),  # answers citing [patient]
        "helpbook_cite_rate_pct": pct(used_helpbook),
        "fallback_rate_pct": pct(fallback),
        "empty_context_rate_pct": pct(empty_ctx),
        "avg_context_chars": mean(ctx_chars),
        "avg_answer_chars": mean(ans_chars),
        "avg_retrieved_docs_patient": mean(rdp),
        "avg_retrieved_docs_helpbook": mean(rdh),
        "hallucination_rate_pct": round(
            100.0 * (sum(1 for s in f_scores if s < faith_threshold) / n), 1
        ) if n else 0.0,
    }

# append summary to csv file
def append_session_summary(csv_path: str, session_id: str,
                           summary: Dict[str, float],
                           extra: Optional[Dict[str, object]] = None):
    row = {"session_id": session_id, **summary}
    if extra:
        for k, v in extra.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
    path = pathlib.Path(csv_path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

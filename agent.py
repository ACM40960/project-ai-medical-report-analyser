# agent.py (only the tools assembly part changes)
from functools import partial
from typing import Callable
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from llm import get_llm
from rag_tools import rag_tool, summarise_patient_report, interpret_lab
from web_tools import get_web_tools   # <-- NEW

#To create a langchain agent
def create_agent(general_index: str, patient_index: str, session_id: str):
    llm = get_llm() # Get a defined llm
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # Define conversation buffer memory

    # Bind params for RAG tools
    rag_bound: Callable[[str], str] = partial(rag_tool, general_index=general_index, patient_index=patient_index, session_id=session_id)
    summary_bound: Callable[[], str] = partial(summarise_patient_report, general_index=general_index, patient_index=patient_index, session_id=session_id)
    interpret_bound: Callable[[str], str] = partial(interpret_lab, general_index=general_index, patient_index=patient_index, session_id=session_id)

    # Define tools ie. RAG, websearch etc.
    tools = [
        Tool(
            name="RAG_QA",
            func=rag_bound,
            description=("DEFAULT. Use FIRST for ANY user question about this patient. "
                         "Ground in patient docs; fall back to helpbook if needed."),
        ),
        Tool(
            name="Summarise_Patient_Report",
            func=lambda _: summary_bound(),
            description="Summarise the patient's report with values, ranges, flags, interpretation, next steps.",
        ),
        Tool(
            name="Interpret_Lab",
            func=interpret_bound,
            description="Explain one lab (input: test name). Cite patient value & range if present; state low/normal/high.",
        ),
    ]

    # Append web search tools
    tools += get_web_tools(num_results=5)

    system_msg = (
        "Patient and helpbook docs are embedded for THIS session. "
        "ALWAYS try a RAG tool first (RAG_QA / Summarise_Patient_Report / Interpret_Lab). "
        "Use WEB_SEARCH_* only if the needed info is not in patient docs/helpbook or must be up-to-date. "
        "Do not ask the user to upload; documents are already ingested."
    )

    #Initialise an agent with all the components defined above
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_msg},
    )
    return agent, memory

# ===========================================
# file: llm.py
# Gemini 1.5 Flash via LangChain
# ===========================================
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Define llm using gemini
def get_llm():
    # Requires GOOGLE_API_KEY env var
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        max_output_tokens=1024,
        convert_system_message_to_human= True 
    )

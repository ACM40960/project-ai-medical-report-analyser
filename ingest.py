# ===========================================
# file: ingest.py
# Load & chunk PDFs/TXT and upsert to Pinecone
# ===========================================
from typing import List, BinaryIO
import os
import uuid
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from vectorstore import get_vectorstore


# ----------------------------
# Chunking
# ----------------------------
def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# ----------------------------
# Loaders (Windows/Cloud-safe)
# ----------------------------
def _load_pdf(file: BinaryIO, source_name: str) -> List[Document]:
    """
    Writes the uploaded file to a *real* temp file (portable),
    parses with PyPDFLoader, and attaches basic metadata.
    """
    # Create an actual temp file path; close the OS handle right away
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    try:
        # Reset pointer if supported
        try:
            file.seek(0)
        except Exception:
            pass

        # Write bytes (supports Streamlit's UploadedFile)
        with open(tmp_path, "wb") as out:
            try:
                out.write(file.getbuffer())
            except AttributeError:
                out.write(file.read())

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()  # one Document per page

        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = source_name

        return docs

    finally:
        # Always clean up the temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _load_txt(file: BinaryIO, source_name: str) -> List[Document]:
    """
    Reads a text-like file into a single Document with basic metadata.
    """
    try:
        file.seek(0)
    except Exception:
        pass

    # Read raw bytes, then decode
    raw = getattr(file, "read", lambda: b"")()
    if isinstance(raw, (bytes, bytearray)):
        content = raw.decode("utf-8", errors="ignore")
    else:
        # Some file-like objects return str
        content = str(raw)

    return [Document(page_content=content, metadata={"source": source_name})]


# ----------------------------
# Public ingest functions
# ----------------------------
def ingest_helpbook_pdf(uploaded_file, general_index_name: str) -> int:
    """
    - Saves uploaded PDF to a temp path (Windows/Cloud-safe)
    - Parses with PyPDFLoader
    - Adds minimal metadata
    - Splits and upserts into your vectorstore
    Returns: number of chunks upserted.
    """
    idx = get_vectorstore(general_index_name)

    # 1) Write to temp, parse
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)

    try:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        with open(tmp_path, "wb") as f:
            try:
                f.write(uploaded_file.getbuffer())
            except AttributeError:
                f.write(uploaded_file.read())

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # 2) Attach metadata
        src_name = getattr(uploaded_file, "name", "uploaded.pdf")
        batch_id = uuid.uuid4().hex[:8]
        for i, d in enumerate(docs):
            d.metadata = d.metadata or {}
            d.metadata.update(
                {
                    "source": src_name,
                    "batch_id": batch_id,
                    "page": d.metadata.get("page", i),
                    "kind": "helpbook",
                }
            )

        # 3) Split
        chunks = _split_docs(docs)
        if not chunks:
            return 0

        # 4) Stable IDs to avoid dupes on re-ingest
        for j, c in enumerate(chunks):
            c.metadata = c.metadata or {}
            c.metadata.setdefault("id", f"{batch_id}-{j}")

        # 5) Upsert
        idx.add_documents(chunks)
        return len(chunks)

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

# ingest patient files
def ingest_patient_files(files, patient_index_name: str, session_id: str) -> int:
    """
    Load one or more patient files and embed into the PATIENT index with session metadata.
    Returns: number of chunks upserted.
    """
    # Get pinecone vector store
    idx = get_vectorstore(patient_index_name)
    all_chunks: List[Document] = []

    for f in files:
        name = getattr(f, "name", "patient_upload")
        src = f"patient_{name}"

        if name.lower().endswith(".pdf"):
            docs = _load_pdf(f, source_name=src)
        else:
            docs = _load_txt(f, source_name=src)

        # Tag page docs with session info
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata["session_id"] = session_id
            d.metadata["kind"] = "patient"
            d.metadata.setdefault("source", src)

        # Split and propagate metadata
        chunks = _split_docs(docs)
        batch_id = uuid.uuid4().hex[:8]

        for j, c in enumerate(chunks):
            c.metadata = c.metadata or {}
            c.metadata.setdefault("session_id", session_id)
            c.metadata.setdefault("kind", "patient")
            c.metadata.setdefault("source", src)
            c.metadata.setdefault("id", f"{batch_id}-{j}")

        all_chunks.extend(chunks)

    if all_chunks:
        idx.add_documents(all_chunks)

    return len(all_chunks)

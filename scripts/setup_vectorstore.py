"""
One-time ChromaDB setup — chunks policy docs and embeds them.

Run once before starting the app:
    python scripts/setup_vectorstore.py

The policy_rag_agent also lazy-initialises on first call, so this script
is only needed if you want to pre-warm the vectorstore (e.g. in CI/Docker).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

from utils.llm import get_embeddings
from config import config


def setup() -> None:
    docs_dir    = Path(config.POLICY_DOCS_DIR)
    persist_dir = config.CHROMA_PERSIST_DIR

    print(f"Loading policy documents from {docs_dir} …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )

    documents: list[Document] = []
    for md_file in sorted(docs_dir.glob("*.md")):
        text   = md_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": md_file.name, "chunk_index": i},
            ))
        print(f"  {md_file.name}: {len(chunks)} chunks")

    print(f"\nEmbedding {len(documents)} chunks (first run downloads ~80MB model) …")
    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"ChromaDB persisted to {persist_dir}")
    print("Setup complete.")


if __name__ == "__main__":
    setup()

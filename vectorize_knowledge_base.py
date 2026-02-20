"""
RAG Vectorization Script - Qwen Full Suite Version
Vectorizes the Zino's Petrel literature library and stores it in ChromaDB

Usage:
    python vectorize_knowledge_base.py

Features:
    - Batch processes PDF files
    - Optimized document splitting (chunk_overlap=200)
    - Qwen Embedding (text-embedding-v3)
    - Progress tracking and error handling
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import json

# Load environment variables
load_dotenv()

# Configuration
PDF_FOLDER = "zinos_petrel_knowledge"
MUSEUM_DOCS_FOLDER = "museum_summaries"  # put your JSON/JSONL or txt here
COLLECTION_NAME = "mmf_zinospetrel_knowledge"
VECTOR_DB_PATH = "db5_qwen"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # 20% overlap, maintains context continuity
# Read Embedding model from environment variables (consistent with rag_utils.py)
EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

def sanitize_metadata(meta: dict) -> dict:
    """
    Chroma requires metadata values to be str/int/float/bool.
    Convert lists/dicts to strings.
    """
    clean = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ",".join(map(str, v))
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = str(v)
    return clean


def get_pdf_files(folder_path):
    """Get all PDF files in the folder"""
    pdf_path = Path(folder_path)
    if not pdf_path.exists():
        print(f"❌ Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  Warning: No PDF files found in folder '{folder_path}'")
        sys.exit(1)
    
    return pdf_files

def load_and_split_pdf(pdf_path, text_splitter):
    """Load and split a single PDF file"""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # Add metadata to each document
        for i, page in enumerate(pages):
            page.metadata.update({
                "doc_type": "pdf",
                "priority": 10,          # default lower than MMF summaries
                "scope": ["general"],
                "language": "unknown"    # optional; you can set "pt"/"en" if you know per file
            })

        
        # Split documents
        chunks = text_splitter.split_documents(pages)
        return chunks, None
    
    except Exception as e:
        return None, str(e)

def vectorize_documents(pdf_files, embeddings, text_splitter):
    """Vectorize all documents"""
    all_chunks = []
    failed_files = []
    
    print(f"\n📚 Starting to process {len(pdf_files)} PDF files...\n")
    
    # Use tqdm to show progress
    for pdf_file in tqdm(pdf_files, desc="Processing PDF", unit="file"):
        chunks, error = load_and_split_pdf(pdf_file, text_splitter)
        
        if error:
            failed_files.append((pdf_file.name, error))
            tqdm.write(f"❌ Failed: {pdf_file.name} - {error}")
        else:
            all_chunks.extend(chunks)
            tqdm.write(f"✅ Success: {pdf_file.name} ({len(chunks)} chunks)")
    
    print(f"\n📊 Statistics:")
    print(f"  - Successful: {len(pdf_files) - len(failed_files)} files")
    print(f"  - Failed: {len(failed_files)} files")
    print(f"  - Total chunks: {len(all_chunks)} chunks")
    
    if failed_files:
        print(f"\n⚠️  Failed files list:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    return all_chunks

def load_museum_docs(folder_path: str):
    """
    Loads museum summary docs from JSON or JSONL files.
    Expected fields:
      - text (or text_pt / text_en)
      - priority (int)
      - scope (list)
      - language ("pt"/"en")
      - source, title, id (optional)
    """
    docs = []
    folder = Path(folder_path)
    if not folder.exists():
        print(f"ℹ️  Museum docs folder '{folder_path}' not found — skipping.")
        return docs

    for fp in folder.glob("*"):
        if fp.suffix.lower() not in [".json", ".jsonl", ".txt", ".md"]:
            continue

        if fp.suffix.lower() in [".txt", ".md"]:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": "MMF internal summary",
                    "source_file": fp.name,
                    "priority": 100,
                    "scope": ["mmf", "specimen"],
                    "language": "pt"
                }
            ))
            continue

        raw = fp.read_text(encoding="utf-8", errors="ignore")
        if fp.suffix.lower() == ".jsonl":
            lines = [l for l in raw.splitlines() if l.strip()]
            items = [json.loads(l) for l in lines]
        else:
            items = [json.loads(raw)]
            if isinstance(items[0], list):
                items = items[0]

        for item in items:
            # Prefer English text if you stored it; otherwise keep PT.
            text = item.get("text_en") or item.get("text") or item.get("text_pt")
            if not text:
                continue

            meta = {
                "source": item.get("source", "MMF internal summary"),
                "title": item.get("title", fp.stem),
                "doc_id": item.get("id", fp.stem),
                "source_file": fp.name,
                "priority": int(item.get("priority", 100)),
                "scope": item.get("scope", ["mmf"]),
                "language": item.get("language", "pt"),
                "doc_type": "museum_summary"
            }
            docs.append(Document(page_content=text, metadata=meta))

    print(f"✅ Loaded {len(docs)} museum summary docs")
    return docs

def split_documents(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(docs)


def create_vector_store(chunks, embeddings, persist_directory):
    """Create and persist vector database"""
    print(f"\n🔄 Creating vector database...")
    print(f"  - Vector store path: {persist_directory}")
    print(f"  - Embedding model: {EMBEDDING_MODEL}")
    print(f"  - Document chunk count: {len(chunks)}")
    
    try:
        # Clear old database (if exists)
        if Path(persist_directory).exists():
            import shutil
            shutil.rmtree(persist_directory)
            print(f"  - Cleared old database")
        
        # Batch process vectorization (DashScope limit: batch_size ≤ 10)
        batch_size = 10
        vectordb = None
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Vectorizing", unit="batch"):
            batch = chunks[i:i + batch_size]

            for d in batch:
                d.metadata = sanitize_metadata(d.metadata)

            if vectordb is None:
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name=COLLECTION_NAME
                )
            else:
                vectordb.add_documents(batch)

        print(f"\n✅ Vector database created successfully!")
        return vectordb
    
    except Exception as e:
        print(f"\n❌ Vector database creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_retrieval(vectordb):
    print(f"\n🧪 Testing retrieval functionality (MMF monk seal)...")

    test_queries = [
        "When was the MMF monk seal specimen captured?",
        "How many monk seals live in Madeira?",
        "What threats do monk seals face near Madeira?"
    ]

    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = retrieve_with_priority(vectordb, query, k_final=4)

        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    - Priority: {doc.metadata.get('priority')}")
            print(f"    - Scope: {doc.metadata.get('scope')}")
            print(f"    - Source: {doc.metadata.get('source_file', doc.metadata.get('source', 'Unknown'))}")
            print(f"    - Page: {doc.metadata.get('page', 'N/A')}")
            preview = doc.page_content[:160]
            preview = preview.replace("\n", " ").strip()
            print(f"    - Preview: {preview}...")



def retrieve_with_priority(vectordb, query: str, k_final: int = 6):
    candidates = vectordb.similarity_search(query, k=30)

    def score(doc):
        priority = int(doc.metadata.get("priority", 0))

        scope = doc.metadata.get("scope", "")  # now string
        bonus = 0
        if "specimen" in scope or "mmf" in scope:
            bonus = 50

        return priority + bonus

    candidates.sort(key=score, reverse=True)
    return candidates[:k_final]



def main():
    """Main function"""
    print("=" * 60)
    print("📚 RAG Vectorization Script - Qwen Full Suite Version")
    print("=" * 60)
    
    # 1. Check API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ Error: DASHSCOPE_API_KEY not found")
        print("Please configure API Key in .env file")
        sys.exit(1)
    
    print(f"✅ API Key configured")

    museum_docs = load_museum_docs(MUSEUM_DOCS_FOLDER)

    # Use smaller chunks for museum facts (high signal)
    museum_chunks = split_documents(museum_docs, chunk_size=500, chunk_overlap=80) if museum_docs else []

    # 2. Get PDF file list
    pdf_files = get_pdf_files(PDF_FOLDER)
    print(f"✅ Found {len(pdf_files)} PDF files")
    
    # 3. Initialize Embeddings
    print(f"\n🔧 Initializing Embedding model...")
    embeddings = DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=api_key
    )
    print(f"✅ Using model: {EMBEDDING_MODEL}")
    
    # 4. Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    print(f"✅ Text splitting configuration: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    # 5. Vectorize documents
    chunks = vectorize_documents(pdf_files, embeddings, text_splitter)
    
    chunks = museum_chunks + chunks
    print(f"✅ Total chunks after adding museum docs: {len(chunks)}")

    if not chunks:
        print("❌ No documents processed successfully")
        sys.exit(1)
    
    # 6. Create vector database
    vectordb = create_vector_store(chunks, embeddings, VECTOR_DB_PATH)
    
    # 7. Test retrieval
    test_retrieval(vectordb)
    
    print("\n" + "=" * 60)
    print("🎉 Vectorization completed!")
    print("=" * 60)
    print(f"\n📁 Vector store location: {VECTOR_DB_PATH}")
    print(f"📊 Total document chunks: {len(chunks)}")
    print(f"\nNext step: Run 'streamlit run main.py' to start using!")

if __name__ == "__main__":
    main()


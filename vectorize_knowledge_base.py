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

# Load environment variables
load_dotenv()

# Configuration
PDF_FOLDER = "Mediterranean Monk Seal"
VECTOR_DB_PATH = "db6_qwen"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200  # 20% overlap, maintains context continuity
# Read Embedding model from environment variables (consistent with rag_utils.py)
EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

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
                "source_file": pdf_path.name,
                "page": i + 1,
                "total_pages": len(pages)
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
            
            if vectordb is None:
                # First time creation
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name="zinos_petrel_knowledge"
                )
            else:
                # Add to existing database
                vectordb.add_documents(batch)
        
        print(f"\n✅ Vector database created successfully!")
        return vectordb
    
    except Exception as e:
        print(f"\n❌ Vector database creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_retrieval(vectordb):
    """Test retrieval functionality"""
    print(f"\n🧪 Testing retrieval functionality...")
    
    test_queries = [
        "What is Zino's Petrel?",
        "Where does Zino's Petrel nest?",
        "What does Zino's Petrel eat?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = vectordb.similarity_search(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    - Source: {doc.metadata.get('source_file', 'Unknown')}")
            print(f"    - Page: {doc.metadata.get('page', 'N/A')}")
            print(f"    - Content preview: {doc.page_content[:150]}...")

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


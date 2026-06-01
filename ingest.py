from dotenv import load_dotenv

load_dotenv()

"""
ingest.py
This script loads text files from a data directory, splits them into chunks,
and stores them in a local Chroma vector database.
"""
import os
import shutil
import re
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Configuration
DATA_PATH = "./data"
DB_PATH = "./chroma_db"

def ingest_documents():
    # 1. Check if data directory exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH}. Please place your book text files (.txt) there and run this script again.")
        return

    # 2. Load Documents
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    if not docs:
        print("No documents found in ./data/")
        return
    print(f"Loaded {len(docs)} documents.")

    # 3. Split Text
    # Chunk size is important for novels to keep enough context (1000 chars is roughly 150-250 words)
    # Overlap helps with continuity across chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    
    final_splits = []

    for doc in docs:
        content = doc.page_content
        
        # Regex patterns
        part_pattern = re.compile(r'(?m)^(?:Part|PART)\s+(?:[0-9]+|[IVXLCDM]+|[a-zA-Z]+).*$')
        chapter_pattern = re.compile(r'(?m)^(?:Chapter|CHAPTER)\s+(?:[0-9]+|[IVXLCDM]+).*$')
        
        parts = []
        part_matches = list(part_pattern.finditer(content))
        
        if not part_matches:
            parts.append({"title": "Whole Text", "content": content, "start_char": 0})
        else:
            if part_matches[0].start() > 0:
                parts.append({"title": "Front Matter", "content": content[:part_matches[0].start()], "start_char": 0})
            for i, match in enumerate(part_matches):
                start = match.start()
                end = part_matches[i+1].start() if i + 1 < len(part_matches) else len(content)
                parts.append({"title": match.group(0).strip(), "content": content[start:end], "start_char": start})

        for part in parts:
            part_content = part["content"]
            part_start = part["start_char"]
            
            chapters = []
            chap_matches = list(chapter_pattern.finditer(part_content))
            
            if not chap_matches:
                chapters.append({"title": "Section", "content": part_content, "rel_start": 0})
            else:
                if chap_matches[0].start() > 0:
                    chapters.append({"title": "Part Intro", "content": part_content[:chap_matches[0].start()], "rel_start": 0})
                for i, match in enumerate(chap_matches):
                    start = match.start()
                    end = chap_matches[i+1].start() if i + 1 < len(chap_matches) else len(part_content)
                    chapters.append({"title": match.group(0).strip(), "content": part_content[start:end], "rel_start": start})
            
            for chap in chapters:
                chap_doc = Document(page_content=chap["content"], metadata=doc.metadata.copy())
                chap_doc.metadata["part"] = part["title"]
                chap_doc.metadata["chapter"] = chap["title"]
                
                # Split chapter into chunks (truncates at end of chapter)
                chap_splits = text_splitter.split_documents([chap_doc])
                
                for split in chap_splits:
                    # Calculate absolute page number
                    absolute_pos = part_start + chap["rel_start"] + split.metadata.get("start_index", 0)
                    split.metadata["page_number"] = (absolute_pos // 1500) + 1
                
                final_splits.extend(chap_splits)
            
    print(f"Split into {len(final_splits)} chunks.")

    # 4. Embed and Store
    # Clear existing DB if you want a fresh start (optional)
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("Embedding and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(
        documents=final_splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f"Ingestion complete. Vector store saved to {DB_PATH}")

if __name__ == "__main__":
    ingest_documents()
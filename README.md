# Reading Assistant RAG 📖🤖

An advanced, LangGraph-powered Retrieval-Augmented Generation (RAG) application designed to act as an expert literary assistant. This tool helps readers find connections, cross-references, and specific details within novels or books by parsing raw text files, preserving structural metadata (Parts, Chapters, and estimated Page Numbers), and querying them with precise citations.

---

## ✨ Key Features

- **Smart Document Ingestion**: Automatically parses text files into structural sections (Parts and Chapters) using regular expressions, maintaining the narrative context.
- **Page Number Estimation**: Calculates approximate page numbers based on character offsets (assuming ~1500 characters per page) to provide realistic citations.
- **Vector Database**: Embeds text chunks using OpenAI Embeddings and stores them in a local, persistent ChromaDB instance.
- **LangGraph Workflow**: Implements a state-based graph (`retrieve` ➔ `generate`) using LangGraph to manage the query-and-response lifecycle cleanly.
- **Source Citations**: The assistant is prompted to always cite the specific Part, Chapter, and Page number for every piece of information it retrieves.
- **Interactive CLI**: A command-line interface to chat with your book in real-time.

---

## 📂 Project Structure

```bash
reading_assistant_RAG/
│
├── ingest.py          # Loads, parses, chunks, and embeds book text into ChromaDB
├── assistant.py       # LangGraph application running the interactive Q&A loop
├── requirements.txt   # Project dependencies
└── data/              # Directory where your raw book text files (.txt) are stored
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You will also need an OpenAI API key.

### 2. Installation
Clone this repository and install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Add Your Books
Create a directory named `data` in the root folder (if it doesn't exist yet) and place your book text files (`.txt`) inside it:

```bash
mkdir data
# Place your book files (e.g., war_and_peace.txt) inside ./data
```

---

## 🛠️ Usage

### Step 1: Ingest and Index the Book
Run `ingest.py` to parse the text files, split them into chunks, and build the local vector database:

```bash
python ingest.py
```

*This script will scan the `./data` directory, identify Parts and Chapters, calculate page numbers, and save the embeddings to `./chroma_db`.*

### Step 2: Start the Reading Assistant
Run `assistant.py` to start the interactive CLI session:

```bash
python assistant.py
```

You can now ask questions about the book, such as:
* *"Who is introduced in Chapter 2 of Part 1?"*
* *"What are the recurring themes of betrayal in the story?"*
* *"Where does the main character go after the confrontation in Part 2?"*

Type `exit` or `quit` to end the session.

---

## 🧠 How It Works

### 1. Ingestion Pipeline (`ingest.py`)
- **Regex Parsing**: The script scans the text for patterns like `PART <num>` and `CHAPTER <num>`.
- **Hierarchical Chunking**: It splits the text by Part and Chapter first, ensuring that chunks do not cross chapter boundaries.
- **Metadata Enrichment**: Each chunk is tagged with:
  - `part`: The title of the Part it belongs to.
  - `chapter`: The title of the Chapter.
  - `page_number`: Calculated as `(absolute_character_offset // 1500) + 1`.
- **Vector Storage**: Chunks are embedded using `OpenAIEmbeddings` and saved to a local `Chroma` database.

### 2. LangGraph Workflow (`assistant.py`)
The assistant uses a state-based graph to manage the RAG pipeline:

```
[Start] ➔ [Retrieve Node] ➔ [Generate Node] ➔ [End]
```

- **State**: A `TypedDict` containing the `question`, retrieved `context` (list of Documents), and the final `answer`.
- **Retrieve Node**: Queries the Chroma vector store to fetch the top 10 most relevant chunks (providing a broad context for cross-references).
- **Generate Node**: Formats the retrieved chunks with their metadata and prompts `gpt-4o` to generate a detailed response with precise citations.

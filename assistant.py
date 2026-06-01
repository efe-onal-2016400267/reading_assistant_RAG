from dotenv import load_dotenv

load_dotenv()

"""
assistant.py
A LangGraph application that acts as a reading assistant.
"""
import os
from typing import List, TypedDict

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Configuration
DB_PATH = "./chroma_db"

# 1. Setup Resources (LLM & Retriever)
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Vector store not found at {DB_PATH}. Run ingest.py first.")

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
# Retrieve top 5 chunks to get broader context for cross-references
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Define Graph State
class AssistantState(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 3. Define Nodes

def retrieve(state: AssistantState):
    """
    Node to retrieve relevant documents from the vector store.
    """
    print(f"--- RETRIEVING CONTEXT FOR: {state['question']} ---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"context": documents}

def generate(state: AssistantState):
    """
    Node to generate the answer using the retrieved context.
    """
    print("--- GENERATING ANSWER ---")
    question = state["question"]
    context = state["context"]
    
    # Format context for the prompt
    context_text = "\n\n".join([
        f"[Part: {doc.metadata.get('part', 'N/A')} | Chapter: {doc.metadata.get('chapter', 'Unknown')} | Page: {doc.metadata.get('page_number', '?')}]\n{doc.page_content}"
        for doc in context
    ])
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert literary assistant helping a reader with a novel. 
        Your goal is to help the user find connections, cross-references, and specific details within the text.
        
        Use the following pieces of retrieved context to answer the question.
        Always cite the part (if applicable), chapter, and page number from the context when providing details.
        
        Context:
        {context}

        Question: 
        {question}

        Answer:"""
    )
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": question})
    return {"answer": answer}

# 4. Build the Graph
workflow = StateGraph(AssistantState)

# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Define edges
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()

# 5. Helper function to run the assistant
def query_book(question: str):
    inputs = {"question": question}
    result = app.invoke(inputs)
    return result["answer"]

if __name__ == "__main__":
    # Interactive loop
    print("Reading Assistant Ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nAsk about the book: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = query_book(user_input)
        print(f"\nAssistant:\n{response}")
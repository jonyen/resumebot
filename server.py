from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from datetime import date

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain_core.runnables import RunnableLambda

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
PGVECTOR_CONN = os.getenv("PGVECTOR_CONN")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class ChatRequest(BaseModel):
    session_id: str  # required to identify conversation
    message: str

# LLM + Prompt
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

resume_text = extract_text_from_pdf("jonathan-yen-resume-july-2025.pdf")

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([resume_text])

# Embed using Hugging Face
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in pgvector
COLLECTION_NAME = "resume_chunks"

vectorstore = PGVector.from_documents(
    embedding=embedding,
    documents=chunks,
    collection_name=COLLECTION_NAME,
    connection_string=PGVECTOR_CONN,
)

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant." + "Today's date is " + date.today().strftime('%Y-%m-%d')),
    ("system", "Keep answers to 1-2 sentences"),
    ("system", "Context from resume:\n{context}"),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{input}"),
])

def build_rag_chain():
    return (
        RunnableLambda(lambda x: {"context": retriever.get_relevant_documents(x["input"]), **x})
        | RunnableLambda(lambda x: {
              "context": "\n\n".join([doc.page_content for doc in x["context"]]),
              "input": x["input"],
              "messages": x.get("messages", [])
          })
        | prompt
        | llm
    )

# Memory registry (in-memory map of sessions)
memory_store = {}

chat_chain = RunnableWithMessageHistory(
    build_rag_chain(),
    lambda session_id: memory_store.setdefault(session_id, InMemoryChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="messages",
)

# Endpoint
@app.post("/chat")
async def chat(req: ChatRequest):
    response = chat_chain.invoke(
        {"input": req.message},
        config={"configurable": {"session_id": req.session_id}}
    )
    return {"reply": response.content}


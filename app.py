
# Agentic RAG Customer Support Assistant - Full Code with Streamlit UI (Free stack)

# === Step 1: Imports ===
import streamlit as st
import faiss
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# === Step 2: Initialize Embeddings and Vector Store ===
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

if not os.path.exists("faiss_index"):
    st.info("Creating vector store from sample documents...")
    loader = TextLoader("sample_data/support_docs.txt")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(chunks, embedder)
    vectordb.save_local("faiss_index")
else:
    vectordb = FAISS.load_local("faiss_index", embedder)

# === Step 3: Load Local LLM (Using HuggingFace pipeline) ===
@st.cache_resource
def load_local_llm():
    llm_pipeline = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)

llm = load_local_llm()

# === Step 4: Build QA Chain ===
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

# === Step 5: Streamlit UI ===
st.set_page_config(page_title="Customer Support RAG Bot", layout="wide")
st.title("🤖 Customer 360 Support Assistant (Free LLM + FAISS + LangChain)")

user_query = st.text_input("Ask your support question (e.g., 'Why was I charged twice?')")

if user_query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(user_query)
        st.markdown("### 📩 Response:")
        st.success(result)

# === Optional: Upload new documents ===
st.sidebar.title("📄 Add New Docs")
uploaded_file = st.sidebar.file_uploader("Upload a TXT file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    doc = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    vectordb.add_documents(chunks)
    vectordb.save_local("faiss_index")
    st.sidebar.success("Document added to knowledge base!")

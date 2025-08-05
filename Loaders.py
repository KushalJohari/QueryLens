from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

llm = ChatGroq(model="llama3-8b-8192", api_key=	st.secrets["GROQ_API_KEY"])

prompt = PromptTemplate(template=f"""
You are a multilingual assistant. The user may ask questions in any language.
The content you receive (context) might be in a different language (e.g., Hindi, Tamil, French).
Your job is to **always answer in the language of the question**, even if the context is written in another language.

If the question is in English, and the context is in Hindi, **translate the context and answer in English.**
Always prioritize the question language over the context language.
                        
example: user asks in hindi "इसमें क्या लिखा है" you will read the document user uploaded and then tell about the document in summary.

Context:    
{{context}}

Question:
{{question}}

Answer in the same language as the question:
""", input_variables=["context", "question"])


def webloader(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    return document

def pdfloader(pdf):
    pdf_loader = PyPDFLoader(pdf)
    pdf_document = pdf_loader.load()
    return pdf_document

def qa_chain_web(documents):
    embeddings = HuggingFaceEmbeddings(model_name= "BAAI/bge-m3")
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = textsplitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever(), chain_type_kwargs={'prompt':prompt})

def qa_chain_pdf(pdf_documents):
    embeddings = HuggingFaceEmbeddings(model_name= "BAAI/bge-m3")
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = textsplitter.split_documents(pdf_documents)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever(), chain_type_kwargs={'prompt':prompt})


from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import  MBart50TokenizerFast, MBartForConditionalGeneration
from pathlib import Path
import os
from langdetect import detect
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama3-8b-8192", api_key=	os.getenv("GROQ_API_KEY"))
lang_model = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(lang_model)
model = MBartForConditionalGeneration.from_pretrained(lang_model)


def webloader(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    return document

def pdfloader(pdf):
    pdf_loader = PyPDFLoader(pdf)
    pdf_document = pdf_loader.load()
    return pdf_document

def qa_chain_web(documents):
    embeddings = HuggingFaceEmbeddings(model_name= "BAAI/bge-base-en-v1.5")
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = textsplitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

def qa_chain_pdf(pdf_documents):
    embeddings = HuggingFaceEmbeddings(model_name= "BAAI/bge-base-en-v1.5")
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = textsplitter.split_documents(pdf_documents)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  

def translate(text, source_lang="hi_IN", target_lang="en_XX"):
    tokenizer.src_lang = source_lang
    encoded = tokenizer(text, return_tensors="pt")
    forced_bos_token_id = tokenizer.lang_code_to_id[target_lang]
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

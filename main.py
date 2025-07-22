from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv()
load_dotenv()


def webloader(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    return document

def qa_chain(documents):
    llm = ChatGroq(model="llama3-8b-8192", api_key=	os.getenv("GROQ_API_KEY"))
    embeddings = HuggingFaceBgeEmbeddings(model_name= "BAAI/bge-base-en-v1.5")
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = textsplitter.split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

url = input("Enter the url: ")
docs = webloader(url)
chain = qa_chain(docs)

while True:
    print('-'*100)
    query = input('Ask a question: ')
    if query.lower() in ['exit', 'quit']:
        break
    result = chain.invoke(query)
    print('Answer: ', result['result'])

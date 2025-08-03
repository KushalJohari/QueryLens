import streamlit as st
from Loaders import webloader, pdfloader, qa_chain_pdf, qa_chain_web
import tempfile

st.title("QueryLens")

with st.expander("Ask Question from webpage", expanded=False):
    url = st.text_input("Enter your url", key='url_input')
    if st.button("Load Url"):
        if url:
            docs = webloader(url)
            st.session_state.chain = qa_chain_web(docs)
    if 'chain' in st.session_state:
        query = st.text_input('Ask a question', key='query_input')
        if query:
            result = st.session_state.chain.invoke(query)
            st.success(f"Answer: {result['result']}")

with st.expander("Ask Question from PDF", expanded=False):
    pdf = st.file_uploader("upload your pdf")
    if pdf and st.button("Process PDF"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            pdf_path = tmp_file.name
        docs = pdfloader(pdf_path)
        st.session_state.pdfchain = qa_chain_pdf(docs)
    if 'pdfchain' in st.session_state:
        query = st.text_input('Ask a question', key='query_input_pdf')
        if query:
            result = st.session_state.pdfchain.invoke(query)
            st.success(f"Answer: {result['result']}")
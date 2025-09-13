import streamlit as st
from langchain_community.document_loaders import  PyPDFLoader
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime 


st.set_page_config(page_title="RAG-based QA bot",layout="wide" )
st.title("PDF Question Answering with RAG")
year = datetime.now().year
footer = f"""
<div style="
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 8px 0;
    background: rgba(255,255,255,0.0); /* transparent bg */
    color: #6b7280;
    font-size: 12px;
    z-index: 9999;
">
    © {year} Muralikrishna
</div>
"""
st.markdown(footer, unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF"):
        pdf_path = f"temp.pdf"
        with open(pdf_path,"wb") as f:
            f.write(uploaded_file.getbuffer())



        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

    spliter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = spliter.split_documents(docs)


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

    def rag_qa(question, top_k=3):
        docs = vectorstore.similarity_search(question, k=top_k)
        if not docs:
            return "No relevant information found in PDF",[]
            
        context = ' '.join(doc.page_content for doc in docs)
        result = qa_pipeline({
            'context' : context,
            'question' : question
        })
        return result["answer"], docs



    query = st.text_input("Ask a question about pdf:")

    if query: 
        with st.spinner("Searching and answering"):
            answer, sources = rag_qa(query)

        st.subheader("Answer")
        st.write(answer)


import os
from pandas import read_orc
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model - Q & A ")

llm = ChatGroq(model_name="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Question: {input} 
"""
)

def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

        st.session_state.loader = PyPDFDirectoryLoader("./census")

        st.session_state.docs = st.session_state.loader.load()
        print(len(st.session_state.docs))
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)

        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:])

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)


inp_prompt = st.text_input("Ask a question related to the documents")

if st.button("Create a vector Store"):
    vector_embedding()
    st.write("Vector Store DB is created")

import time

if inp_prompt:
    document_chain = create_stuff_documents_chain(llm , prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever , document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': inp_prompt})
    st.write(response['answer'])
    
    with st.expander("Document Simliarity"):

        for i , doc in enumerate(response['context']):
            st.write(f"Document {i+1}")
            st.write(doc.page_content)
            st.write("-----------------------------------")

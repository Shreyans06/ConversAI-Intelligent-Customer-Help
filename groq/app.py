import chunk
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv
load_dotenv()

## load Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader =   WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents[:50] , st.session_state.embeddings)


st.title("Groq Chat demo")
llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "gemma2-9b-it")

prompt = ChatPromptTemplate.from_template("""
Answer questions based on the provided context only. Provide the most accurate response with respect to the questions.
<context>
{context}
<context>
Questions: {input}
                                          
""")

document_chain = create_stuff_documents_chain(llm , prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever , document_chain)

prompt = st.text_input("Input the prompt")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt})
    print("Response time:" , time.process_time() - start)
    st.write(response["answer"])

    with st.expander("SHowing context"):
        for i , doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------")

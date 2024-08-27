from pyexpat import model
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.title("Groq Chat demo with Llama3")

llm = ChatGroq(groq_api_key = os.getenv("GROQ_API_KEY"), model_name = "Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
                                        Answer questions based on the provided context only. 
                                        Provide the most accurate response with respect to the questions.
                                        <context>
                                        {context}
                                        <context>
                                        Questions: {input}
                                        """)

def vector_embeddings():
    
    if "vector" not in st.session_state:
        ## Initializing the Ollama embeddings
        st.session_state.embeddings = OllamaEmbeddings()
        ## Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader("../huggingface/census")
        ## Document Loading
        st.session_state.docs = st.session_state.loader.load()
        ## Chunk size and overlap
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800 , chunk_overlap = 200)
        ## Splitting the documents
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        ## Creating the vector store using Ollama embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents , st.session_state.embeddings)

custom_prompt = st.text_input("Enter question related to the document")

if st.button("Documents Embedding"):
    vector_embeddings()
    st.write("Vector store DB is ready")



if custom_prompt:
    document_chain = create_stuff_documents_chain(llm , prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever , document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input':custom_prompt})
    end = time.process_time() - start
    print("Response time:" , end)

    st.write(response["answer"])

    ## Using streamlit expander to show the context
    with st.expander("Showing context"):
        ## Find relevant context
        for i , doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------")


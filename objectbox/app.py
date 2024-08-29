from urllib import response
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, prompt
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
import time

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.title("ObjectBox Vector Store DB with Llama3 demo")

llm = ChatGroq(model_name = "Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following questions based on the context provided. 
    Please provide the most accurate response based on the context
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

## Vector embedding and ObjectBox vector Store DB

def vector_embedding():
    
    if "vectors" not in st.session_state:

        st.session_state.embeddings = OllamaEmbeddings()

        st.session_state.loader = PyPDFDirectoryLoader("./census")

        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)

        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        st.session_state.vectors = ObjectBox.from_documents(st.session_state.final_docs, 
                                                            st.session_state.embeddings , embedding_dimensions = 768)
        

input_prompt = st.text_input("Enter question related to the document")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Document Embedding Done and ObjectBox Vector Store DB created")

if input_prompt:
    document_chain = create_stuff_documents_chain(llm , prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever , document_chain)

    start = time.process_time()

    response = retrieval_chain.invoke({'input': input_prompt})

    print("Response Time:" , time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Show Context"):
        for i , doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------")
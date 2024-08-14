from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

# OPEN AI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# LangSmith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# LangChain API Key
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system" , "Please respond to the user queries"),
    ("user", "Question:{question}")
])

# Initialize the streamlit framework
st.title("Langchain demo with llama2")
input_text = st.text_input("Search for a topic")

# Define the LLM model
llm = Ollama(model="llama2")

# Output parser
output_parser = StrOutputParser()

# Define the chain
chain = prompt | llm | output_parser

# Write the response
if input_text:
    st.write(chain.invoke({"question":input_text}))

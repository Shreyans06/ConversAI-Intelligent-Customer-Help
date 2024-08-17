from fastapi import FastAPI, Request
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import requests
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="LangChain Server", version="1.0", description="API server")

llm = Ollama(model="llama2")

prompt = ChatPromptTemplate.from_template(
    "Write an essay about {topic} with about 100 words."
)

add_routes(app, prompt | llm, path="/essay")

if __name__ == "__main__":

    uvicorn.run(app, host="localhost", port=8000)

import requests
import streamlit as st


def get_llm_response(input_text):

    try:
        # Send a POST request to the LangChain Server
        response = requests.post(
            "http://localhost:8000/essay/invoke", json={"input": {"topic": input_text}}
        )

        # Check if the request was successful
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"An error occured : {e}")
        return None

    except ValueError:
        print("Eroor: Unable to decode JSON response")
        return None

    return response.json()["output"]


st.title("LangChain Demo with llama2 API")
input_text = st.text_input("Enter your essay topic")

if input_text:
    st.write(get_llm_response(input_text))

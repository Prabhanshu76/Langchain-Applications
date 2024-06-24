import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

load_dotenv()

def get_response(question):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    prompt_template = PromptTemplate.from_template(question)
    tweet_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    response = tweet_chain.run(input=question)
    return response

st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

input_text = st.text_input("Input: ")
response = get_response(input_text)

submit_button = st.button("Ask the Question")

if submit_button:
    st.subheader("The Response is")
    st.write(response)

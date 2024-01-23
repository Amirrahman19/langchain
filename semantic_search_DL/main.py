import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db



st.title("Cordlife FAQ 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input("Question: ")

def format(response):
  try:
    out = ''
    for question, answer, score in response:
        out += f'{question}\n{answer}\n(score: {score})\n\n\n'
    return out
  except:
     return response[0]

# if question:
#     chain = get_qa_chain()
#     response = chain(question)

#     st.header("Answer")
#     st.write(format(response))


if question:
    chain = get_qa_chain()
    response = chain({'prompt': question, 'response': ''})  # Pass an empty response initially

    st.header("Answer")
    st.write(format(response))





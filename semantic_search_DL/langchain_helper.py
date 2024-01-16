from langchain.vectorstores import FAISS
# from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
# llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='cordlife_faqs1.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)




import pandas as pd

raw = pd.read_csv('cordlife_faqs1.csv')

lookup = {}
for a, b, *_ in raw.values:
    lookup[a] = b

from sentence_transformers import SentenceTransformer, util

sentences = list(raw['prompt'].values)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sentences_embeddings = [model.encode(s, convert_to_tensor=True) for s in sentences]

def search(q, THRESHOLD=0.65):
    q_embedding = model.encode(q, convert_to_tensor=True)
    out = []
    for sentence, sentence_embedding in zip(sentences, sentences_embeddings):
        score = util.pytorch_cos_sim(q_embedding, sentence_embedding).item()
        out.append((sentence, score))
    out.sort(key=lambda x:-x[-1])
    if out[0][1] < THRESHOLD:
        return ['I don\'t know']
    top = out[:3]
    return [(qns, lookup[qns], score) for qns,score in top]

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return search

    # chain = RetrievalQA.from_chain_type(llm=llm,
    #                                     chain_type="stuff",
    #                                     retriever=retriever,
    #                                     input_key="query",
    #                                     return_source_documents=True,
    #                                     chain_type_kwargs={"prompt": PROMPT})

    # return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("How does one enrol and how much does it cost?"))
import os
import csv
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer, util
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Palm LLM model
# llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
# Initialize instructor embeddings using the Hugging Face model
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

# Load existing data from XLSX file if it exists
search_log_xlsx_file = 'search_logs.xlsx'
xlsx_headers = ['prompt', 'response', 'result_1', 'result_1_info', 'result_2', 'result_2_info', 'result_3', 'result_3_info']
logging.basicConfig(filename='process_logs.txt', level=logging.INFO)

if os.path.exists(search_log_xlsx_file):
    df = pd.read_excel(search_log_xlsx_file)
else:
    df = pd.DataFrame(columns=xlsx_headers)
    df.to_excel(search_log_xlsx_file, index=False)

# Load data from CSV file
raw = pd.read_csv('cordlife_faqs1.csv')

lookup = {}
for a, b, *_ in raw.values:
    lookup[a] = b

sentences = list(raw['prompt'].values)
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
sentences_embeddings = [model.encode(s, convert_to_tensor=True) for s in sentences]

def log_search_to_xlsx(query, result):
    data = [query['prompt'], query['response']] + \
      [f"{qns} ({_}) ({score})" for qns, _, score in result]

    if os.path.exists(search_log_xlsx_file):
        df = pd.read_excel(search_log_xlsx_file)
    else:
        df = pd.DataFrame(columns=xlsx_headers)

    new_data = dict(zip(xlsx_headers, data))
    df = pd.concat([df, pd.DataFrame([new_data], columns=xlsx_headers)], ignore_index=True)

    with pd.ExcelWriter(search_log_xlsx_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, index=False)

from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline

model_ckpt = "papluca/xlm-roberta-base-language-detection"
guess_lang = pipeline("text-classification", model=model_ckpt)

def get_lang(q):
  out = guess_lang([q])
  return out[0]['label']

model_name = 'liam168/trans-opus-mt-en-zh'
model_translate = AutoModelWithLMHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
eng_to_ch = pipeline("translation_en_to_zh", model=model_translate, tokenizer=tokenizer)

model_name_id = 'mesolitica/t5-base-standard-bahasa-cased'
model_translate_id = AutoModelWithLMHead.from_pretrained(model_name_id)
tokenizer = AutoTokenizer.from_pretrained(model_name_id)
eng_to_id = pipeline("translation_en_to_id", model=model_translate_id, tokenizer=tokenizer)

import re

def translate(sentence, lang):
    try:
      map = {'zh':eng_to_ch, 'ru':eng_to_id}
      f = map[lang]
      sentence = re.sub('[^a-zA-Z0-9 ?.!]', '', sentence)
      return f(sentence, max_length=8000)[0]['translation_text']
    except:
      return sentence

from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline

model_ckpt = "papluca/xlm-roberta-base-language-detection"
guess_lang = pipeline("text-classification", model=model_ckpt)

def get_lang(q):
  out = guess_lang([q])
  return out[0]['label']

def search(query, THRESHOLD=0.65):
    q = query['prompt']
    q_embedding = model.encode(q, convert_to_tensor=True)
    out = []
    for sentence, sentence_embedding in zip(sentences, sentences_embeddings):
        score = util.pytorch_cos_sim(q_embedding, sentence_embedding).item()
        out.append((sentence, score))
    out.sort(key=lambda x:-x[-1])
    if out[0][1] < THRESHOLD:
        return ['I don\'t know']
    top = out[:3]

    lang = get_lang(q)

    if lang == 'zh' or lang=='ru':
      result = [(translate(qns, lang), translate(lookup[qns], lang), score) for qns,score in top]
    else:
      result = [(qns, lookup[qns], score) for qns,score in top]
        
    # Log the search query and result to XLSX
    log_search_to_xlsx(query, result)

    # Log the search query and result to the text file
    logging.info(f"Search Query: {query}")
    logging.info(f"Search Result: {result}")
    
    return result

# def search(query, THRESHOLD=0.65):
#     q = query['prompt']
#     q_embedding = model.encode(q, convert_to_tensor=True)
#     out = []
#     for sentence, sentence_embedding in zip(sentences, sentences_embeddings):
#         score = util.pytorch_cos_sim(q_embedding, sentence_embedding).item()
#         out.append((sentence, score))
#     out.sort(key=lambda x:-x[-1])
#     if out[0][1] < THRESHOLD:
#         return ['I don\'t know']
#     top = out[:3]

#     lang = get_lang(q)
#     print(lang)
#     if lang == 'zh':
#       result = [(translate_ch(qns), translate_ch(lookup[qns]), score) for qns,score in top]
#     else:
#       result = [(qns, lookup[qns], score) for qns,score in top]
        
#     # Log the search query and result to XLSX
#     log_search_to_xlsx(query, result)

#     # Log the search query and result to the text file
#     logging.info(f"Search Query: {query}")
#     logging.info(f"Search Result: {result}")
    
#     return result


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

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain({"prompt": "How does one enroll and how much does it cost?", "response": ""}))

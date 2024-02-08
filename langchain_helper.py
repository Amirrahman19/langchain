import re
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
import pandas as pd
import os
import logging
import fitz  # PyMuPDF for PDF handling

import logging

# Configure the root logger
logging.basicConfig(filename='logs.txt', level=logging.INFO)
logger = logging.getLogger()

# Define a StreamHandler to display logs in Streamlit's app UI
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

# Initialize transformers pipelines and models
model_ckpt = "papluca/xlm-roberta-base-language-detection"
guess_lang = pipeline("text-classification", model=model_ckpt)

model_name = 'liam168/trans-opus-mt-en-zh'
model_translate = AutoModelWithLMHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
eng_to_ch = pipeline("translation_en_to_zh", model=model_translate, tokenizer=tokenizer)

model_name_id = 'mesolitica/t5-base-standard-bahasa-cased'
model_translate_id = AutoModelWithLMHead.from_pretrained(model_name_id)
tokenizer = AutoTokenizer.from_pretrained(model_name_id)
eng_to_id = pipeline("translation_en_to_id", model=model_translate_id, tokenizer=tokenizer)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

model_ch_to_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
ch_to_en = pipeline('translation_ch_to_en', model=model_ch_to_en, tokenizer=tokenizer)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-id-en")

model_id_to_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-id-en")
id_to_en = pipeline('translation_id_to_en', model=model_id_to_en, tokenizer=tokenizer)


# Function to translate text
def translate(text, target_lang):

  lang = guess_lang(text)[0]['label']
  if lang == 'zh':
    new_text = ch_to_en(text)[0]["translation_text"]
  elif lang == 'id':
    new_text =  id_to_en(text)[0]["translation_text"]
  else:
    new_text = text

  print(new_text)

  if target_lang == 'zh':
    output = eng_to_ch(new_text)
  elif target_lang == 'id':
    output = eng_to_id(new_text)
  else:
    if lang != 'en':
       return id_to_en(new_text)[0]['translation_text']
    return new_text

  return output[0]['translation_text']

# Function to log translation into an Excel file
import pandas as pd
import os

def log_translation(input_text, output_text, log_file_path='translation_logs.xlsx'):
    try:
        log_data = {'Input': [input_text], 'Output': [output_text]}
        df = pd.DataFrame(log_data)

        if os.path.exists(log_file_path):
            # Load existing data from the log file
            existing_df = pd.read_excel(log_file_path)
            # Append new data to the existing data
            df = pd.concat([existing_df, df], ignore_index=True)

        # Save the combined data to the log file
        df.to_excel(log_file_path, index=False, mode='w', header=True)  # mode='w' to overwrite the existing file
    except Exception as e:
        print(f"Error logging translation: {e}")



# Function to translate PDF
def translate_pdf(pdf_file, target_lang):
    try:
        doc = fitz.open(pdf_file)
        pdf_text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            translated_page_text = translate(page.get_text(), target_lang)
            pdf_text += translated_page_text
            print(translated_page_text)

        return pdf_text

    except Exception as e:
        print(f"Error translating PDF: {e}")
        return None

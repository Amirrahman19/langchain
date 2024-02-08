import streamlit as st
from langchain_helper import translate, log_translation, translate_pdf
import logging
import pandas as pd
import os
import base64
import logging
import streamlit as st

# Create a custom logger
logger = logging.getLogger("translation_logger")
logger.setLevel(logging.INFO)

# Create a file handler
log_file_path = 'logs.txt'
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# Create a formatter and set the formatter for the handler
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Function to log translation into both Streamlit UI and a file
def log_translation(input_text, output_text):
    # Display in Streamlit UI
    # st.text(f"Input: {input_text}")
    # st.text(f"Output: {output_text}")
    # st.text("-------------------------")

    # Log to the file
    logger.info(f"Input: {input_text}")
    logger.info(f"Output: {output_text}")
    logger.info("-------------------------")


# Create folders to store uploaded and translated files
upload_folder = "uploaded_files"
translated_folder = "translated_files"
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(translated_folder, exist_ok=True)

st.title("Translation App 🌐")

lang_map = {'English': 'en', 'Chinese': 'zh', 'Indonesian': 'id'}

# Language selection
target_lang = st.selectbox("Select Target Language:", ["English", "Chinese", "Indonesian"])
target_lang = lang_map[target_lang]
# Text input
text_input = st.text_area("Enter Text:", "")
# Translate button for text
if st.button("Translate Text"):
    if text_input:
        try:
            text_translation = translate(text_input, target_lang)
            st.write(f"Translated Text:\n{text_translation}")
            # Log translation
            log_translation(text_input, text_translation)
        except Exception as e:
            st.error(f"Error translating text: {e}")
    else:
        st.warning("Please enter text to translate.")

# PDF file upload
pdf_file = st.file_uploader("Upload PDF file:", type=["pdf"])
# Translate button for PDF
if st.button("Translate PDF"):
    if pdf_file:
        try:
            pdf_path = os.path.join(upload_folder, pdf_file.name)
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.getvalue())
            pdf_translation = translate_pdf(pdf_path, target_lang)
            st.write(f"Translated PDF:\n{pdf_translation}")

            # Save translated PDF
            translated_pdf_path = os.path.join(translated_folder, f"translated_{pdf_file.name}")
            with open(translated_pdf_path, "w", encoding="utf-8") as translated_file:
                translated_file.write(pdf_translation)

            # Download button for PDF
            download_button = st.button("Download Translated PDF")
            if download_button:
                st.markdown(get_binary_file_downloader_html(translated_pdf_path, 'Translated_PDF'), unsafe_allow_html=True)

            # Log translation
            log_translation(pdf_file.name, pdf_translation)
        except Exception as e:
            st.error(f"Error translating PDF: {e}")
    else:
        st.warning("Please upload a PDF file to translate.")



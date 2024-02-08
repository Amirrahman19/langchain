
# Moms Up Translation platform Based on Deep Learning Models  

This is an end to end LLM project based on Multilingual Deep Learning Models. We are building a translation platform for Moms Up in Cordlife SG.

## Project Highlights

- Choose target language to translate to and input text in platform to get translated text
- Their human staff can verify the content of the translated text
- This platform is built reduce the workload of the human staff.
- Users should be able to use this system to translate their text and get the output within seconds

## Tech stack includes,
  - Sentence Transformers and Translation models
  - Streamlit: UI
  - Huggingface instructor embeddings: Text embeddings

## Installation

1.Clone this repository to your local machine using:

```bash
  git clone git@gitlab.com:cgl-digitalmarketing/faq_nlp_searchengine.git
```
2.Navigate to the project directory:

```bash
  cd faq_nlp_searchengine
```
3. Switch branch to v3

```bash
  git checkout v3
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

## Project Structure

- main.py: The main Streamlit application script.
- langchain_helper.py: This has all the langchain code
- requirements.txt: A list of required Python packages for the project.
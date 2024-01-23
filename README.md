
# Cordlife Q&A: Question and Answer System Based on Semantic Search with Deep Learning Models  

This is an end to end LLM project based on Semantic Search and Sentence Transformers. We are building a Q&A system for an Cordlife SG.  

## Project Highlights

- Use a real CSV file of FAQs that Cordlife company is using right now. 
- Their human staff will use this file to assist their customers.
- We will build an LLM based question and answer system that can reduce the workload of their human staff.
- Customersshould be able to use this system to ask questions directly and get answers within seconds

## You will learn following,
  - Semantic Search + Sentence Transformers: LLM based Q&A
  - Streamlit: UI
  - Huggingface instructor embeddings: Text embeddings
  - FAISS: Vector databse

## Installation

1.Clone this repository to your local machine using:

```bash
  git clone git@gitlab.com:cgl-digitalmarketing/faq_nlp_searchengine.git
```
2.Navigate to the project directory:

```bash
  cd faq_nlp_searchengine
```
3. Install the required dependencies using pip:

```bash
  pip install -r requirements.txt
```
4.Acquire an api key through makersuite.google.com and put it in .env file

```bash
  GOOGLE_API_KEY="your_api_key_here"
```

5. Switch to v2 branch
```bash
  git checkout v2 
```
## Usage

1. Run the Streamlit app by executing:
```bash
streamlit run main.py

```

2.The web app will open in your browser.

- To create a knowledebase of FAQs, click on Create Knolwedge Base button. It will take some time before knowledgebase is created so please wait.

- Once knowledge base is created you will see a directory called faiss_index in your current folder

- Now you are ready to ask questions. Type your question in Question box and hit Enter


## Project Structure

- main.py: The main Streamlit application script.
- langchain_helper.py: This has all the langchain code
- requirements.txt: A list of required Python packages for the project.
- .env: Configuration file for storing your Google API key.
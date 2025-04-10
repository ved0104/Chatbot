import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from rank_bm25 import BM25Okapi
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
import os
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import pydantic
pydantic.version.VERSION = "1.8.2"
# Data loading function with caching
@st.cache_data
def load_data():
    data_dir = "MedQuAD"
    qa_pairs = []

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.xml'):
                file_path = os.path.join(root, filename)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    qa_section = root_elem.find('QAPairs')
                    
                    if qa_section is not None:
                        for qa in qa_section.findall('QAPair'):
                            question = qa.find('Question')
                            answer = qa.find('Answer')
                            
                            if question is not None and answer is not None:
                                q_text = question.text.strip() if question.text else ""
                                a_text = answer.text.strip() if answer.text else ""
                                
                                if q_text and a_text:
                                    qa_pairs.append({
                                        'question': q_text,
                                        'answer': a_text
                                    })
                except Exception as e:
                    continue
    df = pd.DataFrame(qa_pairs)
    df['processed_question'] = df['question'].apply(preprocess)
    return df
def calculate_metrics(df):
    test_fraction = 0.2
    test_indices = df.sample(frac=test_fraction, random_state=42).index
    answer_groups = df.groupby('answer').apply(lambda x: x.index.tolist())
    
    accuracy = 0.0
    mrr = 0.0
    total = len(test_indices)
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['processed_question'])

    tokenized_corpus = [doc.split() for doc in df['processed_question']]
    bm25 = BM25Okapi(tokenized_corpus)

    for idx in test_indices:
        test_question = df.loc[idx, 'question']
        test_answer = df.loc[idx, 'answer']
        
        processed_test = preprocess(test_question)
        tokenized_query = processed_test.split()
        scores = bm25.get_scores(tokenized_query)
        scores[idx] = -1  # Exclude current question
        query_vec = vectorizer.transform([processed_test])
        query_vec = vectorizer.transform([preprocess(test_question)])
        similarities = cosine_similarity(query_vec, X)
        similarities[0, idx] = -1
        
        most_similar_idx = similarities.argmax()
        retrieved_answer = df.loc[most_similar_idx, 'answer']
        
        accuracy += 1 if retrieved_answer == test_answer else 0
        
        correct_indices = [i for i in answer_groups.get(test_answer, []) if i != idx]
        if correct_indices:
            sorted_indices = np.argsort(similarities[0])[::-1]
            for rank, sorted_idx in enumerate(sorted_indices, 1):
                if sorted_idx in correct_indices:
                    mrr += 1.0 / rank
                    break
    
    return accuracy/len(test_indices), mrr/len(test_indices)

# Load medical NER model with error handling
@st.cache_resource
def load_ner_model():
    try:
        # Load the medical model
        return spacy.load("en_ner_bc5cdr_md")
    except OSError:
        st.error(
            "Medical NER model not found. Install with:\n"
            "pip install scispacy\n"
            "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz"
        )
        return None
# Initialize application components
def initialize_app():
    df = load_data()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['question'])
    nlp = load_ner_model()
    return df, vectorizer, X, nlp

# Medical entity recognition function
def extract_medical_entities(text, nlp):
    if nlp is None:
        return []
    
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            entities.append(("Disease", ent.text))
        elif ent.label_ == "CHEMICAL":
            entities.append(("Treatment", ent.text))
    return entities

# Streamlit UI components
def main():
    st.title("Medical Q&A Chatbot 🩺")
    
    # Initialize components
    df, vectorizer, X, nlp = initialize_app()
    with st.sidebar:
        st.header("Performance Metrics")
        accuracy, mrr = calculate_metrics(df)
        st.metric("Top-1 Accuracy", f"{accuracy:.2%}")
        st.metric("Mean Reciprocal Rank", f"{mrr:.4f}")
    # User input
    user_question = st.text_input("Ask a medical question:")
    
    if st.button("Get Answer"):
        if not user_question.strip():
            st.error("Please enter a question.")
            return
            
        try:
            processed_query = preprocess(user_question)
            query_vec = vectorizer.transform([processed_query])
            similarities = cosine_similarity(query_vec, X)
            max_index = similarities.argmax()
            answer = df.iloc[max_index]['answer']

            # Display results
            st.subheader("Answer:")
            st.write(answer)

            # Show entities
            entities = extract_medical_entities(user_question, nlp)
            if entities:
                st.subheader("Recognized Medical Entities:")
                for entity_type, entity_text in entities:
                    st.write(f"**{entity_type}**: {entity_text}")
            else:
                st.info("No medical entities recognized in the question.")

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

if __name__ == "__main__":
    main()
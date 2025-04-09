import os
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
    return pd.DataFrame(qa_pairs)

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
    
    # User input
    user_question = st.text_input("Ask a medical question:")
    
    if st.button("Get Answer"):
        if not user_question.strip():
            st.error("Please enter a question.")
            return
            
        try:
            # Find similar questions
            query_vec = vectorizer.transform([user_question])
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
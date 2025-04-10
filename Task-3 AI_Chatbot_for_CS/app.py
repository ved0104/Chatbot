# Add to imports
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sys

# Nuclear option - override NLTK's resource paths
nltk.data.path = [
    "C:/nltk_data",  # Your custom path
    "/usr/share/nltk_data",  # Linux fallback
    "/usr/local/share/nltk_data",  # Mac/Linux fallback
    "/usr/lib/nltk_data",  # Linux fallback
    "/usr/local/lib/nltk_data",  # Mac/Linux fallback
    sys.prefix + "/nltk_data"  # Virtual env path
]

# Force download ALL resources with atomic checks
resources = {
    'punkt': 'tokenizers/punkt',
    'wordnet': 'corpora/wordnet',
    'stopwords': 'corpora/stopwords'
}

for res_name, res_path in resources.items():
    try:
        nltk.data.find(res_path)
    except LookupError:
        print(f"Downloading {res_name}...")
        nltk.download(res_name, force=True, download_dir="C:/nltk_data")
        nltk.data.path.append("C:/nltk_data")
        

import streamlit as st
from streamlit_chat import message
import pandas as pd
import numpy as np
import spacy
import faiss
import ast
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rank_bm25 import BM25Okapi
import sys
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Set custom NLTK data path before any other imports
nltk.data.path.append("C:/nltk_data")  # Create this folder manually
print("NLTK Version:", nltk.__version__)
print("Data Paths:", nltk.data.path)
print("Punkt Exists:", nltk.data.find('tokenizers/punkt'))
# Force downloads to custom path
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', force=True, download_dir="C:/nltk_data")

nltk.download('wordnet', force=True, download_dir="C:/nltk_data")
nltk.download('stopwords', force=True, download_dir="C:/nltk_data")
print("NLTK Data Paths:", nltk.data.path)
print("Punkt exists:", nltk.data.find('tokenizers/punkt'))

# Rest of your imports
import streamlit as st
from streamlit_chat import message
import pandas as pd
# ... other imports ...

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Define global variables for text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._classes")

# Add before any torch imports for Windows compatibility
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ----------------------
# Data Loading
# ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('src//arxiv_cs_processed.csv')
    df['authors'] = df['authors'].apply(ast.literal_eval)
    return df

# ----------------------
# Model Setup
# ----------------------
@st.cache_resource
def setup_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device="cpu"
    )
    return embedding_model, generator

# ----------------------
# Search Systems
# ----------------------
@st.cache_resource
def setup_search_systems(_df, _embedding_model):
    tokenized_corpus = [doc.split() for doc in _df['processed_abstract']]
    bm25 = BM25Okapi(tokenized_corpus)
    
    paper_embeddings = _embedding_model.encode(_df['abstract'].tolist())
    dimension = paper_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(paper_embeddings.astype('float32'))
    
    return bm25, index

# ----------------------
# Core Functions
# ----------------------
def preprocess_query(text):
    # Verify punkt exists
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError as e:
        raise RuntimeError("Punkt tokenizer missing!") from e

    # Rest of your preprocessing code
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def hybrid_search(query, _df, bm25, index, embedding_model, k=3, alpha=0.5):
    tokenized_query = preprocess_query(query)
    query_embedding = embedding_model.encode([query])
    
    # Semantic search
    semantic_distances, semantic_indices = index.search(query_embedding.astype('float32'), k*2)
    
    # Keyword search
    keyword_scores = bm25.get_scores(tokenized_query)
    keyword_indices = np.argsort(keyword_scores)[-k*2:][::-1]
    
    # Combine results
    combined_indices = list(set(semantic_indices[0]).union(set(keyword_indices)))
    combined_scores = []
    
    max_semantic = np.max(semantic_distances) if semantic_distances.any() else 1
    max_keyword = np.max(keyword_scores) if keyword_scores.any() else 1
    
    for idx in combined_indices:
        semantic_score = 0
        if idx in semantic_indices[0]:
            pos = list(semantic_indices[0]).index(idx)
            semantic_score = 1 - (semantic_distances[0][pos] / max_semantic)
            
        keyword_score = keyword_scores[idx] / max_keyword if max_keyword > 0 else 0
        combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
        combined_scores.append(combined_score)
    
    sorted_indices = [idx for _, idx in sorted(zip(combined_scores, combined_indices), reverse=True)]
    return _df.iloc[sorted_indices[:k]]
# Add this function after the hybrid_search function
def evaluate_model(df, bm25, index, embedding_model, eval_path='evaluation_questions.csv'):
    eval_df = pd.read_csv(eval_path)
    eval_df['expected_paper_ids'] = eval_df['expected_paper_ids'].apply(ast.literal_eval)
    
    y_true = []
    y_pred = []
    
    for _, row in eval_df.iterrows():
        # Get actual relevant papers
        true_ids = set(row['expected_paper_ids'])
        retrieved_papers = hybrid_search(row['question'], df, bm25, index, embedding_model, k=5)
        retrieved_ids = set(retrieved_papers['id'].tolist())
        
        # Create binary relevance vectors
        all_ids = list(true_ids.union(retrieved_ids))
        true_vector = [1 if i in true_ids else 0 for i in all_ids]
        pred_vector = [1 if i in retrieved_ids else 0 for i in all_ids]  # Fixed line
        
        y_true.extend(true_vector)
        y_pred.extend(pred_vector)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return precision, recall, conf_matrix
def generate_response(query, context_papers, generator):
    context = ""
    max_context_length = 2000
    current_length = 0
    
    for _, row in context_papers.iterrows():
        paper_context = f"Title: {row['title']}\nAbstract: {row['abstract']}\n"
        paper_length = len(paper_context)
        if current_length + paper_length > max_context_length:
            break
        context += paper_context
        current_length += paper_length
    
    prompt = f"""Answer this CS research question based on the context:
    
    Context: {context}
    
    Question: {query}
    
    Answer concisely:"""
    
    try:
        return generator(
            prompt, 
            max_length=500,
            truncation=True,
            temperature=0.7,
            repetition_penalty=1.2
        )[0]['generated_text']
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ----------------------
# Streamlit UI
# ----------------------
def main():
    st.set_page_config(page_title="arXiv CS Expert Chatbot", layout="wide")
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Load data and models
    df = load_data()
    embedding_model, generator = setup_models()
    bm25, faiss_index = setup_search_systems(df, embedding_model)
    
    # Sidebar
    st.sidebar.title("Navigation")
    menu_choice = st.sidebar.radio("Select Feature", ["Chat", "Paper Search", "Concept Analysis"])
    if st.sidebar.button('Run Evaluation'):
        with st.spinner("Running comprehensive evaluation..."):
            precision, recall, conf_matrix = evaluate_model(df, bm25, faiss_index, embedding_model)
            
            st.subheader("Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Precision", f"{precision:.2%}")
                
            with col2:
                st.metric("Recall", f"{recall:.2%}")
                
            with col3:
                st.metric("F1-Score", f"{2*(precision*recall)/(precision+recall):.2%}")
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax,
                        xticklabels=['Irrelevant', 'Relevant'],
                        yticklabels=['Irrelevant', 'Relevant'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
    # Main Content
    st.title("arXiv Computer Science Expert Chatbot")
    
    if menu_choice == "Chat":
        user_input = st.text_input("Ask your computer science research question:", key="input")
        
        if user_input:
            with st.spinner("Searching papers and generating response..."):
                try:
                    relevant_papers = hybrid_search(user_input, df, bm25, faiss_index, embedding_model)
                    response = generate_response(user_input, relevant_papers, generator)
                    references = "\n".join(
                        [f"{i+1}. {row['title']} [PDF]({row['pdf_url']})" 
                         for i, (_, row) in enumerate(relevant_papers.iterrows())]
                    )
                    full_response = f"{response}\n\n**References:**\n{references}"
                    
                    st.session_state.history.append(("user", user_input))
                    st.session_state.history.append(("bot", full_response))
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
        
        # Display chat history
        if st.session_state.history:
            for idx, (role, msg) in enumerate(reversed(st.session_state.history)):
                if role == "user":
                    message(msg, is_user=True, key=f"{idx}_user")
                else:
                    message(msg, key=str(idx))
                    with st.expander("View References"):
                        st.markdown(msg.split("**References:**")[1])
        else:
            st.info("No conversation history yet. Ask a question to get started!")

    elif menu_choice == "Paper Search":
        st.header("Paper Search")
        search_query = st.text_input("Enter search keywords:")
        
        if search_query:
            processed_query = ' '.join(preprocess_query(search_query))
            results = df[df['processed_abstract'].str.contains(processed_query)]
            
            if not results.empty:
                st.subheader(f"Found {len(results)} papers:")
                for _, row in results.head(10).iterrows():
                    st.markdown(f"### {row['title']}")
                    st.write(f"**Authors:** {', '.join(row['authors'])}")
                    st.write(f"**Published:** {row['published']}")
                    st.write(f"**Abstract:** {row['abstract'][:500]}...")
                    st.markdown(f"[PDF Link]({row['pdf_url']})")
                    st.write("---")
            else:
                st.warning("No papers found matching your query.")

    elif menu_choice == "Concept Analysis":
        st.header("Concept Analysis")
        concept = st.text_input("Enter a concept to analyze:")
        
        if concept:
            processed_concept = ' '.join(preprocess_query(concept))
            concept_papers = df[df['processed_abstract'].str.contains(processed_concept)]
            
            if not concept_papers.empty:
                st.subheader(f"Analysis of '{concept}'")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Papers", len(concept_papers))
                    st.write("**Top Authors:**")
                    authors = [a for sublist in concept_papers['authors'] for a in sublist]
                    author_counts = pd.Series(authors).value_counts().head(5)
                    for author, count in author_counts.items():
                        st.write(f"- {author} ({count} papers)")
                
                with col2:
                    st.write("**Trend Over Time:**")
                    timeline = concept_papers.groupby(pd.to_datetime(concept_papers['published']).dt.year).size()
                    st.line_chart(timeline)
                
                st.write("**Related Concepts:**")
                all_words = ' '.join(concept_papers['processed_abstract']).split()
                word_counts = pd.Series(all_words).value_counts().head(10)
                st.bar_chart(word_counts)
            else:
                st.warning("Concept not found in any paper abstracts.")


if __name__ == "__main__":
    main()

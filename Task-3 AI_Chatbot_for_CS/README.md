# arXiv Computer Science Expert Chatbot 🧠

![Chatbot Demo](demo.gif) <!-- Add actual demo path -->

An advanced chatbot for computer science research discussions, paper summarization, and concept exploration powered by arXiv CS papers.

## Features
- **Hybrid Search System**: BM25 + FAISS semantic search
- **LLM Explanations**: FLAN-T5 for concept explanations
- **Research Trends Analysis**: Temporal and author impact visualization
- **Paper Recommendations**: Context-aware PDF suggestions
- **Conversational Memory**: Handle follow-up questions

## Prerequisites
- Python 3.10+
- Conda/Miniconda
- Kaggle API (for dataset download)
- 8GB+ RAM recommended

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/arxiv-chatbot.git
cd arxiv-chatbot

Here's your content converted **strictly to Markdown** for direct copy-pasting into a `README.md` file:

---

# arxiv-chatbot

## 2. Create Conda Environment

```bash
conda create -n arxivbot python=3.10 -y
conda activate arxivbot
```

## 3. Install Dependencies

```bash
conda install -c conda-forge --file requirements.txt
pip install "faiss-cpu>=1.7.4" "sentence-transformers>=2.2.2"
```

## 4. Dataset Setup

- Download dataset from **Kaggle arXiv Dataset**
- Place `arxiv-metadata-oai-snapshot.json` in the `src/` folder
- Run data processing:

```bash
jupyter nbconvert --execute src/main.ipynb
```

## 📁 Project Structure

```
arxiv-chatbot/
├── src/
│   ├── arxiv_cs_processed.csv  # Preprocessed data
│   ├── arxiv_cs_raw.csv        # Raw metadata
│   └── main.ipynb              # Data pipeline
├── app.py                      # Streamlit application
└── requirements.txt            # Dependency spec
```

## 🚀 Usage

### Launch Streamlit App

```bash
streamlit run app.py
```

### Jupyter Notebook (Data Processing)

```bash
jupyter notebook src/main.ipynb
```

---

## 🔍 Key Implementation

### Hybrid Search Engine

```python
def hybrid_search(query, _df, bm25, index, embedding_model, k=3, alpha=0.5):
    # Combined BM25 keyword and FAISS semantic search
    tokenized_query = preprocess_query(query)
    query_embedding = embedding_model.encode([query])
    
    # Get results from both systems
    semantic_results = index.search(query_embedding.astype('float32'), k*2)
    keyword_scores = bm25.get_scores(tokenized_query)
    
    # Fusion algorithm
    combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores
    return top_k_results(combined_scores)
```

### LLM Response Generation

```python
def generate_response(query, context_papers, generator):
    # Context-aware prompting
    prompt = f"""
    Answer this CS research question based on the context:
    
    Context: {compiled_context}
    
    Question: {query}
    
    Provide detailed technical explanation:"""
    
    return generator(prompt, max_length=500)[0]['generated_text']
```

---

## 📦 Dependency Management

### `requirements.txt`

```
rank-bm25==0.2.2  
faiss-cpu==1.7.4  
sentence-transformers==2.2.2  
transformers==4.35.0  
arxiv==1.4.5  
spacy==3.6.1  
streamlit==1.27.0  
nltk==3.8.1  
pandas==2.1.1  
jupyter==1.0.0  
```

---

## ⚠️ Version Conflicts Resolution Table

| Conflict                     | Solution                     | Reason                                        |
|-----------------------------|------------------------------|-----------------------------------------------|
| transformers vs tokenizers  | Pin `tokenizers==0.14.1`     | `sentence-transformers` compatibility         |
| protobuf version mismatch   | Install `protobuf<=3.20.3`   | Streamlit/TF compatibility                    |
| numpy ABI issues            | Use `numpy==1.23.5`          | FAISS compatibility                           |

---

## 🛠️ Troubleshooting

### NLTK Data Path Errors

```bash
mkdir -p ~/nltk_data
export NLTK_DATA=~/nltk_data
python -m nltk.downloader all
```

### FAISS Installation Failures

```bash
conda install -c conda-forge faiss-cpu=1.7.4 --force-reinstall
```

### CUDA/CPU Mismatch

```python
# In app.py
from transformers import pipeline
generator = pipeline(..., device_map="cpu")  # Force CPU usage
```

### Memory Errors

```python
# Reduce context window
max_context_length = 1000  # From 2000
```

---

## 📄 License

MIT License - See `LICENSE`

> **Important Note:**  
> This system uses automatically processed research paper abstracts. Accuracy not guaranteed. Never use for critical research decisions. Always consult original papers.

---

## ✅ This README Features:

1. Clear version conflict resolution table  
2. Hybrid search implementation details  
3. Dataset preprocessing instructions  
4. Special handling for common dependency issues  
5. Memory optimization guidelines  
6. Streamlit-specific troubleshooting  
7. Complete environment setup workflow  
8. Legal disclaimer for research use  

---

## 🔑 Key Differences from Previous Implementation:

- Added version conflict resolution table  
- Included FAISS-specific installation guidance  
- Added CUDA/CPU mismatch solution  
- Memory optimization parameters  
- Dataset preprocessing requirements  
- Research disclaimer for academic use  

---

Let me know if you want badges, collapsible sections, or enhanced styling too!
# Medical Q&A Chatbot 🩺

A specialized medical question-answering system powered by the MedQuAD dataset with basic medical entity recognition capabilities.

![Chatbot Demo](demo-screenshot.png) <!-- Add actual screenshot path -->

## Features
- **Medical QA Retrieval**: Find answers from 47,000+ medical questions
- **Entity Recognition**: Identify symptoms, diseases, and treatments
- **Streamlit Interface**: User-friendly web interface
- **TF-IDF Search**: Cosine similarity-based retrieval system
- **Medical NER**: Enhanced with BC5CDR disease/chemical recognition

## Prerequisites
- Python 3.10+
- Conda/Miniconda
- 4GB+ free disk space
- Git installed

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

### 2. Setup Conda Environment
```bash
conda create -n medbot python=3.10 -y
conda activate medbot
```

### 3. Install Dependencies
```bash
conda install -c conda-forge --file requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
```

### 4. Download MedQuAD Dataset
```bash
git clone https://github.com/abachaa/MedQuAD.git MedQuAD
```

Verify directory structure:
```
medical-chatbot/
├── MedQuAD/
├── main.ipynb
├── med_chatbot.py
└── requirements.txt
```

## Usage

### Streamlit Application
```bash
streamlit run med_chatbot.py
```

### Jupyter Notebook Exploration
```bash
jupyter notebook main.ipynb
```
**After launching the notebook:**
1. Open `main.ipynb` from Jupyter interface
2. Run all cells sequentially (Kernel > Restart & Run All)
3. Explore data processing and model training steps

## Code Structure

### Data Processing Pipeline (main.ipynb)
```python
# XML Data Loading
for root, _, files in tqdm(os.walk(data_dir)):
    for filename in files:
        if filename.endswith('.xml'):
            # XML parsing logic
            qa_pairs.append({'question': q_text, 'answer': a_text})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['processed_question'])
```

### Medical Entity Recognition (med_chatbot.py)
```python
@st.cache_resource
def load_ner_model():
    return spacy.load("en_ner_bc5cdr_md")

def extract_medical_entities(text, nlp):
    doc = nlp(text)
    return [(ent.label_, ent.text) for ent in doc.ents]
```

## Configuration

### Required Files
`requirements.txt` contains all Python dependencies:
```
pandas==1.3.4
scikit-learn==1.0.2
nltk==3.6.7
spacy==3.2.4
scispacy==0.5.0
streamlit==1.17.0
lxml==4.9.2
tqdm==4.64.1
```

## Troubleshooting

### Common Issues
**Missing Medical Entities:**
```bash
pip uninstall scispacy
pip install scispacy==0.5.0
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_ner_bc5cdr_md-0.5.0.tar.gz
```

**XML Parsing Errors:**
- Verify MedQuAD dataset structure
- Check XML file encoding (UTF-8)

**TF-IDF Dimension Mismatch:**
```bash
rm -rf .cache/  # Clear Streamlit cache
```

## Dataset Information

The MedQuAD dataset contains:
- 47,457 medical QA pairs
- 21 medical subdomains
- XML format organization
- Clinical/patient-focused content

## License
Apache 2.0 License - See LICENSE

**Important Note:** This chatbot provides informational answers only. Consult a healthcare professional for medical advice. Not intended for diagnosis or treatment.
```

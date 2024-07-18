from sentence_transformers import SentenceTransformer


# This function inits a new BERT model
def setup_sbert():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


# This function extracts embeddings from the data
def extract_embeddings(texts, model):
    return model.encode(texts, show_progress_bar=True, convert_to_tensor=True)

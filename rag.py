from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAG:
    def __init__(self, model_name, chunk_size, overlap):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.texts = []
        self.index = None

    def load_document(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.texts = self._chunk_text(text)
        if not self.texts:
            print("Atenção: documento vazio ou chunks não gerados!")

    def _chunk_text(self, text):
        chunks = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(text), step):
            chunks.append(text[i:i+self.chunk_size])
        return chunks

    def create_index(self):
        embeddings = self.model.encode(self.texts)
        embeddings = np.array(embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=1):
        q_emb = self.model.encode([query])
        q_emb = np.array(q_emb)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        D, I = self.index.search(q_emb, top_k)
        return [self.texts[i] for i in I[0]]

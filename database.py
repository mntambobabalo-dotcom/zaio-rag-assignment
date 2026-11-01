from dataclasses import asdict
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import re

import faiss
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import FeatureUnion
from scipy.sparse import csr_matrix

from read_pdf import extract_chunks, Chunk


@lru_cache(maxsize=2)
def sentence_model(name):
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)
    return SentenceTransformer(name, local_files_only=True, device='cpu')


class SentenceEncoder:
    def __init__(self, model):
        self.model = model

    def transform(self, texts):
        vectors = sentence_model(self.model).encode(texts, normalize_embeddings=True,
                                                    show_progress_bar=False)
        return csr_matrix(np.asarray(vectors, dtype=np.float32))

    fit_transform = transform


class KnowledgeBase:
    def __init__(self, chunks, vectorizer, index, manifest):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.index = index
        self.manifest = manifest

    @classmethod
    def build(cls, pdf: Path, directory: Path, backend='semantic'):
        chunks = extract_chunks(pdf)
        # Lightweight, reproducible lexical embeddings, with no model download.
        # These are TF-IDF vectors, not pretrained neural semantic embeddings.
        lexical = FeatureUnion([
            ('words', TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                     sublinear_tf=True, max_features=12000,
                                     dtype=np.float32)),
            ('characters', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                          max_features=16000, dtype=np.float32))
        ], transformer_weights={'words': 1.0, 'characters': 0.65})
        model_name = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        if backend not in {'semantic', 'tfidf'}:
            raise ValueError('Embedding backend must be semantic or tfidf.')
        vectorizer = SentenceEncoder(model_name) if backend == 'semantic' else lexical
        vectors = np.ascontiguousarray(vectorizer.fit_transform(
            [chunk.text for chunk in chunks]).toarray(), dtype=np.float32)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        manifest = {'schema': 2, 'embedding': model_name if backend == 'semantic' else 'TF-IDF word and character n-grams',
                    'embedding_backend': backend,
                    'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    'chunks': len(chunks), 'dimension': index.d,
                    'page_numbering': '1-based physical PDF pages'}
        directory.mkdir(parents=True, exist_ok=True)
        if backend == 'tfidf':
            joblib.dump(vectorizer, directory / 'vectorizer.joblib')
        faiss.write_index(index, str(directory / 'vectors.faiss'))
        (directory / 'chunks.json').write_text(json.dumps(
            [asdict(c) for c in chunks], indent=2), encoding='utf-8')
        (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        return cls(chunks, vectorizer, index, manifest)

    @classmethod
    def load(cls, directory: Path):
        # Only load indexes generated locally: joblib must not load untrusted files.
        chunks = [Chunk(**c) for c in json.loads((directory / 'chunks.json').read_text())]
        manifest = json.loads((directory / 'manifest.json').read_text())
        vectorizer = (SentenceEncoder(manifest['embedding'])
                      if manifest.get('embedding_backend') == 'semantic'
                      else joblib.load(directory / 'vectorizer.joblib'))
        index = faiss.read_index(str(directory / 'vectors.faiss'))
        dimension = vectorizer.transform(['dimension check']).shape[1]
        if index.ntotal != len(chunks) or index.d != dimension:
            raise ValueError('Index files do not match. Rebuild the index.')
        return cls(chunks, vectorizer, index, manifest)

    def search(self, question: str, top_k: int = 4, min_score: float | None = None):
        if min_score is None:
            min_score = 0.30 if self.manifest.get('embedding_backend') == 'semantic' else 0.09
        vector = np.ascontiguousarray(self.vectorizer.transform([question]).toarray(),
                                     dtype=np.float32)
        if not np.any(vector):
            return []
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, min(top_k, len(self.chunks)))
        hits = [(self.chunks[int(i)], float(score))
                for i, score in zip(ids[0], scores[0]) if i >= 0 and score >= min_score]
        if self.manifest.get('embedding_backend') == 'semantic' and hits:
            def terms(text):
                words = set(re.findall(r'[a-z]{3,}', text.lower())) - ENGLISH_STOP_WORDS
                return {w[:-1] if len(w) > 3 and w.endswith('s') else w for w in words}
            query_terms = terms(question)
            coverage = [len(query_terms & terms(c.text)) for c, _ in hits]
            # Hybrid reranking keeps equally relevant evidence while removing
            # adjacent topics (e.g. class hours when asked for tutor hours).
            best = max(coverage)
            if best:
                hits = [hit for hit, count in zip(hits, coverage) if count == best]
        return hits

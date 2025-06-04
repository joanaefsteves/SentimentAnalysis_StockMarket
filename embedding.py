from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class TextEmbedder:
    def __init__(self, method, **kwargs):
        """
        General-purpose text embedding class.

        Parameters:
        - method: string, the embedding method to use
        - kwargs: method-specific keyword arguments (e.g., vector_size, window, min_count)
        """
        self.method = method
        self.kwargs = kwargs
        self.model = None

    def fit(self, texts):
        """
        Fits the embedding model to the input texts.
        """
        if self.method == 'word2vec':
            tokenized = [text.split() for text in texts]

            self.model = Word2Vec(
                sentences=tokenized,
                vector_size=self.kwargs.get('vector_size', 100),
                window=self.kwargs.get('window', 5),
                min_count=self.kwargs.get('min_count', 1),
                workers=self.kwargs.get('workers', 4)
            )
        elif self.method == 'bow':
            self.model = CountVectorizer(
                max_features=self.kwargs.get('max_features', None),
                ngram_range=self.kwargs.get('ngram_range', (1, 1)),
                stop_words=self.kwargs.get('stop_words', None)
            )
            self.model.fit(texts)

       

    def transform(self, texts):
        """
        Transforms input texts into vector embeddings.
        """
        if self.method == 'word2vec':
            vector_size = self.model.vector_size

            def embed(text):
                words = text.split()
                vectors = [self.model.wv[word] for word in words if word in self.model.wv]
                return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

            return np.array([embed(text) for text in texts])
        
        elif self.method == 'bow':
            return self.model.transform(texts).toarray()

    

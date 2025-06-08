from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, method, **kwargs):
        """
        General-purpose text embedding class.

        Parameters:

        - method (str): The embedding method to use. Options:
            - 'word2vec': Trains a Word2Vec model on the texts.
            - 'bow': Uses a Bag-of-Words representation (via CountVectorizer).
            - 'transformer': Uses a pretrained SentenceTransformer from Hugging Face.

        - kwargs: method-specific keyword arguments

            For method='word2vec':
                - vector_size (int): Dimensionality of word vectors. Default=100
                - window (int): Max distance between current and predicted word. Default=5
                - min_count (int): Ignores words with total frequency lower than this. Default=1
                - workers (int): Number of worker threads. Default=4
                - sg (int): Set to 1 to use Skip-gram; 0 for CBOW (default)

            For method='bow':
                - max_features (int): Max number of features to keep.
                - ngram_range (tuple): e.g., (1,1) for unigrams; (1,2) for unigrams + bigrams.
                - stop_words (str): e.g., 'english' to remove English stopwords.

            For method='transformer':
                - model_name (str) [REQUIRED]: Full Hugging Face model name.
                    Example: 'sentence-transformers/all-MiniLM-L6-v2'
                    See: https://www.sbert.net/docs/pretrained_models.html
                
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
                sg=self.kwargs.get('sg', 1) # Use skipgram by default
            )
        elif self.method == 'bow':
            self.model = CountVectorizer(
                max_features=self.kwargs.get('max_features', None),
                ngram_range=self.kwargs.get('ngram_range', (1, 1)),
                stop_words=self.kwargs.get('stop_words', None)
            )
            self.model.fit(texts)

        elif self.method == 'transformer':
            model_name = self.kwargs.get('model_name')
            
            if model_name is None:
                raise ValueError("You must specify 'model_name' when using method='transformer'")
            
            # Load pretrained transformer model from Hugging Face
            self.model = SentenceTransformer(model_name)

       

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

        elif self.method == 'transformer':
            return self.model.encode(texts, convert_to_numpy=True)
    
#--------------------
""" Example of configuration 
embedding_configs = [
    {'method': 'word2vec', 'vector_size': 100, 'window': 5},
    {'method': 'bow', 'max_features': 1000, 'ngram_range': (1, 2)},
    {'method': 'transformer', 'model_name': 'sentence-transformers/all-MiniLM-L6-v2'},
    {'method': 'transformer', 'model_name': 'sentence-transformers/all-mpnet-base-v2'}
]

"""
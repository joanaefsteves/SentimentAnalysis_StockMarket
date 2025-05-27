import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

class Preprocessing:
    def __init__(self, lemmatize=True, stem=False):
        self.lemmatize = lemmatize
        self.stem = stem
        self.punctuation = set(string.punctuation)
        # STILL NEED TO ADAPT TO CONSIDER MORE LANGUAGES
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess(self, texts):
        return [self._clean_text(text) for text in texts]

    def _clean_text(self, text):

        # Normalize patterns 
        text = re.sub(r'[$€£]\d+(?:[\.,]?\d+)?', '#COST', text)   # costs
        text = re.sub(r'http\S+|www\S+|https\S+', '#URL', text)   # urls
        text = re.sub(r'#\w+', '#HASHTAG', text)                  # hashtags
        text = re.sub(r'@\w+', '#USER', text)                     # mentions
        text = re.sub(r'\$[A-Z]{1,6}\b', '#TICKER', text)         # tickers
        text = re.sub(r'(\w)\1{2,}', r'\1\1', text)

        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())

        # Remove punctuation and stopwords
        tokens = [t for t in tokens if t not in self.punctuation and t not in self.stopwords]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        if self.stem:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens

from abc import ABC, abstractmethod

# Preprocess
import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import emoji
from transformers import BertTokenizer

class Preprocess(ABC):
    def __init__(self, text_col='text'):
        self.text_col = text_col
    
    @abstractmethod
    def demojize(self, df):
        pass
    
    @abstractmethod
    def clean(self, df):
        pass
    
    @abstractmethod
    def tokenize(self, df):
        pass

class StandardPreprocess(Preprocess):
    def __init__(self, lemmatize=True, stem=False, text_col='text'):
        self.lemmatize = lemmatize
        self.stem = stem
        self.punctuation = set(string.punctuation)
        self.text_col = text_col
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def demojize(self, df):
        # Separate emojis
        df["emojis"] = df[self.text_col].apply(lambda x: [d["emoji"] for d in emoji.emoji_list(str(x))])
        # Add column with emojis' text representation - can be used as categoric feature 
        df['demojized'] = df["emojis"].apply(lambda emjs: ''.join(emoji.demojize(e) for e in emjs) if emjs else 'No emojis')

        # Remove emojis
        df[self.text_col] = df[self.text_col].apply(lambda x: ''.join(char for char in str(x) if not emoji.is_emoji(char)))

    def clean(self, df):
        df[self.text_col] = df[self.text_col].apply(self._clean_text)

    def tokenize(self, df):
        df[self.text_col] = df[self.text_col].apply(lambda x: word_tokenize(x))

    
    def _clean_text(self, text):

        # Normalize patterns 
        text = re.sub(r'[$€£]\d+(?:[\.,]?\d+)?', '#COST', text)   # costs
        text = re.sub(r'http\S+|www\S+|https\S+', '#URL', text)   # urls
        text = re.sub(r'#\w+', '#HASHTAG', text)                  # hashtags
        text = re.sub(r'@\w+', '#USER', text)                     # mentions
        text = re.sub(r'\$[A-Z]{1,6}\b', '#TICKER', text)         # tickers

        # Tokenize
        tokens = word_tokenize(text)  

        # Lowercase 
        tokens = [t.lower() if not t.startswith('#') else t for t in tokens]

        # Remove punctuation and stopwords
        tokens = [t for t in tokens if (t not in self.stopwords or t.startswith('#')) and t not in self.punctuation]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        if self.stem:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return ' '.join(tokens)

class PreprocessBERT(StandardPreprocess):
    def __init__(self, lemmatize=True, stem=False, text_col='text', bert_model='bert-base-uncased'):
        super().__init__(lemmatize=lemmatize, stem=stem, text_col=text_col)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

    def demojize(self, df):
        df[self.text_col] = df[self.text_col].apply(lambda x: emoji.demojize(str(x)))
    
    def tokenize(self, df):
        # Tokenize using BERT tokenizer
        inputs = self.tokenizer(
            df[self.text_col].tolist(),
            padding=True,
            truncation=True,
            return_tensors='pt'
            )
        
        return inputs 
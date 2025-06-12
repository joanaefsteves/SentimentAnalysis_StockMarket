from abc import ABC, abstractmethod

# Preprocess
import re, string
import emoji
import torch
import pandas as pd

import nltk
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

from transformers import MarianMTModel, MarianTokenizer
from transformers import BertTokenizer

import langid

# Downloads
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')
nltk.download('punkt')

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
    def __init__(self, lemmatize=True, stem=False, text_col='text', translate= True):
        self.translate = translate
        self.lemmatize = lemmatize
        self.stem = stem
        self.punctuation = set(string.punctuation)
        self.text_col = text_col
        self.stopwords = set(stopwords.words("english"))
        self.english_vocab = set(w.lower() for w in words.words())
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.translator_name = "Helsinki-NLP/opus-mt-mul-en"
        self.tokenizer = MarianTokenizer.from_pretrained(self.translator_name)
        self.model = MarianMTModel.from_pretrained(self.translator_name)
        self.elongation_pattern = re.compile(r"\b\w*(\w)\1{2,}\w*\b")
    
    def demojize(self, df):
        # Separate emojis
        df["emojis"] = df[self.text_col].apply(lambda x: [d["emoji"] for d in emoji.emoji_list(str(x))])
        # Add column with emojis' text representation - can be used as categoric feature 
        df['demojized'] = df["emojis"].apply(lambda emjs: ''.join(emoji.demojize(e) for e in emjs) if emjs else 'No emojis')

        # Remove emojis
        df[self.text_col] = df[self.text_col].apply(lambda x: ''.join(char for char in str(x) if not emoji.is_emoji(char)))

    def clean(self, df):
        df[self.text_col] = df[self.text_col].apply(self._basic_clean)
        valid_words = self._get_valid_elongations(df[self.text_col])
        df[self.text_col] = df[self.text_col].apply(lambda x: self._fix_valid_elongations(x, valid_words))
    
        if self.translate:
            df[self.text_col] = self._translate(df[self.text_col])
        
    def tokenize(self, df):
        def process_tokens(text):
            tokens = word_tokenize(text)
            tokens = [t.lower() if not t.startswith('#') else t for t in tokens]
            tokens = [t for t in tokens if (t not in self.stopwords or t.startswith('#')) and t not in self.punctuation]
            if self.lemmatize:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            if self.stem:
                tokens = [self.stemmer.stem(t) for t in tokens]
            return ' '.join(tokens)

        df[self.text_col] = df[self.text_col].apply(process_tokens)

    def _basic_clean(self, text):
        text = str(text)
        text = text.replace('\n', ' ')
        text = re.sub(r'[$€£]\d+(?:[\.,]?\d+)?', '#COST', text)   # costs
        text = re.sub(r'http\S+|www\S+|https\S+', '#URL', text)   # urls
        text = re.sub(r'#\w+', '#HASHTAG', text)                  # hashtags
        text = re.sub(r'@\w+', '#USER', text)                     # mentions
        text = re.sub(r'\$[A-Z]{1,6}\b', '#TICKER', text)         # tickers
        return text.strip()

    def _get_valid_elongations(self, series):
        all_elongated = series.apply(lambda x: [m.group(0) for m in self.elongation_pattern.finditer(str(x))])
        all_words = all_elongated.explode().dropna().tolist()
        valid = list({w for w in all_words if w.isalpha() and len(w) > 3 and not w.isupper()})
        return valid

    def _fix_valid_elongations(self, text, elong_words):
        for word in elong_words:
            if word in text:
                fixed = re.sub(r'(.)\1{2,}', r'\1\1', word)
                text = text.replace(word, fixed)
        return text    

    def _translate(self, series, threshold=0.9):
        translated = []
        for text in series:
            text_str = str(text)
            lang, prob = langid.classify(text_str)
            tokens = text_str.lower().split()
            has_stops = sum(word in self.stopwords for word in tokens) >= 2
            has_vocab = sum(word in self.english_vocab for word in tokens) >= 1
            if prob < 0.90 and (has_stops or has_vocab):
                lang = 'en'
            if lang != 'en':
                try:
                    inputs = self.tokenizer([text_str], return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        generated = self.model.generate(**inputs)
                    translated.append(self.tokenizer.decode(generated[0], skip_special_tokens=True))
                except Exception:
                    translated.append(text_str)
            else:
                translated.append(text_str)
        return pd.Series(translated, index=series.index)
        
     

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
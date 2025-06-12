
import re, string
import emoji
import torch
import pandas as pd

import nltk # type: ignore
from nltk.corpus import stopwords, words # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.stem import WordNetLemmatizer, PorterStemmer # type: ignore

from transformers import MarianMTModel, MarianTokenizer
from transformers import BertTokenizer

import langid

from emoji import demojize # type: ignore


"""
    ascii 65-90
    ascii 97-122
"""

class Preprocessing:
    def __init__(self, lemmatize=True, stem=False, text_col='text', emoji_support_level=0, translate=False):
        self.text_col = text_col
        self.translate = translate
        
        self._download_nltk_data()
        self.lemmatize = lemmatize
        self.stem = stem
        self.punctuation = set(string.punctuation)
        self.emoji_support_level = emoji_support_level
        self.emoji_list = []
        
        self.translator_name = "Helsinki-NLP/opus-mt-mul-en"
        self.tokenizer = MarianTokenizer.from_pretrained(self.translator_name)
        self.model = MarianMTModel.from_pretrained(self.translator_name)
        self.elongation_pattern = re.compile(r"\b\w*(\w)\1{2,}\w*\b")
    
        
        # Dictionary of regex patterns and their normalized replacements
        self.normalization_patterns = {
            r'\n': ' ',                             # line breaks 
            r'[$€£]\d+(?:[\.,]?\d+)?': 'COST',      # costs
            r'http\S+|www\S+|https\S+': 'URL',      # urls
            r'#\w+': 'HASHTAG',                      # hashtags
            r'@\w+': 'USER',                         # mentions
            r'\$[A-Z]{1,6}\b': 'TICKER',             # tickers
            r':\w+:': 'EMOJI',                        # emojis in text
            r"\'": '',                             # remove single quotes
            
        }
        
        # STILL NEED TO ADAPT TO CONSIDER MORE LANGUAGES
        self.stopwords = set(stopwords.words("english"))
        self.english_vocab = set(w.lower() for w in words.words())
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
    def _download_nltk_data(self):
        """Download required NLTK data if not already present"""
        resources = [
            ('corpora/stopwords', 'stopwords'),
            ('corpora/words', 'words'),
            ('tokenizers/punkt', 'punkt'),  # Tokenizer for word tokenization
            ('tokenizers/punkt_tab', 'punkt_tab'),  # Updated tokenizer
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4')  # Additional wordnet data
        ]
        
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                print(f"Downloading {resource_name}...")
                nltk.download(resource_name, quiet=True)

    def preprocess(self, texts):
        return [self._clean_text(text) for text in texts]
    
    def final_ascii_clean(self, texts):
        """Final cleaning to ensure all characters are ASCII"""
        cleaned_texts_list = []
        print(f"Initial texts: {texts}")  # Debugging line to print initial texts
        for text in texts:
            print(f"Processing text: {text}")  # Debugging line to print each text being processed
            print(f"ord values: {[ord(char) for char in text]}")  # Debugging line to print ord values of characte
            cleaned_text = []
            for char in text:
                if 65 <= ord(char) <= 90 or 97 <= ord(char) <= 122:
                    cleaned_text.append(char)
            cleaned_texts_list.append(''.join(cleaned_text))
        return cleaned_texts_list
    
    def _translate(self, text, threshold=0.9):
  
        if not self.translate:
            return text
        try:
            text_str = str(text)
            lang, prob = langid.classify(text_str)
            tokens = text_str.lower().split()
            has_stops = sum(word in self.stopwords for word in tokens) >= 2
            has_vocab = sum(word in self.english_vocab for word in tokens) >= 1

            # Força para inglês se baixa confiança mas parecer inglês
            if prob < threshold and (has_stops or has_vocab):
                lang = 'en'

            if lang != 'en':
                inputs = self.tokenizer([text_str], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    generated = self.model.generate(**inputs)
                return self.tokenizer.decode(generated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Translation failed: {e}")
        return text 

    def _clean_text(self, text):
        
        text = text.lower()  # Convert to lowercase
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns.items():
            if replacement == 'EMOJI' and self.emoji_support_level > 0:
                print(f"Processing emojis with support level {self.emoji_support_level}")
                rep = text[text.find(':')+1:text.rfind(':')] if ':' in text else ''
                text = re.sub(pattern, rep, text)
            else:
                text = re.sub(pattern, replacement, text)


        text = self._fix_elongations(text)  # Fix elongations

        if self.translate: 
            text = self._translate(text)  # Translate if needed
        # Tokenize and lowercase
        
        ascii_text  = self.final_ascii_clean(text)
        print(f"ASCII cleaned text: {ascii_text}")  # Debugging line to print ASCII cleaned text
        tokens = word_tokenize(ascii_text)

        # Remove punctuation and stopwords
        tokens = [t for t in tokens if (t not in self.punctuation and t not in self.stopwords)]
        
        #[print(t) for t in tokens ] # Debugging line to print tokens
        # Get normalized pattern values for comparison
        normalized_values = set(self.get_normalized_patterns())

        # Apply lemmatization and stemming only to non-normalized tokens
        processed_tokens = []
        for token in tokens:
            if token.upper() in normalized_values:  # Check if it's a normalized pattern
                processed_tokens.append(token)
            else:
                if self.lemmatize:
                    token = self.lemmatizer.lemmatize(token)
                if self.stem:
                    token = self.stemmer.stem(token)
                processed_tokens.append(token)

        return processed_tokens
    
    def _get_valid_elongations(self, text):
        
        return [m.group(0) for m in self.elongation_pattern.finditer(str(text))]

    def _fix_elongations(self, text):
        
        elong_words = self._get_valid_elongations(text)
        for word in elong_words:
            fixed = re.sub(r'(.)\1{2,}', r'\1\1', word)
            text = text.replace(word, fixed)
        return text

    def demojize(self, df):
        df[self.text_col] = df[self.text_col].apply(lambda x: demojize(str(x)))
    
    def get_stopwords(self):
        """Get the set of stopwords used in preprocessing"""
        return self.stopwords
    def get_punctuation(self):
        """Get the set of punctuation characters used in preprocessing"""
        return self.punctuation
    def get_lemmatizer(self):
        """Get the lemmatizer used in preprocessing"""
        return self.lemmatizer
    def get_stemmer(self):
        """Get the stemmer used in preprocessing"""
        return self.stemmer
    def get_normalization_patterns(self):
        """Get the normalization patterns used in preprocessing"""
        return self.normalization_patterns
    def get_normalized_patterns(self):
        """Get the normalized patterns used in preprocessing"""
        return list(self.normalization_patterns.values())

from string import punctuation

import nltk
import pandas as pd

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords

from stopwords import additional_stopwords

nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer


def read_file(file_name: str) -> pd.DataFrame:
    """ Reads the specified CSV file. Returns dataframe object. """
    data_frame = pd.read_csv(file_name)
    return data_frame

def tokenize_review(review: str) -> list:
    """ Tokenize a sentence. The term 'tokenizing' is the process
        of separating the words or sentence to work with smaller
        pieces of text. For this case, we are separating each review
        by sentences, then each sentence to words.
    """
    def not_symbol(word: str) -> bool:
        for symbol in punctuation:
            if symbol == word:
                return False
        return True
    return list(filter(not_symbol, word_tokenize(review.lower())))

def reviews_to_list(dataframe: object) -> list:
    """ Lowercases the reviews and converts them into 
        a list of reviews 
    """
    return dataframe['reviews'].values.tolist()

def text_normalizing(words: list) -> list:
    """ Transforms text to a single canonical form. """
    
    def tag_word(words: list) -> list:
        """ Takes a list of words and returns a list of tuples, 
            with the first element in the tuple being the
            word, and the second being the tag.
            Possible tags: NNP, NN, IN, VBG, VBN
            Returns a list of tuples.
            More in this link: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        """
        return pos_tag(words)
    
    def lemmatize_words(tokens: list) -> list:
        """ Reduces each word in the review to its most basic form. 
            Returns a list containing lemmatized words (tokens).
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = []
        for word, tag in tag_word(tokens):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a' # Label other tags as 'a', as they are not as relevant
            lemmatized_tokens.append(lemmatizer.lemmatize(word, pos))
        return lemmatized_tokens

    def remove_noise(tokens: list, stopwords: set) -> list:
        """ Returns a list containing lemmatized words, excluding stopwords """

        def contains_numbers(token: str) -> bool:
            """ Checks if the token contains numbers """
            numbers = '1234567890'
            for number in numbers:
                if number in token:
                    return True
            return False

        tokens_no_stopwords = []
        for token in tokens:
            if token not in stopwords and not contains_numbers(token):
                tokens_no_stopwords.append(token)
        return tokens_no_stopwords

    lemmetize_tokens = lemmatize_words(words)
    
    return remove_noise(lemmetize_tokens, set(stopwords.words('english')).union(additional_stopwords))

def text_preprocessing(review: str) -> list:
    """ Text preprocessing. Takes the review, tokenizes it into words,
        and then performes the lemmatization and noise removing procedure (to
        remove stopwords and numbers).

        Returns a list of words in a review.    
    """
    return text_normalizing(tokenize_review(review))

def form_corpus(reviews: list) -> list:
    """ Takes the list of processed reviews and form a corpus for each review. """
    return [' '.join(review) for review in reviews]

def vectorizer() -> TfidfVectorizer:
    """ Creates a new TfidfVectorizer object to extract 
        features from reviews
    """
    return TfidfVectorizer()

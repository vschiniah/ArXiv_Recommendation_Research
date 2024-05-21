import ast
import re
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')


# TODO: Function to preprocess text
def preprocess_text(text, stopwords):
    """
    Preprocesses text by removing stopwords and lemmatizing words
    :param text: text to be preprocessed
    :param stopwords: stopwords to be removed from the text - stopwords read from txt file
    :return: preprocessed text
    """

    # Lowercasing
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords]
    return ' '.join(lemmatized_words)


def extract_authors(author_list_str):
    """
    Extracts authors from a string of authors
    :param author_list_str: list of authors
    :return: list of authors extracted from author_list_str
    """
    try:
        author_list_str = author_list_str.replace("\\'", "'")
        author_list = ast.literal_eval(author_list_str)
        full_names = [f"{author[1].strip()} {author[0].strip()}" for author in author_list if author[0] and author[1]]
        return full_names
    except SyntaxError as e:
        print(f"Error parsing author list: {e}")
        return []


def convert_name(name):
    """
    Converts Author name from Jane Doe to J. Doe to match ArXiv Data to Semantic Scholar Data
    """
    parts = name.split()
    if len(parts) > 1:
        first_name_initial = parts[0][0]
        last_name = parts[-1]
        return f"{first_name_initial}. {last_name}"
    return name

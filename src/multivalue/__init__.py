__version__ = "0.1"
__organization__ = "Social And Language Technology Lab"
import nltk
from spacy.cli import download


download("en_core_web_sm")

nltk.download("cmudict")
nltk.download("wordnet")

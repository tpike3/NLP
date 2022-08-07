from streamlit import secrets

class BaseConfig:
    APP_TITLE = "Natural Language Processing Tool"
    API_KEY = None
    

class ExternalConfig:
    TEXT_CATEGORIZATION_URL = "https://api-inference.huggingface.co/models/dbmdz/bert-large-cased-finetuned-conll03-english"
    TEXT_SUMMARIZATION_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    DOCUMENT_CLUSTERING_URL = "https://api-inference.huggingface.co/models/Craig/paraphrase-MiniLM-L6-v2"


class InternalConfig:
    TEXT_CATEGORIZATION_REPO = "facebook/bart-large-mnli"
    TEXT_SUMMARIZATION_REPO = "facebook/bart-large-cnn"
    DOCUMENT_CLUSTERING_REPO = "sentence-transformers/paraphrase-MiniLM-L6-v2"

    TEXT_CATEGORIZATION_CLASSIFIER = "zero-shot-classification"
    TEXT_SUMMARIZATION_CLASSIFIER = "summarization"


def get_config():
    if __has_secrets() and secrets.has_key('api_key'):
        config = {
            **BaseConfig.__dict__,
            **ExternalConfig.__dict__
        }
        config['API_KEY'] = secrets.get('api_key')
    else:
        config = {
            **BaseConfig.__dict__,
            **InternalConfig.__dict__
        }
    return config


def __has_secrets():
    from os.path import exists
    return exists(secrets._file_path)
import json
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from config import get_config
from os import path

appconfig = get_config()

#### While some tasks (Basic Text Analysis, Named Entity Recognition, and Topic Modeling) //
# are run locally, the others (Text Categorization, Text Summarization, and Document Clustering) //
# require making API calls to models in the Hugging Face model hub. The following //
# is used to set up the API calls to the model hub.
def query(payload, url, key):
    if key:
        headers = {'Authorization': key}
        data = json.dumps(payload)
        response = requests.request("POST", url, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))
    else:
        print('No API Key provided')
        return None


def convert_method_string(method_string):
    return '_'.join(method_string.split()).upper()


def get_pipeline(method):
    repo = appconfig[f'{method}_REPO']
    classifier = appconfig[f'{method}_CLASSIFIER']
    
    model_fpath = repo.replace('/', '_')
    if path.exists(model_fpath):
        # model has already been downloaded, load from disk
        return pipeline(classifier, model=model_fpath)
    else:
        # model has not been downloaded yet, download and save locally before returning
        pl = pipeline(classifier, model=repo)
        pl.save_pretrained(model_fpath)
        return pl


def get_sentence_transformer(method):
    repo = appconfig[f'{method}_REPO']
    
    model_fpath = repo.replace('/', '_')
    if path.exists(model_fpath):
        # model has already been downloaded, load from disk
        return SentenceTransformer(model_fpath)
    else:
        # model has not been downloaded yet, download and save locally before returning
        model = SentenceTransformer(repo)
        model.save(model_fpath)
        return model


def run_model(method_string, inputs, parameters):
    """
    Args:
        method (String): The method being run (i.e. "Text Categorization" or "Topic Modeling")
    """
    method = convert_method_string(method_string)
    print(f"Running {method}")
    if appconfig['API_KEY']:
        # run against external API
        url = appconfig[f'{method}_URL']
        payload = {"inputs": inputs, "parameters": parameters}
        return query(payload, url, appconfig['API_KEY'])
    else:
        # use locally downloaded models
        if method == "DOCUMENT_CLUSTERING":
            model = get_sentence_transformer(method)
            return model.encode(inputs)
        else:
            pl = get_pipeline(method)
            return pl(inputs, **parameters)
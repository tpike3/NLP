##### Natural Language Processing Application (Streamlit) ####

## Importing the libraries
from warnings import simplefilter
import streamlit as st
from streamlit_option_menu import option_menu
from analytics.base import Analytic
from config import get_config

appconfig = get_config()

simplefilter(action='ignore', category=FutureWarning)

# Configuring the main page, defining functions that will be used across multiple tasks,//
# and setting the Session State objects (this is data that is kept in memory while the //
# user navigates around through the various tasks).

st.set_page_config(page_title=appconfig['APP_TITLE'], page_icon="🤖")
st.title(appconfig['APP_TITLE'])

with st.sidebar:
    selected = option_menu("Method Menu", ['Home', 'Basic Text Analysis', 'Named Entity Recognition', 'Text Categorization', 'Text Summarization','Topic Modeling','Document Clustering'], icons=['house'], menu_icon="cast", default_index=0)

if "text" not in st.session_state:
    st.session_state.text = ''

if "multitext_all" not in st.session_state:
    st.session_state.multitext_all = ''

if "multitext_short" not in st.session_state:
    st.session_state.multitext_short = ''

## The sidebar navigation pane you see on the app is accessed through "if statements". //
# Below you will see each task tab corresponds to an "if statement", and that some of the //
# tasks have sub-"if statements" nested within them.
def get_analytic(name) -> Analytic:
    if name == 'Home':
        from analytics.home import Home
        return Home(name)

    if name == 'Basic Text Analysis':
        from analytics.basic_text_analysis import TextAnalysis
        return TextAnalysis(name)

    if name == 'Named Entity Recognition':
        from analytics.named_entity_recognition import NER
        return NER(name)

    if name == 'Text Categorization':
        from analytics.text_categorization import TextCategorization
        return TextCategorization(name)

    if name == 'Text Summarization':
        from analytics.text_summarization import TextSummarization
        return TextSummarization(name)

    if name == 'Topic Modeling':
        from analytics.topic_modeling import TopicModeling
        return TopicModeling(name)
        
    if name == 'Document Clustering':
        from analytics.document_clustering import DocumentClustering
        return DocumentClustering(name)


analytic = get_analytic(selected)
analytic.run()
    

#### This last portion is something of a placeholder -- the code allows putting //
# a small amount of text at the bottom of the page and could be usefil later on //
# if this application is deployed more widely.
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            footer:after {
	        content:'Created by Adam J'; 
	        visibility: visible;
	        display: block;
	        position: relative;
	        padding: 5px;
	        top: 2px;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

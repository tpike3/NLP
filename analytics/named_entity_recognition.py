import pandas as pd
import spacy
import streamlit as st
from spacy import displacy
from analytics.base import Analytic

class NER(Analytic):
    def __init__(self, name):
        super().__init__(name)
        st.subheader(self.name)
        self.doc_NERs = list()
        self.options = ["ORG", "GPE", "PERSON", "LOC"]


    def run(self):
        st.write("**Description:** This task involves identifying geopolitical entities, organizations, people, and locations in a body of text.")
        submit4 = st.button('Analyze Text')

        if submit4:
            if len(st.session_state.text) > 0:
                entities, entityLabels = self.get_entities(st.session_state.text)
                if entities:
                    chart = self.ner_chart(entities, entityLabels)
                    st.write(chart)
                    self.download_results(chart)
                else:
                    st.write("No named entities found in the text.")

            elif len(st.session_state.multitext_all) > 0:
                entities1 = []
                entityLabels1 = []
                for i in st.session_state.multitext_all:
                    entities, entityLabels = self.get_entities(i)
                    if entities:
                        entities1.extend(entities)
                        entityLabels1.extend(entityLabels)
                
                if entities1:
                    chart = self.ner_chart(entities1, entityLabels1)
                    st.write(chart)
                    self.download_results(chart)
                else:
                    st.write("No named entities found in the text.")
                
            else:
                st.write("Please enter text or upload file on the Home page.")


    def get_entities(self, text):
        nlp = spacy.load('en_core_web_sm')
        entities = []
        entityLabels = []
        doc_NER = nlp(text)
        self.doc_NERs.append(doc_NER)
        if len(doc_NER.ents) > 0:
            for i in doc_NER.ents:
                entities.append(i.text)
                entityLabels.append(i.label_)
            return entities, entityLabels
        else:
            return None, None


    def ner_chart(self, entities, entityLabels):
        df = pd.DataFrame({'Entity': entities, 'Type of Entity': entityLabels})
        df = df[df['Type of Entity'].isin(self.options)]
        df['Type of Entity'] = df['Type of Entity'].str.replace('PERSON', 'Person').str.replace('ORG', 'Organization').str.replace('LOC', 'Location').str.replace('GPE', 'Geopolitical Entity')
        df.sort_values(['Type of Entity'], ascending=False, inplace=True)
        df = df.groupby(["Type of Entity", "Entity"]).size().reset_index(name="Count").sort_values(by=["Count","Type of Entity"], ascending=False)
        return df


    def download_results(self, chart):
        NER_Results = chart.to_csv(index=False, header=True)
        st.download_button("Download Results", NER_Results, file_name="NER_Results.csv")

        options = {"ents": self.options}
        HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
        
        st.write('--------------------------------------------')
        for doc_NER in self.doc_NERs:
            html = displacy.render(doc_NER, style="ent", options=options)
            html = html.replace("\n", " ")

            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
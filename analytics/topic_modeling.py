from functools import partial
import streamlit as st
import textacy
import textacy.tm
from analytics.base import Analytic

class TopicModeling(Analytic):
    def __init__(self, name):
        super().__init__(name)
        st.subheader(self.name)

    def run(self):
        st.write("**Description:** This task involves finding topics among a collection of documents.")
        st.write("_Note: This task only applies to multiple document files (i.e. an uploaded CSV file)._")
        
        graph_topic_to_highlight = [0,2,4]

        num_topics = st.slider('Number of Topics to Find ', min_value=1, max_value=12, value=6, step=1)
        words_in_topic = 10
        n_grams = st.radio('N-Gram Size', ['Single Word', 'Two Word Phrases', 'Three Word Phrases'])
        model_type = st.radio('Select Topic Modeling Algorithm', ['Latent Dirichlet Allocation','Non-Negative Matrix Factorization', 'Latent Semantic Analysis'])
        if model_type == 'Non-Negative Matrix Factorization':
            model_type = 'nmf'
        elif model_type == 'Latent Dirichlet Allocation':
            model_type = 'lda'
        elif model_type == 'Latent Semantic Analysis':
            model_type = 'lsa'
        if n_grams == 'Single Word':
            n_grams = 1
        elif n_grams == 'Two Word Phrases':
            n_grams = 2
        elif n_grams == 'Three Word Phrases':
            n_grams = 3

        submit4 = st.button('Analyze Text')

        if submit4:
            if len(st.session_state.text) > 0:
                st.write('I am sorry this method does not apply to single texts. Please return to the Home page and upload a CSV file of mutiple texts.')
            elif len(st.session_state.multitext_all) > 0:
                model, id_to_term, doc_term_matrix = self.get_model(model_type, n_grams, num_topics)
                #doc_topic_matrix = model.transform(doc_term_matrix)
                
                st.write('--------------------------------------------')
                st.write('**Below are the top 10 word/phrases for each topic:**')
                for topic_idx, terms in model.top_topic_terms(id_to_term, top_n=words_in_topic):
                    st.write(f"**Topic {topic_idx}**: {'; '.join(terms)}")
                
                st.write('--------------------------------------------')
                st.write('**Chart of Topics and Words**')
                plot1 = model.termite_plot(doc_term_matrix, id_to_term, n_terms=12, highlight_topics=graph_topic_to_highlight,save = "termite_plot.png")
                st.image('./termite_plot.png')
            else:
                st.write("Please upload a CSV file on the Home page.")

    def get_model(self, model_type, n_grams, num_topics):
        corpus = textacy.Corpus("en_core_web_sm", data=st.session_state.multitext_all)
        docs_terms = (textacy.extract.terms(doc, ngs=partial(textacy.extract.ngrams, n=n_grams, include_pos={"PROPN", "NOUN", "ADJ", "VERB"})) for doc in corpus)
        tokenized_docs = (textacy.extract.terms_to_strings(doc_terms, by="lemma") for doc_terms in docs_terms)
        doc_term_matrix, vocab = textacy.representations.build_doc_term_matrix(tokenized_docs, tf_type='linear', idf_type='smooth')
        
        model = textacy.tm.TopicModel(model_type, n_topics=num_topics)
        model.fit(doc_term_matrix)

        id_to_term = {id_: term for term, id_ in vocab.items()}
        return model, id_to_term, doc_term_matrix
    
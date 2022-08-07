import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
from umap import UMAP
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from cluestar import plot_text
from analytics.base import Analytic

class TextAnalysis(Analytic):
    def __init__(self, name):
        super().__init__(name)
        st.subheader(self.name)

    def run(self):
        st.write("**Description:** This task involves producing simple descriptive statistics about your text.")
        submit3 = st.button('Analyze Text')
        if submit3:
            if len(st.session_state.text) > 0:
                word_list = st.session_state.text.split()

                self.stats(word_list, 1)
                
                st.write('--------------------------------------------')
                self.word_cloud(word_list)

                st.write('--------------------------------------------')   
                st.write("**Uploaded Text:** ", st.session_state.text)
                    
            elif len(st.session_state.multitext_all) > 0:
                doc_count1 = len(st.session_state.multitext_all)
                word_list1 =[]
                for i in st.session_state.multitext_all:
                    word_list1.extend(i.split())

                self.stats(word_list1, doc_count1)

                st.write('--------------------------------------------')
                chart = self.embedding_chart(st.session_state.multitext_all)
                st.altair_chart(chart, use_container_width=True)
                
                st.write('--------------------------------------------')
                self.word_cloud(word_list1)

                st.write('--------------------------------------------')
                self.display_multitext(st.session_state.multitext_all)
                
            else:
                st.write("Please enter text or upload file on the Home page.")


    def word_cloud(self, word_list):
        word_cloud = WordCloud(background_color='white', max_words=100, max_font_size=50, random_state=42).generate(' '.join(word_list))
        st.write("Word Cloud: ")
        st.image(word_cloud.to_image())


    def stats(self, word_list, doc_count):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Document Count: ", doc_count)
            st.write("Unique Word Count: ", len(set(word_list)))

        with col2:
            st.write("Word Count: ", len(word_list))
            avg_word = np.mean([len(word) for word in word_list])
            st.write("Average Word Length: ", round(avg_word, 2))


    def embedding_chart(self, alltext):
        pipe = make_pipeline(TfidfVectorizer(), UMAP(random_state=42))
        X = pipe.fit_transform(alltext)
        return plot_text(X, alltext)


    def display_multitext(self, alltext):
        df = pd.DataFrame(alltext, columns=['Documents'])
        st.write("**Uploaded Text Below:** ")
        st.table(df)
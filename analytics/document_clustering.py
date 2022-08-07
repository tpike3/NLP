import pandas as pd
import plotly.express as px
import streamlit as st
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from cluestar import plot_text
import utils
from analytics.base import Analytic

class DocumentClustering(Analytic):
    def __init__(self, name):
        super().__init__(name)
        st.subheader(self.name)

    def run(self):
        st.write("**Description:** This task involves placing documents into groups based on similarity and then extracting the key words/phrases from each group.")
        st.write("_Note: This task only applies to multiple document files (i.e. an uploaded CSV file)._")
        
        num_clusters = st.slider('Number of Clusters to Create ', min_value=2, max_value=10, value=4, step=1)
        submit6 = st.button('Analyze Text')

        if submit6:
            if len(st.session_state.text) > 0:
                st.write('I am sorry this method does not apply to single texts. Please return to the Home page and upload a CSV file of mutiple texts.')
            elif len(st.session_state.multitext_all) > 0:
                word_list = st.session_state.multitext_all
                corpus_embeddings = utils.run_model(self.name, word_list, {"wait_for_model": True})
                
                cluster_df = self.get_cluster(word_list, corpus_embeddings, num_clusters)
                st.write('The table below displays the document clusters and key words/phrases extracted from each cluster.')
                st.table(cluster_df)

                scatter_chart, pca_outputs = self.build_scatter_chart(word_list, corpus_embeddings)                
                st.write('--------------------------------------------')
                st.write('Below is an interactive 2D scatter plot of the documents. Each point represents a document and the colors correspond to the clusters. Click and drag over several points to see the text displayed on the right.')
                st.altair_chart(scatter_chart, use_container_width=True)

                fig = px.scatter_3d(pca_outputs, x='First Component', y='Second Component', z='Third Component', color='cluster', hover_name='short_text')
                st.write('--------------------------------------------')
                st.write('Below is an interactive 3D scatter plot of the documents. Each point represents a document and the colors correspond to the clusters. You can explore the graphic closer by clicking the "view fullscreen" button in the top right corner.')
                st.plotly_chart(fig)

                cluster_df = pca_outputs[['cluster', 'text']]
                Cluster_Results = cluster_df.to_csv(index=False, header=True)
                st.download_button("Download Results", Cluster_Results, file_name="Document_Clustering_Results.csv")

            else:
                st.write("Please upload a CSV file on the Home page.")


    def build_scatter_chart(self, word_list, corpus_embeddings):
        embeds = pd.DataFrame(corpus_embeddings)
        pca = PCA(n_components=3)
        pca_test = pca.fit_transform(embeds)

        cluster_assignment = [str(x) for x in cluster_assignment]
        new_list1 = []
        pca_outputs = pd.DataFrame(pca_test)
        pca_outputs.columns = ['First Component', 'Second Component', 'Third Component']
        pca_outputs['cluster'] = cluster_assignment
        pca_outputs['text'] = word_list
        pca_outputs['short_text'] = new_list1
        pca_outputs.sort_values(by=['cluster'], ascending=True, inplace=True)

        pipe = make_pipeline(TfidfVectorizer(), UMAP(random_state=42))
        X = pipe.fit_transform(word_list)
        chart = plot_text(X, word_list, color_array=pca_outputs.cluster)
        return chart, pca_outputs


    def get_cluster(self, word_list, corpus_embeddings, num_clusters):
        clustering_model = KMeans(n_clusters=num_clusters, random_state=15)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
        cluster_assignment += 1

        df1 = pd.DataFrame(list(zip(word_list, cluster_assignment)), columns=['text','cluster'])
        model = 'all-MiniLM-L6-v2'
        kw_model = KeyBERT(model)
        kp_vect = KeyphraseCountVectorizer(lowercase=False)

        lister = []
        for i in df1.cluster.unique():
            lister.append(i)
            lister.sort()

        lister2 = []
        for i in lister:
            lister2.append(' '.join(df1[df1.cluster == i]['text'].tolist()))
        clust = 1
        new_list = []
        progress_bar = st.progress(0)
        for i in lister2:
            keyphrase_data = kw_model.extract_keywords(docs=i, vectorizer=kp_vect, top_n=6, use_maxsum=True, nr_candidates=18)
            phrases =[]
            for i in keyphrase_data:
                phrases.append(i[0])
            new_list.append(phrases)
            progress_bar.progress(round(clust/len(lister2), 1))
            clust +=1

        joined_list = []
        for i in new_list:
            joined_list.append(' ; '.join(i))

        df = pd.DataFrame(list(zip(lister, joined_list)), columns=['Cluster Number','Key Words/Phrases Within Each Cluster'], index=None)
        df.index = [""] * len(df)
        return df
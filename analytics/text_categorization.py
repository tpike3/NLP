import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import utils
from analytics.base import Analytic

class TextCategorization(Analytic):
    def __init__(self, name):
        super().__init__(name)
        st.subheader(self.name)

    def run(self):
        st.write("**Description:** This task involves placing a piece of text into one or more categories.")
        
        cat1 = st.text_input('Enter each possible category name (separated by a comma). Maximum 5 categories.')
        labels1 = cat1.strip().split(',')
        submit3 = st.button('Analyze Text')
        if submit3:
            if len(st.session_state.text) > 0:
                text_class = utils.run_model(self.name, st.session_state.text, {"candidate_labels": labels1})

                cat1name = text_class['labels'][0]
                cat1prob = text_class['scores'][0]
                st.write('--------------------------------------------')
                st.write('**Text:** ', st.session_state.text)
                st.write('**Category:** {} | **Probability:** {:.1f}%'.format(cat1name.title(),(cat1prob*100)))

                figure = self.prediction_figure(text_class['labels'][::-1][-10:], text_class['scores'][::-1][-10:])
                st.plotly_chart(figure)

            elif len(st.session_state.multitext_all) > 0:
                st.write('--------------------------------------------')
                st.write("_Note: During the current testing phase, this task can only be performed on the first 10 documents._")
                all_text_class = []
                class_name = []
                score_name = []
                my_bar = st.progress(0)
                for i in st.session_state.multitext_short:

                    text_class = utils.run_model(self.name, i, {"candidate_labels": labels1, "wait_for_model": True})

                    all_text_class.append(i)
                    class_name.append(text_class['labels'][0])
                    score_name.append(text_class['scores'][0])
                    my_bar.progress(len(class_name)*10)

                df = pd.DataFrame({'Text': all_text_class, 'Category': class_name, 'Probability': score_name})
                st.write(df)

                Classification_Results = df.to_csv(index=False, header=True)
                st.download_button("Download Results", Classification_Results, file_name="Categorization_Results.csv")

            else:
                st.write("Please enter text or upload file on the Home page.")

    def prediction_figure(self, top_topics, scores):
        top_topics = np.array(top_topics)
        scores = np.array(scores)
        scores *= 100
        fig = px.bar(x=scores, y=top_topics, orientation='h', 
                    labels={'x': 'Probability', 'y': 'Category'},
                    text=scores,
                    range_x=(0,115),
                    title='Top Predictions',
                    color=np.linspace(0,1,len(scores)),
                    color_continuous_scale="Bluered")
        fig.update(layout_coloraxis_showscale=False)
        fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
        return fig
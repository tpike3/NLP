import pandas as pd
import streamlit as st
import utils
from analytics.base import Analytic

class TextSummarization(Analytic):
    def __init__(self, name):
        super().__init__(name)
        st.subheader(self.name)

    def run(self):
        st.subheader(self.name)
        st.write("**Description:** This task involves condensing larger bodies of text into smaller bodies of text.")

        max_lengthy = st.slider('Maximum summary length (words)', min_value=30, max_value=120, value=50, step=5)
        min_lengthy = st.slider('Minimum summary length (words)', min_value=10, max_value=60, value=30, step=5)
        sum_choice = st.radio('If analyzing multiple texts (i.e. a CSV file), please choose how you would like to summarize your data.', ['All documents to one summary.', 'Each document summarized individually.'])
        submit5 = st.button('Analyze Text')  

        if submit5:
            if len(st.session_state.text) > 0:

                data = utils.run_model(self.name, st.session_state.text, {"do_sample": False, "max_length": max_lengthy, "min_length": min_lengthy})
                
                st.write('--------------------------------------------')
                st.write("**Summary:**  ", data[0]['summary_text'])
                st.write("**Original Text:**  ", st.session_state.text)

            elif len(st.session_state.multitext_all) > 0:
                st.write("_Note: During the current testing phase, this task can only be performed on the first 10 documents._")
                if sum_choice == 'All documents to one summary.':
                    sum_text = ' '.join(st.session_state.multitext_short)
                    sum_text2 = sum_text[0:1023]

                    data = utils.run_model(self.name, sum_text2, {"do_sample": False, "max_length": max_lengthy, "min_length": min_lengthy})
                    
                    st.write('--------------------------------------------')
                    st.write("**Summary:**  ", data[0]['summary_text'])
                if sum_choice == 'Each document summarized individually.':
                    sum_list = []
                    my_bar1 = st.progress(0)
                    for i in st.session_state.multitext_short:

                        data = utils.run_model(self.name, i, {"do_sample": False, "max_length": max_lengthy, "min_length": min_lengthy})

                        sum_list.append(data[0]['summary_text'])
                        my_bar1.progress(len(sum_list)*10)
                    
                    sum_all = list(zip(st.session_state.multitext_short, sum_list))
                    sum_df = pd.DataFrame(sum_all, columns=['Text', 'Summary'])
                    st.write('--------------------------------------------')
                    st.write(sum_df)
                    Summary_Results = sum_df.to_csv(index=False, header=True)
                    st.download_button("Download Results", Summary_Results, file_name="Summary_Results.csv")
            else:
                st.write("Please enter text or upload file on the Home page.")
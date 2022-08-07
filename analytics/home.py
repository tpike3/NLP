import streamlit as st    
import pandas as pd
from analytics.base import Analytic

class Home(Analytic):
    def __init__(self, name):
        super().__init__(name)

    def run(self):
        st.write("_**Introduction:** This web application allows you to use machine learning to analyze text. On the left-hand side of the screen you can see six analysis tasks that can be applied to your text._")

        st.write("**Instructions:** The first step is to choose how you would like to input your text. You can either type it into the text box (single document), or upload a spreadsheet with multiple rows of text (multiple documents). Once you have saved and submitted the text, you can proceed to the analysis tasks.")  
        
        choice = st.radio('Please select one of the following options:', ['Type in Text Box', 'Upload Spreadsheet with Text Column'])
        if choice == 'Type in Text Box':
            text = st.text_area('Enter Text Below:', height=200)
            submit1 = st.button('Save and Submit Text')
            if submit1:    
                if len(text) < 1:
                    st.write("Please enter text above.")
                else:
                    st.session_state.text = text
                    st.session_state.multitext_all = ''
                    st.session_state.multitext_short = ''
                    st.write('Text saved successfully! You are now ready to try out the analysis tasks.')
            
        if choice == 'Upload Spreadsheet with Text Column':
            st.write("Below is an example of a properly formatted spreadsheet file. The first column should contain the text you would like analyzed. Additional columns, such as 'Source' and 'Date' in the example below, are optional.")
            demo = {'Text': ['This is the first sentence.', 'And this is the second', 'And this is the third.'], 'Source': ['Internet', 'Newspaper', 'TV'], 'Date': ['2/13/2021','4/27/2021','8/1/2021']}
            demo1 = pd.DataFrame(demo)
            st.dataframe(demo1)
            dfs = st.file_uploader("Please drag and drop your file into the space below (.csv files only)", type=["csv"])
            submit2 = st.button('Save and Submit Text')
            if submit2:
                if dfs is None:
                    st.write("Please upload a file.")
                else:
                    st.write('Text uploaded successfully! You are now ready to try out the analysis tasks.')
                    df = pd.read_csv(dfs, encoding_errors = 'ignore')
                    first_column_all = df.iloc[1:, 0]
                    first_column_short = df.iloc[1:11, 0]  
                    texts_all = first_column_all.to_list()
                    texts_short = first_column_short.to_list()
                    test_all = [str(x) for x in texts_all]
                    test_short = [str(x) for x in texts_short]
                    st.session_state.multitext_all = test_all
                    st.session_state.multitext_short = test_short
                    st.session_state.text = ''
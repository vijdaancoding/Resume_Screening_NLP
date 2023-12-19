# Importing Libraries
import streamlit as st 
import pickle 
import re 
import nltk 
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Downloading data
nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
model_svm = pickle.load(open('model_svm.pkl', 'rb'))

# Cleaning Data Function
porter_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def dataCleaning(txt):
    CleanData = re.sub('https\S+\s', ' ', txt)  #Cleaning links via re
    CleanData = re.sub(r'\d', ' ', CleanData)   #Cleaning numbers via re
    CleanData = re.sub('@\S+', ' ', CleanData)  #Cleaning email address via re
    CleanData = re.sub('#\S+\s', ' ', CleanData)    #Cleaning # symbol via re
    CleanData = re.sub('[^a-zA-Z0-9]', ' ', CleanData)  #Cleaning special characters via re
    CleanData = remove_stopwords(CleanData) #Cleaning stopwords via genism 
    CleanData = porter_stemmer.stem(CleanData)  #Stemming via nltk 
    CleanData = lemmatizer.lemmatize(CleanData) #Lemmatization via nltk 
    return CleanData

# Main Web Function
def main():
    # Set page title and background color
    st.set_page_config(page_title='My NLP Project', page_icon=':bar_chart:', layout='wide')

    # Page title and introduction about the project
    st.title('Welcome to My Resume Screening Algorithm')
    st.subheader('By: Ali Vijdaan')
    st.write('This web app was created by me for my AI-221L Project')
    st.write('I use NLP along with ML models like KNN, Naive Bayes and SVM to predict the departments of different Resumes for faster and efficient categorization.')

    # Button for uploading files (PDF or TXT)
    uploaded_file = st.file_uploader('Upload a PDF or TXT file', type=['pdf', 'txt'])
    
    # Processing uploaded file
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError: 
            resume_text = resume_bytes.decode('latin-1')


        # Button to trigger prediction
        if st.button('Press for Prediction'):

            cleaned_resume = dataCleaning(resume_text)
            input_features = tfidf.transform([cleaned_resume])

            category_mapping = {

                6 : "Data Science",
                12 : "HR", 
                0 : "Advocate",
                1 : "Arts",
                24 : "Web Designing",
                16 : "Mechanical Engineer",
                22 : "Sales",
                14 : "Health and fitness",
                5 : "Civil Engineer",
                15 : "Java Developer",
                4 : "Business Analyst",
                21 : "SAP Developer",
                2 : "Automation Testing",
                11 : "Electrical Engineering",
                18 : "Operations Manager",
                20 : "Python Developer",
                8 : "DevOps Engineer",
                17 : "Network Security Engineer",
                19 : "PMO",
                7 : "Database",
                13 : "Hadoop",
                10 : "ETL Developer",
                9 : "DotNet Developer",
                3 : "Blockchain",
                23 : "Testing"
            }

            # Results section for KNN prediction
            st.header('KNN Prediction')
            prediction_id = clf.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
            st.write("Predicted Category: ", category_name)

            # Results section for Naive Bayes prediction
            st.header('Naive Bayes Prediction')
            prediction_id = model.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
            st.write("Predicted Category: ", category_name)


            # Results section for SVM prediction
            st.header('SVM Prediction')
            prediction_id = model_svm.predict(input_features)[0]
            category_name = category_mapping.get(prediction_id, "Unknown")
            st.write("Predicted Category: ", category_name)


#python main
if __name__ == '__main__':
    main()

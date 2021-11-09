import streamlit as st
import pandas as pd
import pickle


st.header('Application of Abalone\'s Age Prediction:')
st.text('6231302011 Pun Kongsomboon')
st.subheader('User Input:')


v_Sex = st.sidebar.radio('Sex', ['Male', 'Female', 'Infant'])
v_Length = st.sidebar.slider('Length', 0.00, 1.00, 0.50)
v_Diameter = st.sidebar.slider('Diameter', 0.00, 1.00, 0.40)
v_Height = st.sidebar.slider('Height', 0.00, 1.00, 0.10)
v_Whole_weight = st.sidebar.slider('Whole Weight', 0.00, 3.00, 0.80)
v_Shucked_Wegiht = st.sidebar.slider('Shucked Wegiht', 0.00, 2.00, 0.30)
v_Viscera_weight = st.sidebar.slider('Viscera Weight', 0.00, 1.00, 0.20)
v_Shell_weight = st.sidebar.slider('Shell Weight', 0.00, 1.00, 0.20)

if v_Sex == 'Male':
    v_Sex = 'M'
elif v_Sex == 'Female':
    v_Sex = 'F'
else:
    v_Sex = 'I'


data = {'Sex': v_Sex,
        'Length': v_Length,
        'Diameter': v_Diameter,
        'Height': v_Height,
        'Whole_weight': v_Whole_weight,
        'Shucked_weight': v_Shucked_Wegiht,
        'Viscera_weight': v_Viscera_weight,
        'Shell_weight': v_Shell_weight}

df = pd.DataFrame(data, index=[0])
st.write(df)

data_sample = pd.read_csv('abalone_sample_data.csv')
df = pd.concat([df, data_sample],axis=0)

cat_data = pd.get_dummies(df[['Sex']])

X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1]

X_new = X_new.drop(columns=['Sex'])

st.subheader('Pre-Processed Input:')
st.write(X_new)

load_nor = pickle.load(open('normalization.pkl', 'rb'))
X_new = load_nor.transform(X_new)

st.subheader('Normalized Input:')
st.write(X_new)

load_knn = pickle.load(open('best_knn.pkl', 'rb'))
prediction = load_knn.predict(X_new)
st.subheader('Prediction:')
st.write(prediction)



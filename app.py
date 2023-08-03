import pandas as pd
import streamlit as st
import pickle

# Load model and scaler for heart model
lr_reg = pickle.load(open('heartmodel6.pkl', 'rb'))
se=pickle.load(open('heartscaling6.pkl', 'rb'))

#load model and scaler for diabetes model
diabetes_reg_xg = pickle.load(open('diabetesmodel2.pkl', 'rb'))
diabetes_se = pickle.load(open('diabetesscaling2.pkl', 'rb'))

#load model and scaler for liver model
liver_svm_reg= pickle.load(open('livermodel2.pkl','rb'))
liver_se=pickle.load(open('liverscaling2.pkl','rb'))




def main():
    st.set_page_config(
        page_title="Health Prediction Web App",
        page_icon=":heartpulse:",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.title("Diseases Predictions Web App")
    st.sidebar.title("Select Disease")

    disease=st.sidebar.selectbox('select disease',['Heart Disease','Diabetes','liver Disease'])
    
    if disease=='Heart Disease':
        
        st.sidebar.title("Heart Disease prediction")
        st.sidebar.write("Fill in the following information:")
        Age = st.sidebar.slider("Age", 29, 77, 54)
        RestingBP = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 131)
        Cholesterol = st.sidebar.slider("Cholesterol (mm/dl)", 126, 564, 246)
        FastingBS = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
        MaxHR = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 149)
        Oldpeak = st.sidebar.slider("Oldpeak (ST Depression)", 0.0, 6.2, 1.0)
        Sex = st.sidebar.selectbox("Sex", ['M', 'F'])
        ChestPainType = st.sidebar.selectbox("ChestPainType", ['ATA', 'NAP', 'ASY', 'TA'])

        RestingECG = st.sidebar.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
        ExerciseAngina = st.sidebar.selectbox("Exercise-Induced Angina", ['N', 'Y'])
        ST_Slope = st.sidebar.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

        if st.sidebar.button("predict"):
            

            user_df = pd.DataFrame(data=[[Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
                                    Sex,ChestPainType, RestingECG, ExerciseAngina, ST_Slope]],
                                columns=['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                                            'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

            # Encode categorical features
            user_df['Sex'] = user_df['Sex'].map({'M': 0, 'F': 1})
            user_df['ChestPainType'] = user_df['ChestPainType'].map({'ATA': 2, 'NAP': 0, 'ASY': 1, 'TA': 3})
            user_df['RestingECG'] = user_df['RestingECG'].map({'Normal': 2, 'ST': 0, 'LVH': 1})
            user_df['ExerciseAngina'] = user_df['ExerciseAngina'].map({'N': 0, 'Y': 1})
            user_df['ST_Slope'] = user_df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

            scaled_data=se.transform(user_df)
            prediction = lr_reg.predict(scaled_data)

            st.subheader("Prediction")
            if prediction[0] == 0:
                st.success("No heart disease detected. You are healthy!")
                st.image("healthy.jpg",width=400, caption="Healthy")
            else:
                st.warning("Heart problem detected. Please consult with a doctor.")
                st.image("doctor_consulant.jpg",width=400, caption="Disease Detected")
    

    elif disease=="Diabetes":
        st.sidebar.title("Diabetes prediction")
        st.sidebar.write("Fill in the following information:")
        gender = st.sidebar.selectbox("Gender", ['Female', 'Male', 'Other'])
        age= st.sidebar.slider('age',1,100,25)
        hypertension=st.sidebar.selectbox('hypertension',[0,1])
        heart_disease=st.sidebar.selectbox('heart_disease',[0,1])
        smoking_history=st.sidebar.selectbox('smoking_history',['never', 'No Info', 'current', 'former', 'ever', 'not current'])
        bmi=st.sidebar.slider("BMI",10.0, 60.0, 20.0)
        HbA1c_level=st.sidebar.slider('HbA1c_level',3.0, 15.0, 5.0)
        blood_glucose_level=st.sidebar.slider('blood_glucose_level',50.0, 400.0, 100.0)

        if st.sidebar.button("predict"):
            diabetes_user_df = pd.DataFrame(data=[[gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]],
                                            columns=['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

            # Encode categorical features for diabetes
            gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
            diabetes_user_df['gender'] = diabetes_user_df['gender'].map(gender_mapping)

            smoking_mapping = {
                'never': 0,
                'No Info': 1,
                'current': 2,
                'former': 3,
                'ever': 4,
                'not current': 5
            }
            diabetes_user_df['smoking_history'] = diabetes_user_df['smoking_history'].map(smoking_mapping)
            diabetes_scaled_data = diabetes_se.transform(diabetes_user_df)
            diabetes_prediction = diabetes_reg_xg.predict(diabetes_scaled_data)

            st.subheader("Diabetes Prediction")
            if diabetes_prediction[0] == 0:
                st.success("No diabetes detected. You are healthy!")
                st.image("healthy.jpg",width=400, caption="Healthy")
            else:
                st.warning("Diabetes detected. Please consult with a doctor.")
                st.image("doctor_consulant.jpg",width=400, caption="Disease Detected")

    
    elif disease=="liver Disease":
        st.sidebar.write("Fill in the following information:")
        st.sidebar.title('liver diseases prediction')
        Age=st.sidebar.slider('Age',4,90,25)
        Gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
        Total_Bilirubin=st.sidebar.slider('Total_Bilirubin',0.4,75.0)
        Direct_Bilirubin=st.sidebar.slider('Direct_Bilirubin',0.1,19.7)
        Alkaline_Phosphotase=st.sidebar.slider('Alkaline_Phosphotase',63,2110)
        Alamine_Aminotransferase=st.sidebar.slider('Alamine_Aminotransferase',10,2000)
        Aspartate_Aminotransferase=st.sidebar.slider('Aspartate_Aminotransferase',10,5000)
        Total_Protiens=st.sidebar.slider('Total_Protiens',2.7,9.6)
        Albumin=st.sidebar.slider('Albumin',0.9,5.5)
        Albumin_and_Globulin_Ratio=st.sidebar.slider('Albumin_and_Globulin_Ratio',0.3,2.8)

        if st.sidebar.button("predict"):
            liver_user_df=pd.DataFrame(data=[[Age, Gender, Total_Bilirubin, Direct_Bilirubin,
                                             Alkaline_Phosphotase, Alamine_Aminotransferase,
                                             Aspartate_Aminotransferase, Total_Protiens, Albumin,
                                             Albumin_and_Globulin_Ratio]],columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                                                'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                                                'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                                                'Albumin_and_Globulin_Ratio'])
            gender_mapping1 = {'Female': 0, 'Male': 1}
            liver_user_df['Gender'] = liver_user_df['Gender'].map(gender_mapping1)

            liver_scaled_data = liver_se.transform(liver_user_df)
            liver_prediction = liver_svm_reg.predict(liver_scaled_data)

            st.subheader("Liver Disease Prediction")
            if liver_prediction[0] == 0:
                st.success("No Liver Disease detected. You are healthy!")
                st.image("healthy.jpg",width=400, caption="Healthy")
            else:
                st.warning("Liver Disease detected. Please consult with a doctor.")
                st.image("doctor_consulant.jpg",width=400, caption="Disease Detected")

    

    



if __name__ == '__main__':
    main()


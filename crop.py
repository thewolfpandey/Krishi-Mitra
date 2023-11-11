import streamlit as st
import numpy as np
import pickle


loaded_model = pickle.load(open('crop_recomendation_model.pkl','rb'))

def crop_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    
    ynew = loaded_model.predict(input_data_reshaped)

    return ynew[0]

# Streamlit app
def main():
    st.title("Krishi Mitra")

    st.write("Enter the following features:")

    # Input fields for features
    n = st.slider("Nitrogen", min_value=0, max_value=140, value=30)
    p = st.slider("Phosphorous", min_value=5, max_value=145, value=30)
    k = st.slider("Potassium", min_value=5, max_value=205, value=30)
    temp = st.number_input("Temperature", min_value=8.82, max_value=43.67, value=15.00, format="%.2f")
    hum = st.number_input("Humidity", min_value=14.25, max_value=99.98, value=15.00, format="%.2f")
    ph = st.number_input("PH", min_value=3.50, max_value=9.93, value=5.00, format="%.2f")
    rain = st.number_input("Rainfall", min_value=20.21, max_value=298.56, value=25.00, format="%.2f")

    crop = ''

    if st.button('Recommend Crop'):
        crop = crop_prediction([n, p, k, temp, hum, ph, rain])
        st.write("Recommended Crop to grow, according to the climatic and soil condition given to the model : ",crop)
    # Getting the recommendation
    # features = [[n, p, k, temperature, humidity, ph, rainfall]]
    # prediction = model.predict(features)[0]

    # st.write(f"Recommended Crop: {prediction}")


if __name__ == "__main__":
    main()
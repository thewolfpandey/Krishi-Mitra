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

    # image_path = "https://imgs.search.brave.com/2gUr1UJX3A-6TxOihOZWoh5xeZ3sQi0kYzpmjxBbIrk/rs:fit:860:0:0/g:ce/aHR0cHM6Ly90My5m/dGNkbi5uZXQvanBn/LzA0LzA3Lzk2LzMy/LzM2MF9GXzQwNzk2/MzIyNV9kaVRwMGtI/QU9GVk1lbE5WdGdy/S1MzczhtZGwzYUtZ/My5qcGc"
    # page_bg_img = f'''
    # <style>
    # .stApp {{
    # background-image: url("{image_path}");
    # background-size: cover;
    # background-repeat: no-repeat;
    # }}
    # </style>
    # '''
    # st.markdown(page_bg_img, unsafe_allow_html=True)


    with st.sidebar:
        st.image('https://imgs.search.brave.com/WdhChz-0DHOwa3vXCaJs42_wYNkoH45sq757CckmrrM/rs:fit:860:0:0/g:ce/aHR0cHM6Ly90My5m/dGNkbi5uZXQvanBn/LzA1LzE0LzM0LzYw/LzM2MF9GXzUxNDM0/NjA2M19nMDY0YmdQ/TlE5SHUyVFBOR0RH/Znd0QmRjdWhQbmNo/NS5qcGc')
        st.title('For the Krishis🌾')
        st.info('with ❤️ by Anmol and Soumya')
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most important aspects of precision agriculture. Crop recommendations are based on a number of factors. Precision agriculture seeks to define these criteria on a site-by-site basis in order to address crop selection issues. While the "site-specific" methodology has improved performance, there is still a need to monitor the systems' outcomes.Precision agriculture systems aren't all created equal. 
            However, in agriculture, it is critical that the recommendations made are correct and precise, as errors can result in significant material and capital loss.

            """)
    
    st.title("Krishi Mitra👨‍🌾")
    # st.write("Enter the following features:")

    col1, col2  = st.columns([20,2])
    
    with col1: 
        '''
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''
    
    st.image("https://imgs.search.brave.com/qJwcXo6o8qfud_OkhrqeenjcMzMTCyLMdgEJQAsUgSQ/rs:fit:860:0:0/g:ce/aHR0cHM6Ly9lb3Mu/Y29tL3dwLWNvbnRl/bnQvdXBsb2Fkcy8y/MDIwLzAzLzFfMTky/MCVEMSU4NTYwMC1l/MTY3MDUwOTUyNzcz/Mi5qcGc", use_column_width="auto")


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
        st.write("Recommended Crop to grow, according to the climatic and soil condition given to the model :-->",crop)

        
    # Getting the recommendation
    # features = [[n, p, k, temperature, humidity, ph, rainfall]]
    # prediction = model.predict(features)[0]

    # st.write(f"Recommended Crop: {prediction}")


if __name__ == "__main__":
    main()
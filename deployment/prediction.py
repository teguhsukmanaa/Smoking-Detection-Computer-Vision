import streamlit as st

def app():
    # Libraries
    from PIL import Image
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image

    # Load Model
    model_path = "CNN_smoking_model.h5"
    model = load_model(model_path)

    # Define function to preprocess input image
    def preprocess_image(img):
        img = img.resize((250, 250))  # Resize image to model's expected input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
        return img_array

    # Define function to make predictions
    def predict_image(img):
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        return prediction[0][0]  # Since it's binary classification, return the probability of being a smoking image

    # Section for deployment interfaces
    st.title("Smoking Image Prediction")
    with st.container(border=True):

        # Allow user to upload an image
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_img is not None:
            # Display the uploaded image
            st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)

            # Make prediction
            img = Image.open(uploaded_img)
            prediction = predict_image(img)

            with st.container(border=True):
                # Display prediction result
                if prediction >= 0.5:
                    st.write("Prediction: Smoking Image")
                else:
                    st.write("Prediction: Not-Smoking Image")

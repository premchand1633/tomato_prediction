import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

st.title("Is this tomato ripe?")
st.write("This is a deep learning classification app to determine if a tomato is ripe or not.")

ripeTomato = Image.open("ripeTomato.jpg").resize((300, 200))
unripeTomato = Image.open("unripeTomato.jpg").resize((300, 200))

# Show and align images horizontally
col1, col2 = st.columns(2)

with col1:
    st.write("Ripe tomatoes")
    st.image(image=ripeTomato, caption="This is a ripe tomato", width=300)

with col2:
    st.write("Unripe tomatoes")
    st.image(image=unripeTomato, caption="This is an unripe tomato", width=300)

st.write(" ")
st.write(" ")
st.write("Upload an image of a tomato to determine if it is ripe or not.")

inputImage = st.file_uploader('', type='jpg', key=6)

if inputImage is not None:
    tomatoImage = Image.open(inputImage)
    st.image(image=tomatoImage, caption="This is the tomato you uploaded", use_column_width=True)
    # Load the model
    try:
        model = load_model('model.h5')

        # Preprocess the image
        image_size = (224, 224)
        tomatoImage = tomatoImage.resize(image_size)
        tomatoImage = np.array(tomatoImage) / 255.0  # Normalize pixel values
        tomatoImage = np.expand_dims(tomatoImage, axis=0)  # Add batch dimension


        if st.button('Predict'):
            # Predict
            prediction = model.predict(tomatoImage)

            # Get the class with the highest probability
            predicted_class = np.argmax(prediction, axis=1)

            # Map the class index to the corresponding label
            if predicted_class > 0.5:
                is_ripe = "unripe"
            else:
                is_ripe = "ripe"

            # Get the probability of the predicted class
            probability = prediction[0][predicted_class[0]]

            st.write(f"This is a: {is_ripe}.")
            st.write(f"Probability of the {is_ripe} tomato is: {probability:.4f}")

    except Exception as e:
        st.write(f"Model prediction probabilities: {prediction}")


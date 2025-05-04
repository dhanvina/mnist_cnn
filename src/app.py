import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('../models/mnist_cnn_model.h5')
    return model

def preprocess_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    # Normalize
    image = image / 255.0
    # Reshape for model input
    image = image.reshape(1, 28, 28, 1)
    return image

def main():
    st.title("✍️ Handwritten Digit Recognition")
    st.markdown("""
    This application uses a Convolutional Neural Network (CNN) to recognize handwritten digits.
    You can either:
    - Draw a digit using the canvas
    - Upload an image of a handwritten digit
    """)

    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.markdown("""
    - Architecture: CNN
    - Input Shape: 28x28x1
    - Training Dataset: MNIST
    - Accuracy: 98.89%
    """)

    # Main content
    tab1, tab2, tab3 = st.tabs(["Draw Digit", "Upload Image", "Model Performance"])
    
    with tab1:
        st.header("Draw a digit")
        canvas_result = st_canvas(
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if canvas_result.image_data is not None:
            if st.button('Predict'):
                # Preprocess the drawn image
                image = preprocess_image(canvas_result.image_data)
                # Load model and predict
                model = load_model()
                prediction = model.predict(image)
                predicted_digit = np.argmax(prediction)
                confidence = float(prediction[0][predicted_digit])
                
                # Display prediction
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### Predicted Digit: {predicted_digit}")
                with col2:
                    st.markdown(f"### Confidence: {confidence:.2%}")
                
                # Display probability distribution
                fig, ax = plt.subplots()
                sns.barplot(x=list(range(10)), y=prediction[0])
                plt.title("Probability Distribution")
                plt.xlabel("Digit")
                plt.ylabel("Probability")
                st.pyplot(fig)
    
    with tab2:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            if st.button('Predict Image'):
                # Convert PIL Image to numpy array
                image = np.array(image)
                # Preprocess the image
                processed_image = preprocess_image(image)
                # Load model and predict
                model = load_model()
                prediction = model.predict(processed_image)
                predicted_digit = np.argmax(prediction)
                confidence = float(prediction[0][predicted_digit])
                
                # Display prediction
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### Predicted Digit: {predicted_digit}")
                with col2:
                    st.markdown(f"### Confidence: {confidence:.2%}")
                
                # Display probability distribution
                fig, ax = plt.subplots()
                sns.barplot(x=list(range(10)), y=prediction[0])
                plt.title("Probability Distribution")
                plt.xlabel("Digit")
                plt.ylabel("Probability")
                st.pyplot(fig)
    
    with tab3:
        st.header("Model Performance")
        st.markdown("""
        ### Model Architecture
        ```python
        Model: Sequential
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        conv2d (Conv2D)             (None, 26, 26, 32)        320       
        conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     
        max_pooling2d (MaxPooling2D)(None, 12, 12, 64)        0         
        dropout (Dropout)           (None, 12, 12, 64)        0         
        flatten (Flatten)           (None, 9216)              0         
        dense (Dense)               (None, 256)               2359552   
        dropout_1 (Dropout)         (None, 256)               0         
        dense_1 (Dense)             (None, 10)                2570      
        =================================================================
        Total params: 2,380,938
        Trainable params: 2,380,938
        Non-trainable params: 0
        ```
        
        ### Performance Metrics
        - Test Accuracy: 98.89%
        - Test Loss: 0.0439
        
        ### Training History
        The model was trained for 10 epochs with a validation split of 0.3 and achieved:
        - Training Accuracy: 98.97%
        - Validation Accuracy: 98.67%
        """)

if __name__ == "__main__":
    main()
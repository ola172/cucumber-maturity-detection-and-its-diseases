import streamlit as st
import cv2
import numpy as np
from backend import CucumberDetection
import tempfile


# Project Title
st.markdown("<h1 style='text-align: center;'>Cucumber Detection Project</h1>", unsafe_allow_html=True)


# Description Section
st.markdown("<h2 >Description</h2>", unsafe_allow_html=True)
st.markdown("""
    <style>
        .expander-header {
            font-size: 50px; /* Adjust this size as needed */
            font-weight: bold;
            color: #2E86C1;
        }

        .description-container {
            font-size: 18px;
            color: #555;
            text-align: justify;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 30px;
            line-height: 1.6;
        }
        .description-container h3 {
            font-size: 26px;
            font-weight: bold;
            color: #2E86C1;
            margin-top: 10px;
        }
        .description-container ul {
            padding-left: 20px;
            list-style-type: disc;
        }
        .cta-text {
            font-size: 20px;
            color: #333;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            background-color: #2E86C1;
            color: white;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 20px;
        }
        .cta-text:hover {
            background-color: #1f6c98;
        }
    </style>
""", unsafe_allow_html=True)

with st.expander("📄 About the project", expanded=True):
    st.markdown("""
        <div class='description-container'>
            <h3>Welcome to the Cucumber Detection Project!</h3>
            This tool leverages computer vision techniques to identify cucumbers in images and 
            identify its maturity level as weel as its leaf disease.
            
            Key Features: 
        
                - Automatically detects cucumbers in uploaded images and Displays bounding boxes around each detected cucumber.
                - Automatically classify cucumber leaf disease. 
                - Easy-to-use interface for quick testing.
            
            
            Try it out by uploading an image below!
        </div>
    """, unsafe_allow_html=True)


# Try the Model Section
st.markdown("<h2 >Try the Model</h2>", unsafe_allow_html=True)


st.markdown("<h3 >Cucumber detection and classifying its maturity level</h3>", unsafe_allow_html=True)

# Intialize Cucumber Fruit maturity model
cucumber_fruit_maturity_model_path = "/media/ola/608411B684118FA0/cv project/models/Cucumber Fruit maturity model/best.pt"
cucumber_fruit_maturity_model = CucumberDetection(model_path=cucumber_fruit_maturity_model_path)

with st.expander("🎯 Try the Model", expanded=True):
    st.write("Upload an image, and the model will detect cucumbers in it and define its maturity level.")

    # File uploader for image input
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="model_1")

    # Detect cucumbers and display results if an image is uploaded
    if uploaded_image is not None:
         # Save the uploaded file temporarily to use cv2.imread
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.read())
            temp_file_path = temp_file.name

        # Read the image with cv2.imread
        image_np = cv2.imread(temp_file_path)
        image_np = cv2.resize(image_np, (640,640))

        # Dectect and get detection result
        result = cucumber_fruit_maturity_model.detect_cucumbers(image=image_np)
        extracted_result = cucumber_fruit_maturity_model.extract_model_result_information(result)
        detected_image = cucumber_fruit_maturity_model.draw_detections(np.array(image_np), extracted_result)

        # Display original and processed images in columns
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(detected_image, caption="Detection Result", use_column_width=True)
    else:
        st.info("📤 Please upload an image to detect cucumbers.")

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)

# Try Cucumber leaf Disease Recognition
st.markdown("<h3 >Cucumber leaf Disease Recognition</h3>", unsafe_allow_html=True)

# Intialize Cucumber Fruit maturity model
cucumber_fruit_maturity_model_path = "/media/ola/608411B684118FA0/cv project/models/Cucumber Disease Recognition model/best.pt"
cucumber_fruit_maturity_model = CucumberDetection(model_path=cucumber_fruit_maturity_model_path)

with st.expander("🎯 Try the Model", expanded=True):
    st.write("Upload an image, and the model will recognise leaf disease in it.")

    # File uploader for image input
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="model_2")

    # Detect cucumbers and display results if an image is uploaded
    if uploaded_image is not None:
         # Save the uploaded file temporarily to use cv2.imread
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_image.read())
            temp_file_path = temp_file.name

        # Read the image with cv2.imread
        image_np = cv2.imread(temp_file_path)
        image_np = cv2.resize(image_np, (640,640))

        # Dectect and get detection result
        result = cucumber_fruit_maturity_model.detect_cucumbers(image=image_np)
        extracted_result = cucumber_fruit_maturity_model.extract_model_result_information(result)

        if  extracted_result:
            label  = extracted_result[0]['label']
        else:
            label = "Uploaded Image has unknown object"

        # Display original and processed images in columns
        col1, col2 = st.columns(2)
        
        st.image(uploaded_image, caption=extracted_result[0]['label'])
    else:
        st.info("📤 Please upload an image to detect cucumbers.")

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)

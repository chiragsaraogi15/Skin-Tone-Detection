import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load the trained model
model = load_model('model.h5')  # Replace with your model path

# List of class names
classes = ['Fair_Light', 'Medium_Tan', 'Dark_Deep']

# Load the MTCNN face detection model
mtcnn = MTCNN()

# Set app title
st.title('Skin Tone Detection App')

# Upload image through file uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

# Button to predict skin tone
if st.button('Predict Skin Tone'):
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)

        # Detect faces
        try:
            faces = mtcnn.detect_faces(image)
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                x, y, w, h = largest_face['box']
                detected_face = image[y:y+h, x:x+w]

                # Resize the detected face to the desired input shape
                detected_face = cv2.resize(detected_face, (120, 90))

                # Preprocess the detected face for classification
                detected_face = detected_face / 255.0  # Normalize pixel values
                detected_face = tf.keras.applications.mobilenet_v2.preprocess_input(detected_face[np.newaxis, ...])

                # Predict the class of the face
                predictions = model.predict(detected_face)
                predicted_class_idx = np.argmax(predictions)
                predicted_class = classes[predicted_class_idx]

                # Display the detected face with a fixed width
                st.image(detected_face, caption='Detected Face', width=200)

                # Display the prediction message
                st.markdown(f'**Looks like your skin tone is:** {predicted_class}')
            else:
                st.write('No face detected in the uploaded image.')
        except Exception as e:
            st.write(f'Error detecting faces: {e}')

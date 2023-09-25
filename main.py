import numpy as np
import streamlit as st
from PIL import Image, ImageOps  # Install pillow instead of PIL
from keras.models import load_model  # TensorFlow is required for Keras to work
import tensorflow

st.title('Medical Plant Classifier Using CNN')

st.subheader('This is a Medical Plant Classifier Using CNN', divider='rainbow')

picture = st.file_uploader("Choose a file")

st.subheader('or')
photo = st.camera_input("Take a picture")

if photo:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(photo)

    with col3:
        st.write(' ')

    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(photo).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    # predictionTop3 = Nmaxelements(prediction[0], 3)
    # print(predictionTop3)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    st.subheader(f'Leaf is {class_name[2:]}')
    print(class_name)
    st.caption(f'Confidence Score:{confidence_score}')
    st.caption(f'Probabilities:{prediction}')



if picture:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.image(picture)

    with col3:
        st.write(' ')

    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(picture).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    # predictionTop3 = Nmaxelements(prediction[0], 3)
    # print(predictionTop3)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

    st.subheader(f'Leaf is {class_name[2:]}')
    print(class_name)
    st.caption(f'Confidence Score:{confidence_score}')
    st.caption(f'Probabilities:{prediction}')

import sqlite3
import numpy as np
# import pandas as pd
import streamlit as st
import tensorflow as tf

from sqlite3 import Connection
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Ovarian Cancer Classifier",
    page_icon="👨‍⚕️",
    # layout="wide",
    initial_sidebar_state="auto",
)

URI_SQLITE_DB = "predictions.db"

def init_db(conn: Connection):
    conn.execute('CREATE TABLE IF NOT EXISTS userstable(PREDICTION TEXT)')
    conn.commit()

def app():
    interpreter = tf.lite.Interpreter(model_path='cancer.tflite')
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(output_details)

    st.subheader("Ovarian Cancer Classifier", divider='grey')
    st.markdown("")
    st.caption("Upload an image to determine the type of Ovarian Cancer.")

    with st.sidebar:
        st.header("Ovarian Cancer?")

        with st.expander("About "):
            st.write("Ovarian cancer is a type of cancer that originates in the ovaries, which are part of the female reproductive system. "
                    "Ovarian cancer often goes undetected until it has progressed to an advanced stage, making it one of the deadliest gynecological cancers.")

        with st.expander("Symptoms and Signs "):
            st.write("Common symptoms of brain tumors include headaches, urinary symptoms, changes in menstraution, bloating or "
                    "swelling, and difficulty eating or getting full quickly.")

        with st.expander("How to Monitor Ovarian Health and Seek Help"):
            st.write("1. Regular medical check-ups and screenings can help monitor Ovarian health.\n"
                    "2. Pay attention to any unusual symptoms and seek medical advice if you notice persistent changes.\n"
                    "3. Early detection and treatment are crucial for better outcomes.")


    file = st.file_uploader("Please upload your Scan", type=["png", "jpg", "jpeg"])

    conn = get_connection(URI_SQLITE_DB)
    init_db(conn)

    def import_and_predict(image_data):
        size = (256, 256)
        image1 = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        image1 = image1.convert('RGB')
        img = np.array(image1) / 255.0
        img_reshape = img[np.newaxis, ...]

        # Prepare input data for TensorFlow Lite model
        interpreter.set_tensor(input_details[0]['index'], img_reshape.astype(np.float32))
        interpreter.invoke()

        # Get the output from TensorFlow Lite model
        prediction = interpreter.get_tensor(output_details[0]['index'])

        return prediction

    labels = ['EC', 'CC', 'LGSC', 'MC', 'HGSC']


    if file is not None:
        image = Image.open(file)
        st.image(image, width=300)
        predictions = import_and_predict(image)
        predictions = np.argmax(predictions)
        predictions = labels[predictions]
        string = "The patient most likely has " + predictions
        st.success(string)

@st.cache_resource
def get_connection(path: str):
    return sqlite3.connect(path, check_same_thread=False)

if __name__ == '__main__':
    app()

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(r"C:\Users\DELL\Agenti AI Internship\Day 6\jaundice_cnn_model85.h5")

IMG_SIZE = 224

def predict_jaundice(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    if pred > 0.5:
        result = "🟡 Jaundice Detected"
        confidence = pred * 100
    else:
        result = "🟢 No Jaundice Detected"
        confidence = (1 - pred) * 100

    return (
        result,
        f"{confidence:.2f}%"
    )

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🩺 AI-Based Jaundice Detection System
        
        Upload an image to predict whether jaundice is present.
        """
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")

    predict_btn = gr.Button("Predict")

    prediction_output = gr.Textbox(label="Prediction")
    confidence_output = gr.Textbox(label="Confidence")

    predict_btn.click(
        fn=predict_jaundice,
        inputs=image_input,
        outputs=[prediction_output, confidence_output]
    )

demo.launch()
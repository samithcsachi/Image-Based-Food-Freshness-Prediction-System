import gradio as gr
from src.pipeline.prediction_pipeline import PredictionPipeline
import numpy as np
from PIL import Image

pipeline = PredictionPipeline()

def predict_single(image):
    
    if image is None:
        return None, "No image detected!", "No image detected!"
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    result = pipeline.predict(img)
    annotated_img = pipeline.annotate(img, result)
    return annotated_img, result["category"], result["freshness"]

with gr.Blocks() as demo:
    gr.Markdown("# Food Freshness Detection")

    with gr.Tab("Image Upload"):
        image = gr.Image(sources=["upload"], label="Upload an Image")
        out_img = gr.Image()
        cat = gr.Textbox(label="Category")
        fresh = gr.Textbox(label="Freshness")
        btn = gr.Button("Predict on Image")
        btn.click(predict_single, inputs=image, outputs=[out_img, cat, fresh])

        
    with gr.Tab("Live Webcam"):
        webcam = gr.Image(sources=["webcam"], label="Webcam")
        out_img = gr.Image()
        cat = gr.Textbox(label="Category")
        fresh = gr.Textbox(label="Freshness")
        btn = gr.Button("Predict")
        btn.click(predict_single, inputs=webcam, outputs=[out_img, cat, fresh])


    

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

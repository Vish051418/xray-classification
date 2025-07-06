import gradio as gr
from predict import XRayPredictor

predictor = XRayPredictor("models/best_model.pth")

def classify_image(image):
    image.save("temp.jpg")
    result = predictor.predict("temp.jpg")
    return result

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="X-Ray Classification",
    description="Upload a chest X-ray image to classify as COVID19, NORMAL, or PNEUMONIA"
)

iface.launch()
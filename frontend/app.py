import gradio as gr
from PIL import Image
import requests
import os
import io

# Get the API URL from the environment variable set in docker-compose
API_URL = os.getenv("API_URL", "http://localhost:8000")
ANALYZE_ENDPOINT = f"{API_URL}/api/analyze"

def analyze_review(text, image):
    """
    Function to send data to the backend API and get a response.
    """
    if not text:
        return "Please provide some review text.", None

    files = {}
    if image is not None:
        # Gradio provides the image as a numpy array -> save it to a byte stream
        pil_image = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        files = {'image': ('review_image.png', img_byte_arr, 'image/png')}

    try:
        # The 'text' parameter is sent as form data, not JSON
        response = requests.post(ANALYZE_ENDPOINT, data={"text": text}, files=files)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()

        output_text = (
            f"Sentiment: {data.get('sentiment', 'N/A')}\n"
            f"Confidence: {data.get('confidence', 0.0):.2f}\n"
            f"Modality: {data.get('modality', 'N/A')}\n"
            f"Processing Time: {data.get('processing_time_ms', 0.0):.0f} ms"
        )
        return output_text, None # Second return value is for an output component to show raw JSON if needed

    except requests.exceptions.RequestException as e:
        return f"Error connecting to the API: {e}", None

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Multimodal Sentiment Analysis")
    gr.Markdown("Enter a product review text and optionally upload an image to analyze its sentiment.")

    with gr.Row():
        with gr.Column():
            review_text = gr.Textbox(lines=5, label="Review Text")
            review_image = gr.Image(type="numpy", label="Review Image (Optional)")
            submit_button = gr.Button("Analyze Sentiment")
        with gr.Column():
            output_label = gr.Textbox(label="Analysis Result", interactive=False)
            json_output = gr.JSON(label="Full API Response") # To see the raw JSON

    submit_button.click(
        fn=analyze_review,
        inputs=[review_text, review_image],
        outputs=[output_label, json_output] # Map outputs to components
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
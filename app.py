from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)

# Load your model
model = load_model('model-unet.h5')

def process_prediction(image, prediction):
    # Apply a threshold to convert probabilities to binary values
    threshold = 0.5
    binary_mask = (prediction > threshold).astype(np.uint8)

    # Convert binary mask to PIL image
    mask_image = Image.fromarray(np.squeeze(binary_mask * 255))

    # Resize mask to match the original image size
    mask_image = mask_image.resize(image.size)

    # Overlay the mask on the original image
    overlay_image = ImageOps.colorize(mask_image, 'black', 'red')
    result_image = Image.blend(image.convert('RGBA'), overlay_image.convert('RGBA'), alpha=0.5)

    return result_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Read the image
            image_stream = io.BytesIO(uploaded_file.read())
            image = Image.open(image_stream)

            # Convert to grayscale if the image has 3 channels (RGB)
            if image.mode == 'RGB':
                image = image.convert('L')

            # Preprocess the image (resize, convert to array, normalize, etc.)
            original_image = image.resize((128, 128))
            image_array = np.array(original_image) / 255.0
            image_array = np.expand_dims(image_array, axis=-1)  # Ensure grayscale shape
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Predict using the preprocessed image
            prediction = model.predict(image_array)

            # Process the prediction into a readable format
            processed_image = process_prediction(original_image, prediction)
            processed_image.save('static/predicted_image.png')  # Save the processed image

            return render_template('index.html', prediction='static/predicted_image.png')

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)

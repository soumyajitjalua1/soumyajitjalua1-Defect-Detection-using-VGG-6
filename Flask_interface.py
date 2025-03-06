import os
import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('vgg16_finetuned_model.h5')

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Function to preprocess an image for VGG16
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to draw bounding boxes and save coordinates
def process_image(img_path, output_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Run inference (replace with your actual bounding box prediction logic)
    prediction = model.predict(img_array)
    annotation = float(prediction[0][0])  # Example: Get annotation
    
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    
    # Example: Draw a bounding box (replace with actual bounding box coordinates)
    x1, y1, x2, y2 = 50, 50, 200, 200  # Placeholder coordinates
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
    cv2.putText(img, f'Annotation: {annotation:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save the annotated image
    cv2.imwrite(output_path, img)
    
    # Return bounding box coordinates
    return (x1, y1, x2, y2), annotation

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        
        # Save the uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(upload_path)
        
        # Process the image and get bounding box coordinates
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], file.filename)
        (x1, y1, x2, y2), annotation = process_image(upload_path, output_path)
        
        # Save coordinates to a CSV file
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], 'coordinates.csv')
        data = {
            'filename': [file.filename],
            'x1': [x1],
            'y1': [y1],
            'x2': [x2],
            'y2': [y2],
            'annotation': [annotation]
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        # Render the result page
        return render_template('result.html', 
                              image_file=file.filename, 
                              coordinates=(x1, y1, x2, y2), 
                              annotation=annotation)
    
    return render_template('index.html')

# Route to display the annotated image
@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# Route to download the CSV file
@app.route('/download_csv')
def download_csv():
    csv_path = os.path.join(app.config['OUTPUT_FOLDER'], 'coordinates.csv')
    return send_from_directory(app.config['OUTPUT_FOLDER'], 'coordinates.csv', as_attachment=True)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
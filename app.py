# app.py
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np
import pandas as pd
from model.preprocessing import preprocess_input, preprocess_csv
import matplotlib.pyplot as plt
import joblib
import os

plt.switch_backend('Agg')
# Initialize Flask app
app = Flask(__name__)

# Set up image folder
IMAGE_FOLDER = os.path.join('images')
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# Load the trained model
model_path = "model/model.pkl"
model = joblib.load(model_path)

# Home route - HTML upload form
@app.route('/')
def home():
    # Get the image file from the images folder
    #image_url = url_for('static', filename='images/prediction_image.png')
    return render_template('index.html', image_url=None)

# Route to handle file upload and prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Check if a file is part of the POST request
            if 'file' not in request.files:
                return jsonify({"error": "No file part in the request"}), 400

            file = request.files['file']
            
            # Check if the file is empty
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400

            # Process CSV and make predictions
            if file and file.filename.endswith('.csv'):
                # Preprocess the CSV file
                data_feateng,features,targets, mean, std = preprocess_csv(file)
                data = data_feateng[features]
                target = data_feateng[targets]
                # Make predictions
                predictions = model.predict(data)
                
                test_df = target[["target_t1"]]*std+mean
                test_df["pred_t1"] = predictions*std+mean
                test_df.index = test_df.index.date
            
                forecast_range = pd.date_range(start=np.max(test_df.index.values), periods=1440*30, freq="min")
                forecast = []
                for t in range(0, 1440*30):
                    forecast.append(predictions[(t)]*std+mean)
                #print(forecast)
                test_df["target_t1"].plot()
                plt.plot(forecast_range, forecast, c="r", alpha=0.5, label="forecasting")
                plt.ylabel("(MWh)")
                plt.xticks(rotation='vertical')
                plt.title("Forecasting Daily Electricity Consumption (MWh)")
                plt.savefig(os.path.join('static', 'images', 'prediction_image.png'))
                plt.close()
                # Return predictions to the webpage along with image
                return render_template('index.html', 
                                    predictions=predictions.tolist(), 
                                    image_url=url_for('static', filename='images/prediction_image.png'))

            else:
                return jsonify({"error": "Invalid file type, please upload a CSV file."}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
    # For GET requests, render the page without the image
    return render_template('index.html', image_url=None)

if __name__ == '__main__':
    app.run(debug=True)

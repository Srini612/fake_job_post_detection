from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
import pandas as pd
import logging
from fake_job_detector import FakeJobDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Ensure directories exist
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_PATH'] = os.path.join(MODEL_FOLDER, 'fake_job_detector.pkl')

# Initialize the model
detector = FakeJobDetector()

# Try to load the model, if it doesn't exist, train it
try:
    detector.load_model(app.config['MODEL_PATH'])
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.warning("Model not found. Training a new model...")
    dataset_path = "fake_job_postings.csv"
    detector = FakeJobDetector(dataset_path)
    detector.save_model(app.config['MODEL_PATH'])
    logger.info("New model trained and saved")
except Exception as e:
    logger.error(f"Error loading model: {e}")

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for single job prediction"""
    try:
        # Get form data
        job_description = request.form.get('job_description', '')
        
        # Additional job information
        additional_info = {
            'job_id': request.form.get('job_id', 0),
            'location': request.form.get('location', 'unknown'),
            'department': request.form.get('department', 'unknown'),
            'salary_range': request.form.get('salary_range', 'unknown'),
            'has_company_logo': int(request.form.get('has_company_logo', 0)),
            'telecommuting': int(request.form.get('telecommuting', 0)),
            'employment_type': request.form.get('employment_type', 'unknown')
        }
        
        # Make prediction
        result = detector.predict_job_post(job_description, additional_info)
        
        logger.info(f"Prediction result: {result['prediction_text']} with confidence {result['confidence']}%")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred while processing the job post'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """API endpoint for batch predictions from CSV file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the uploaded file (CSV)
        try:
            df = pd.read_csv("fake_job_postings.csv")
            
            # Predict for each job description
            results = []
            for _, row in df.iterrows():
                job_description = row.get('description', '')
                result = detector.predict_job_post(job_description)
                results.append({
                    'description': job_description,
                    'prediction': result
                })
            
            logger.info(f"Processed {len(results)} records from uploaded file")
            return jsonify(results)
        
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            return jsonify({'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    """API endpoint to get model information"""
    info = detector.get_model_info()
    return jsonify(info)

@app.route('/api/health')
def health_check():
    """API endpoint for health check"""
    return jsonify({'status': 'healthy', 'model_loaded': detector.pipeline is not None})

# Only run the app if this file is executed directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
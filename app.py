import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ml_models import analyze_handwriting_real
from model_pipeline import generate_quiz  # Keep quiz generation from old pipeline

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__name__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # If the user does not select a file
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Get optional extracted text if provided
            extracted_text = request.form.get('text', '')
            
            # Use REAL ML-based analysis (no random noise)
            result = analyze_handwriting_real(filepath, extracted_text)
            
            # Clean up the file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify({
                "dyslexia_score": result.get("dyslexia_score", 0.0),
                "dyslexia_details": result.get("dyslexia_details", {}),
                "dysgraphia_score": result.get("dysgraphia_score", 0.0),
                "dysgraphia_details": result.get("dysgraphia_details", {}),
                "analysis_type": result.get("analysis_type", "Real ML Analysis"),
                "features_extracted": result.get("features_extracted", {}),
                "success": "error" not in result
            }), 200
            
        except Exception as e:
             # Clean up the file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e), "dyslexia_score": 0.0, "dysgraphia_score": 0.0}), 500

    return jsonify({"error": "Invalid file type"}), 400

@app.route('/analyze/external', methods=['POST'])
def analyze_external():
    # This route uses the same real ML analysis 
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            extracted_text = request.form.get('text', '')
            result = analyze_handwriting_real(filepath, extracted_text)
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                "dyslexia_score": result.get("dyslexia_score", 0.0),
                "dyslexia_details": result.get("dyslexia_details", {}),
                "dysgraphia_score": result.get("dysgraphia_score", 0.0),
                "dysgraphia_details": result.get("dysgraphia_details", {}),
                "analysis_type": result.get("analysis_type", "Real ML Analysis"),
                "features_extracted": result.get("features_extracted", {}),
                "success": "error" not in result
            }), 200
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e), "dyslexia_score": 0.0, "dysgraphia_score": 0.0}), 500

    return jsonify({"error": "Invalid file type"}), 400


@app.route('/quiz/generate', methods=['POST'])
def generate_quiz_route():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request JSON body required"}), 400

    topic = payload.get('topic', 'General Knowledge')
    text = payload.get('text', '')
    question_count = int(payload.get('question_count', 5))

    quiz = generate_quiz(topic, text, question_count)
    return jsonify(quiz), 200


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Start the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

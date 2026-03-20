import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ml_models import analyze_handwriting_real
from model_pipeline import generate_quiz  # Keep quiz generation from old pipeline
from paper_validator import validate_paper_image

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
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
            # Validate that the image contains handwriting on paper
            skip_validation = request.form.get('skip_validation', 'false').lower() == 'true'

            if not skip_validation:
                validation = validate_paper_image(filepath)
                if not validation['is_valid']:
                    # Clean up and return validation error
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return jsonify({
                        "error": "Invalid image",
                        "validation_error": True,
                        "reason": validation['reason'],
                        "confidence": validation['confidence'],
                        "message": "Please upload a clear image of handwriting on paper. Photos of objects, nature, or non-document images are not accepted.",
                        "details": validation.get('details', {})
                    }), 400

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
            # Validate that the image contains handwriting on paper
            skip_validation = request.form.get('skip_validation', 'false').lower() == 'true'

            if not skip_validation:
                validation = validate_paper_image(filepath)
                if not validation['is_valid']:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return jsonify({
                        "error": "Invalid image",
                        "validation_error": True,
                        "reason": validation['reason'],
                        "confidence": validation['confidence'],
                        "message": "Please upload a clear image of handwriting on paper. Photos of objects, nature, or non-document images are not accepted.",
                        "details": validation.get('details', {})
                    }), 400

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


@app.route('/quiz/analyze', methods=['POST'])
def analyze_quiz_performance():
    """
    Analyze quiz attempt results and provide learning insights.

    Expected payload:
    {
        "studentId": "student123",
        "quizId": "quiz456",
        "topic": "Grammar",
        "totalQuestions": 5,
        "correctAnswers": 3,
        "score": 60.0,
        "totalTimeMs": 180000,
        "questionResponses": [
            {
                "questionId": "q1",
                "questionText": "What is...?",
                "correctAnswer": "A",
                "studentAnswer": "B",
                "isCorrect": false,
                "responseTimeMs": 30000
            }
        ]
    }

    Returns learning gap analysis, strong/weak areas, and recommendations.
    """
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request JSON body required"}), 400

    try:
        topic = payload.get('topic', 'General')
        total_questions = payload.get('totalQuestions', 0)
        correct_answers = payload.get('correctAnswers', 0)
        score = payload.get('score', 0.0)
        total_time_ms = payload.get('totalTimeMs', 0)
        question_responses = payload.get('questionResponses', [])

        # Analyze performance patterns
        analysis = analyze_quiz_results(
            topic=topic,
            total_questions=total_questions,
            correct_answers=correct_answers,
            score=score,
            total_time_ms=total_time_ms,
            question_responses=question_responses
        )

        return jsonify(analysis), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "learningGapSummary": "Unable to analyze quiz performance.",
            "strongAreas": [],
            "weakAreas": [],
            "recommendation": "Please try again later."
        }), 500


def analyze_quiz_results(topic, total_questions, correct_answers, score, total_time_ms, question_responses):
    """
    Analyze quiz performance and generate learning insights.
    """
    # Calculate metrics
    accuracy = score / 100.0 if score else 0
    avg_time_per_question = (total_time_ms / total_questions / 1000) if total_questions > 0 else 0
    incorrect_count = total_questions - correct_answers

    # Analyze response time patterns
    fast_responses = []
    slow_responses = []
    incorrect_questions = []

    for qr in question_responses:
        response_time_sec = qr.get('responseTimeMs', 0) / 1000
        if qr.get('isCorrect'):
            if response_time_sec < 15:
                fast_responses.append(qr)
        else:
            incorrect_questions.append(qr)
        if response_time_sec > 45:
            slow_responses.append(qr)

    # Determine performance level
    if score >= 90:
        performance_level = "EXCELLENT"
    elif score >= 70:
        performance_level = "GOOD"
    elif score >= 50:
        performance_level = "NEEDS_IMPROVEMENT"
    else:
        performance_level = "STRUGGLING"

    # Generate learning gap summary
    learning_gap_summary = generate_learning_summary(
        topic=topic,
        score=score,
        performance_level=performance_level,
        accuracy=accuracy,
        avg_time=avg_time_per_question,
        incorrect_count=incorrect_count,
        fast_correct=len(fast_responses),
        slow_count=len(slow_responses)
    )

    # Identify strong and weak areas
    strong_areas = []
    weak_areas = []

    if len(fast_responses) / max(correct_answers, 1) > 0.5:
        strong_areas.append(f"Quick recall in {topic}")
    if accuracy >= 0.8:
        strong_areas.append(f"Good understanding of {topic} concepts")
    if avg_time_per_question < 30 and accuracy >= 0.6:
        strong_areas.append("Efficient test-taking skills")

    if incorrect_count > total_questions * 0.4:
        weak_areas.append(f"Core {topic} concepts need review")
    if len(slow_responses) > total_questions * 0.3:
        weak_areas.append("Time management during tests")
    if accuracy < 0.5:
        weak_areas.append(f"Fundamental {topic} knowledge gaps")

    # Generate recommendation
    recommendation = generate_recommendation(performance_level, topic, weak_areas)

    return {
        "performanceLevel": performance_level,
        "learningGapSummary": learning_gap_summary,
        "strongAreas": strong_areas if strong_areas else ["Keep practicing!"],
        "weakAreas": weak_areas if weak_areas else [],
        "recommendation": recommendation,
        "metrics": {
            "accuracy": round(accuracy * 100, 1),
            "avgTimePerQuestion": round(avg_time_per_question, 1),
            "fastCorrectAnswers": len(fast_responses),
            "slowResponses": len(slow_responses)
        }
    }


def generate_learning_summary(topic, score, performance_level, accuracy, avg_time, incorrect_count, fast_correct, slow_count):
    """Generate a human-readable learning summary."""

    if performance_level == "EXCELLENT":
        summary = f"Excellent performance in {topic}! "
        if fast_correct > 0:
            summary += f"Strong quick recall ability with {fast_correct} rapid correct answers. "
        summary += "Keep up the great work and consider helping classmates who may be struggling."

    elif performance_level == "GOOD":
        summary = f"Good understanding of {topic} demonstrated. "
        if incorrect_count > 0:
            summary += f"Review the {incorrect_count} missed question(s) to strengthen comprehension. "
        if slow_count > 0:
            summary += "Practice can help improve response speed."

    elif performance_level == "NEEDS_IMPROVEMENT":
        summary = f"Some {topic} concepts need more practice. "
        summary += f"Scored {score:.0f}% with {incorrect_count} incorrect answers. "
        summary += "Focused review of the missed topics is recommended."

    else:  # STRUGGLING
        summary = f"Significant gaps in {topic} understanding identified. "
        summary += "Recommend one-on-one support and foundational concept review. "
        summary += "Breaking down topics into smaller parts may help."

    return summary


def generate_recommendation(performance_level, topic, weak_areas):
    """Generate actionable recommendation based on performance."""

    if performance_level == "EXCELLENT":
        return f"Challenge yourself with advanced {topic} materials or help tutor other students."

    elif performance_level == "GOOD":
        if weak_areas:
            return f"Focus on: {', '.join(weak_areas[:2])}. Practice with additional exercises."
        return f"Continue practicing {topic} to maintain and improve your skills."

    elif performance_level == "NEEDS_IMPROVEMENT":
        return f"Schedule dedicated study time for {topic}. Review class notes and try practice problems daily."

    else:  # STRUGGLING
        return f"Request additional support from teacher. Consider breaking {topic} into smaller, manageable sections."


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/validate', methods=['POST'])
def validate_image():
    """
    Validate if an uploaded image contains handwriting on paper.
    Returns validation result without performing full analysis.
    """
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
            validation = validate_paper_image(filepath)

            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                "is_valid": validation['is_valid'],
                "confidence": validation['confidence'],
                "reason": validation['reason'],
                "details": validation.get('details', {}),
                "message": "Image is valid for analysis" if validation['is_valid']
                          else "Please upload a clear image of handwriting on paper"
            }), 200

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e), "is_valid": False}), 500

    return jsonify({"error": "Invalid file type", "is_valid": False}), 400

if __name__ == '__main__':
    # Start the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

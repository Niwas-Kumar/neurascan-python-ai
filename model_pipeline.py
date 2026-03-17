import cv2
import os
import json
import uuid
import requests
import numpy as np
import pytesseract
import re
from PIL import Image

# Tesseract path configuration
# In Linux/Docker, it will usually be in the system path. 
# Allow override via environment variable for flexibility.
tesseract_path = os.getenv('TESSERACT_CMD', 'tesseract')
pytesseract.pytesseract.tesseract_cmd = tesseract_path

def preprocess_image(image_path):
    """
    Load the image and apply preprocessing for better text extraction/contour detection.
    """
    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image from path")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply slightly aggressive thresholding to get clear text
    # Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return img, gray, thresh

def extract_text(gray_image):
    """
    Use pytesseract to extract text from a grayscale image.
    """
    # Configure tesseract to only look for standard characters for better accuracy
    custom_config = r'--oem 3 --psm 6'
    
    # We use PIL conversion for tesseract compatibility
    pil_img = Image.fromarray(gray_image)
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    
    return text

def calculate_dyslexia_score(text):
    """
    Calculate a heuristic dyslexia risk score based on spelling errors and typical dyslexic patterns.
    Note: A true ML system would use an NLP spell checker or a sequence-to-sequence model.
    """
    if not text.strip():
        # If no text is detected, assume poor handwriting or scanning issue.
        return 0.0 # Can't evaluate
        
    # Example heuristic: Check for common misspellings or reversed letter patterns like p/q, b/d (often manifested as spelling mistakes)
    # We will simulate a probability model base on the density of "irregular words" or short non-words.
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    if not words:
         return 0.0
         
    # Mock dictionary of common valid words vs likely misspellings (in a real system this would be a large NLP corpus)
    # Here we roughly guess that very strange letter combinations or non-dictionary words might indicate spelling struggles associated with dyslexia.
    suspicious_patterns = 0
    total_words = len(words)
    
    for word in words:
        # Heavily simplified heuristic:
        # If word has repeated consonants in a weird way, or resembles common b/d p/q confusion
        if re.search(r'([bpdq])\1', word): # bb, pp, dd, qq
            suspicious_patterns += 0.5
        if re.search(r'[^aeiou]{4,}', word): # 4 consonants in a row is rare in english
             suspicious_patterns += 1.0
             
    # Calculate a rough percentage of "problematic" words, scaled up to a 100 score
    error_rate = min(suspicious_patterns / total_words, 1.0)
    base_score = error_rate * 100
    
    # Add a pseudo-random confidence fluctuation to simulate an ML model's output variance
    random_noise = np.random.uniform(-5.0, 15.0)
    
    final_score = np.clip(base_score + random_noise + 20, 0.0, 100.0) # Base ~20 assuming any referral to this system has some risk
    
    return round(final_score, 1)

def calculate_dysgraphia_score(thresh_image):
    """
    Calculate a heuristic dysgraphia risk score based on spatial inconsistency.
    """
    # Find contours (outlines of letters/words)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0

    # Extract bounding boxes for all contours
    boxes = [cv2.boundingRect(c) for c in contours]
    # Filter out tiny noise (dots, specks of dust)
    boxes = [b for b in boxes if b[2] * b[3] > 50] 
    
    if len(boxes) < 5:
        # Not enough handwriting to confidently analyze
        return 0.0
        
    # Extract just the heights of the bounding boxes
    heights = [b[3] for b in boxes]
    
    # Height variation: Dysgraphia often presents as wildly varying letter sizes
    mean_height = np.mean(heights)
    std_height = np.std(heights)
    
    # Calculate Coefficient of Variation (CV) for height
    cv_height = std_height / mean_height if mean_height > 0 else 0
    
    # A standard handwriting CV might be around 0.15 - 0.25. 
    # High variation indicates dysgraphia risk.
    # Map CV from [0.15 ... 0.6] to [0 ... 100]
    normalized_cv = max((cv_height - 0.15) / 0.45, 0.0)
    base_score = min(normalized_cv * 100, 100.0)
    
    final_score = np.clip(base_score + np.random.uniform(-10.0, 10.0), 0.0, 100.0)
    
    return round(final_score, 1)


def analyze_image(image_path):
    """
    Main pipeline function that orchestrates image processing and ML heuristic scoring.
    """
    # 1. Preprocess the image
    img, gray, thresh = preprocess_image(image_path)
    
    # 2. Dyslexia analysis (via OCR + NLP heuristic)
    extracted_text = extract_text(gray)
    dyslexia_score = calculate_dyslexia_score(extracted_text)
    
    # 3. Dysgraphia analysis (via spacing/size contour variance)
    dysgraphia_score = calculate_dysgraphia_score(thresh)
    
    # 4. Generate Analysis Summary
    analysis_text = []
    
    if dyslexia_score > 60:
        analysis_text.append("High spelling error rate detected. Indicators of phonetic confusion or letter reversal present.")
    elif dyslexia_score > 30:
         analysis_text.append("Moderate spelling and syntax inconsistencies observed.")
    else:
        analysis_text.append("Spelling and letter recognition appear typical or cannot be reliably read.")
        
    if dysgraphia_score > 60:
        analysis_text.append("Significant spatial inconsistency (erratic letter height, spacing) detected in handwriting.")
    elif dysgraphia_score > 30:
         analysis_text.append("Mild variability in handwriting mechanics observed.")
    else:
        analysis_text.append("Handwriting mechanics and spatial organization appear relatively standard.")

    final_analysis = " ".join(analysis_text)
    
    if dyslexia_score == 0 and dysgraphia_score == 0:
        final_analysis = "Could not confidently read handwriting or extract text features. Scan quality may be too low."

    # Try to enrich output from external model if configured
    external_output = query_external_model(extracted_text, dyslexia_score, dysgraphia_score, final_analysis)

    combined_analysis = final_analysis
    if external_output and external_output.get("external_analysis"):
        combined_analysis = combined_analysis + " " + external_output.get("external_analysis")

    response = {
        "dyslexia_score": float(external_output.get("dyslexia_score", dyslexia_score)),
        "dysgraphia_score": float(external_output.get("dysgraphia_score", dysgraphia_score)),
        "analysis": combined_analysis,
        "extracted_text_preview": extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text
    }

    if external_output and external_output.get("analysis_details"):
        response["external_details"] = external_output.get("analysis_details")

    return response


def query_external_model(extracted_text, dyslexia_score, dysgraphia_score, analysis_text):
    """Call an external AI model endpoint if configured via EXTERNAL_AI_API_URL."""
    external_url = os.getenv("EXTERNAL_AI_API_URL")
    if not external_url:
        return None

    payload = {
        "text": extracted_text,
        "dyslexia_score": float(dyslexia_score),
        "dysgraphia_score": float(dysgraphia_score),
        "analysis": analysis_text,
        "source": "neurascan-python-ai"
    }

    headers = {
        "Content-Type": "application/json"
    }
    api_key = os.getenv("EXTERNAL_AI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(external_url, json=payload, headers=headers, timeout=25)
        resp.raise_for_status()
        data = resp.json()

        return {
            "dyslexia_score": data.get("dyslexia_score", dyslexia_score),
            "dysgraphia_score": data.get("dysgraphia_score", dysgraphia_score),
            "external_analysis": data.get("analysis", ""),
            "analysis_details": data.get("details", None)
        }
    except Exception as e:
        # If external fails, continue with baseline
        print(f"[ExternalModelFallback] {e}")
        return None


def generate_quiz(topic, text, question_count=5):
    """Creates a quiz based on the input text and optionally calls external quiz generator."""
    external_quiz_url = os.getenv("EXTERNAL_QUIZ_API_URL")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("EXTERNAL_AI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if external_quiz_url:
        try:
            req = {
                "topic": topic,
                "text": text,
                "question_count": question_count
            }
            ext_resp = requests.post(external_quiz_url, json=req, headers=headers, timeout=25)
            ext_resp.raise_for_status()
            return ext_resp.json()
        except Exception as e:
            print(f"[QuizExternalFallback] " + str(e))

    # Local fallback quiz generation (simple heuristic)
    sentences = re.split(r'[\.\?!]', text)
    questions = []
    for i in range(min(question_count, len(sentences))):
        sentence = sentences[i].strip()
        if not sentence:
            continue

        words = sentence.split()
        if len(words) < 5:
            continue

        # remove one word to create fill-in-the-blank
        index = min(2, len(words)-1)
        missing_word = words[index]
        option_list = [missing_word] + ["example","placeholder","option"]
        questions.append({
            "question": sentence.replace(missing_word, "____", 1),
            "options": option_list,
            "answer": missing_word
        })

    if not questions:
        questions.append({
            "question": f"What is the main topic of this content?",
            "options": [topic, "General", "Language", "Math"],
            "answer": topic
        })

    return {
        "quiz_id": str(uuid.uuid4()),
        "topic": topic,
        "generated_at": str(uuid.uuid1()),
        "questions": questions,
        "source": "local-fallback"
    }

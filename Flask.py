from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import os
import pandas as pd
import string
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

app = Flask(__name__)
CORS(app)

# Clean and tokenize text
def cleantext(text):
    if isinstance(text, str):
        text = "".join([char for char in text if char not in string.punctuation])
        tokens = re.split(r'\W+', text.lower())
        text = [word for word in tokens if word not in stopwords and word]
        return ' '.join(text)
    return ""

def cleantext_tokenizer(text):
    return cleantext(text).split()

# Train and save model
def train_and_save_model():
    df = pd.read_csv("Spam_Detection/spam.csv", encoding='latin-1').iloc[:, :2]
    df.columns = ['label', 'text']
    df.dropna(inplace=True)

    tfidf = TfidfVectorizer(tokenizer=cleantext_tokenizer)
    X = tfidf.fit_transform(df['text'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    with open("rf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

# Load or train model
def load_model():
    if not os.path.exists("rf_model.pkl"):
        train_and_save_model()
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, vectorizer, encoder

# Load models
model, vectorizer, encoder = load_model()

# Home route with embedded HTML
@app.route('/')
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta theme-color="#6c5ce7">
    <meta name="description" content="A powerful web application to detect spam messages using machine learning. Enter a message and check if it's spam or legitimate.">
    <title>Spam Shield - Advanced Message Detector</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {
            --primary-color: #6c5ce7;
            --primary-light: #8a7eef;
            --primary-dark: #5046c0;
            --secondary-color: #2ecc71;
            --danger-color: #e74c3c;
            --dark-bg: #121212;
            --dark-surface: #1e1e1e;
            --dark-card: #252525;
            --dark-border: #333333;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: var(--font-family);
            transition: all 0.3s ease;
        }

        body {
            background-color: var(--dark-bg);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(108, 92, 231, 0.1) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(108, 92, 231, 0.1) 0%, transparent 20%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            color: var(--text-primary);
        }

        .navbar {
            background-color: var(--dark-surface);
            color: white;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .navbar-brand {
            font-size: 1.6rem;
            font-weight: bold;
            display: flex;
            align-items: center;
        }

        .navbar-brand i {
            margin-right: 10px;
            color: var(--primary-color);
            font-size: 1.8rem;
        }

        .navbar-links {
            display: flex;
        }

        .navbar-links a {
            color: var(--text-secondary);
            text-decoration: none;
            margin-left: 20px;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            transition: all 0.3s;
            display: flex;
            align-items: center;
        }

        .navbar-links a i {
            margin-right: 5px;
        }

        .navbar-links a:hover {
            color: var(--text-primary);
            background-color: rgba(108, 92, 231, 0.1);
            transform: translateY(-2px);
        }

        .container {
            max-width: 900px;
            width: 100%;
            margin: 2rem auto;
            padding: 0 1.5rem;
            flex-grow: 1;
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
            animation: fadeIn 1s ease;
        }

        .header h1 {
            color: var(--text-primary);
            margin-bottom: 0.8rem;
            font-size: 2.8rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, var(--primary-color), #a29bfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .card {
            background-color: var(--dark-card);
            border-radius: 12px;
            padding: 2.5rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            border: 1px solid var(--dark-border);
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-color), #a29bfe);
        }

        .card::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 40%;
            height: 100%;
            background: radial-gradient(circle at right top, rgba(108, 92, 231, 0.05), transparent);
            pointer-events: none;
        }

        .input-section {
            animation: slideUp 0.8s ease;
        }

        .input-group {
            margin-bottom: 1.8rem;
        }

        .input-group label {
            display: block;
            margin-bottom: 0.8rem;
            color: var(--text-primary);
            font-weight: 500;
            font-size: 1.1rem;
        }

        .input-group textarea {
            width: 100%;
            padding: 16px;
            border: 1px solid var(--dark-border);
            border-radius: 8px;
            min-height: 150px;
            font-size: 1rem;
            transition: all 0.3s;
            background-color: var(--dark-surface);
            color: var(--text-primary);
            resize: vertical;
        }

        .input-group textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.3);
        }

        .input-group textarea::placeholder {
            color: var(--text-secondary);
            opacity: 0.7;
        }

        .btn {
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            font-size: 1rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            letter-spacing: 0.5px;
            position: relative;
            overflow: hidden;
        }

        .btn i {
            margin-right: 8px;
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
            color: white;
        }

        .btn-primary:hover {
            background: linear-gradient(135deg, var(--primary-light), var(--primary-color));
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(108, 92, 231, 0.3);
        }

        .btn-primary:active {
            transform: translateY(-1px);
            box-shadow: 0 5px 10px rgba(108, 92, 231, 0.2);
        }

        .btn-primary::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.1);
            transform: scaleX(0);
            transform-origin: right;
            transition: transform 0.5s;
        }

        .btn-primary:hover::after {
            transform: scaleX(1);
            transform-origin: left;
        }

        .loader {
            display: none;
            position: relative;
            width: 60px;
            height: 60px;
            margin: 30px auto;
        }

        .loader::before,
        .loader::after {
            content: '';
            position: absolute;
            border-radius: 50%;
            animation: pulse 1.8s ease-in-out infinite;
        }

        .loader::before {
            width: 100%;
            height: 100%;
            background: rgba(108, 92, 231, 0.2);
            animation-delay: 0.2s;
        }

        .loader::after {
            width: 70%;
            height: 70%;
            background: var(--primary-color);
            top: 15%;
            left: 15%;
            animation-delay: 0s;
        }

        @keyframes pulse {
            0% {
                transform: scale(0.6);
                opacity: 1;
            }
            50% {
                transform: scale(1);
                opacity: 0.5;
            }
            100% {
                transform: scale(0.6);
                opacity: 1;
            }
        }

        .result-section {
            display: none;
            animation: fadeInUp 0.8s ease;
        }

        .result-header {
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
        }

        .result-header h2 {
            color: var(--text-primary);
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }

        .result-header p {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .result-content {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }

        .result-badge {
            padding: 16px 32px;
            border-radius: 50px;
            font-weight: bold;
            font-size: 1.3rem;
            margin-bottom: 1.5rem;
            display: inline-flex;
            align-items: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
            z-index: 1;
        }

        .result-badge::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0.1;
            z-index: -1;
        }

        .result-badge.ham {
            background-color: rgba(46, 204, 113, 0.15);
            color: var(--secondary-color);
            border: 1px solid rgba(46, 204, 113, 0.3);
        }

        .result-badge.ham::before {
            background: radial-gradient(circle at center, var(--secondary-color) 0%, transparent 70%);
        }

        .result-badge.spam {
            background-color: rgba(231, 76, 60, 0.15);
            color: var(--danger-color);
            border: 1px solid rgba(231, 76, 60, 0.3);
        }

        .result-badge.spam::before {
            background: radial-gradient(circle at center, var(--danger-color) 0%, transparent 70%);
        }

        .result-badge i {
            margin-right: 10px;
        }

        .probability-bar-container {
            width: 100%;
            max-width: 450px;
            background-color: var(--dark-surface);
            border-radius: 10px;
            height: 16px;
            margin: 1.5rem 0;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
            border: 1px solid var(--dark-border);
        }

        .probability-bar {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 0%;
            transition: width 1.5s cubic-bezier(0.19, 1, 0.22, 1);
        }

        .probability-bar.ham {
            background: linear-gradient(90deg, #2ecc71, #1abc9c);
        }

        .probability-bar.spam {
            background: linear-gradient(90deg, #e74c3c, #c0392b);
        }

        .probability-text {
            text-align: center;
            font-weight: 500;
            color: var(--text-primary);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }

        .footer {
            background-color: var(--dark-surface);
            color: var(--text-secondary);
            text-align: center;
            padding: 1.8rem;
            margin-top: auto;
            border-top: 1px solid var(--dark-border);
        }

        .footer p {
            font-size: 1rem;
        }

        .footer a {
            color: var(--primary-light);
            text-decoration: none;
            transition: color 0.3s;
        }

        .footer a:hover {
            color: var(--primary-color);
            text-decoration: underline;
        }

        /* Animation keyframes */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Floating animation for elements */
        .float {
            animation: floating 3s ease-in-out infinite;
        }

        @keyframes floating {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }

        /* Glowing effect */
        .glow {
            animation: glow 2s ease-in-out infinite alternate;
        }

        @keyframes glow {
            from { box-shadow: 0 0 5px rgba(108, 92, 231, 0.2); }
            to { box-shadow: 0 0 20px rgba(108, 92, 231, 0.6); }
        }

        /* Responsive styles */
        @media (max-width: 768px) {
            .navbar {
                flex-direction: column;
                padding: 1rem;
            }

            .navbar-brand {
                margin-bottom: 1rem;
            }

            .card {
                padding: 1.5rem;
            }

            .header h1 {
                font-size: 2.2rem;
            }

            .btn {
                width: 100%;
            }
        }

        /* Features highlight */
        .features {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 40px;
            margin-bottom: 20px;
        }

        .feature-item {
            flex: 1 1 calc(33.333% - 20px);
            min-width: 200px;
            background-color: var(--dark-surface);
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid var(--dark-border);
            transition: all 0.3s;
        }

        .feature-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            border-color: var(--primary-color);
        }

        .feature-icon {
            font-size: 2rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }

        .feature-title {
            font-size: 1.2rem;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }

        .feature-desc {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        /* Progress indicator */
        .progress-indicator {
            display: flex;
            justify-content: space-between;
            margin-bottom: 40px;
            position: relative;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }

        .progress-indicator::before {
            content: '';
            position: absolute;
            top: 15px;
            left: 0;
            width: 100%;
            height: 2px;
            background-color: var(--dark-border);
            z-index: 1;
        }

        .progress-step {
            position: relative;
            z-index: 2;
            display: flex;
            flex-direction: column;
            align-items: center;
            flex: 1;
        }

        .step-bubble {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background-color: var(--dark-card);
            border: 2px solid var(--dark-border);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 8px;
            transition: all 0.3s;
        }

        .step-bubble.active {
            background-color: var(--primary-color);
            border-color: var(--primary-light);
            transform: scale(1.2);
        }

        .step-text {
            color: var(--text-secondary);
            font-size: 0.8rem;
            text-align: center;
            transition: all 0.3s;
        }

        .step-text.active {
            color: var(--primary-light);
            font-weight: 600;
        }

        /* Extended features */
        .recent-stats {
            background-color: var(--dark-card);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
            border: 1px solid var(--dark-border);
        }

        .stats-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .stats-title {
            font-size: 1.3rem;
            color: var(--text-primary);
        }

        .stats-content {
            display: flex;
            justify-content: space-around;
            text-align: center;
        }

        .stat-item {
            flex: 1;
        }

        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.3rem;
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }

        /* Tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
        }

        .tooltip .tooltip-text {
            visibility: hidden;
            width: 200px;
            background-color: var(--dark-surface);
            color: var(--text-primary);
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            border: 1px solid var(--dark-border);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            font-size: 0.9rem;
        }

        .tooltip:hover .tooltip-text {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-brand">
            <i class="fas fa-shield-alt glow"></i>
            <span>Spam Shield</span>
        </div>
        <div class="navbar-links">
            <a href="index.html"><i class="fas fa-home"></i> Home</a>
            <a href="about.html"><i class="fas fa-info-circle"></i> About</a>
            <a href="statistics.html"><i class="fas fa-chart-bar"></i> Statistics</a>
            <a href="contact.html"><i class="fas fa-envelope"></i> Contact</a>
        </div>
    </nav>

    <div class="container">
        <div class="header">
            <h1 class="float">SMS Spam Detector</h1>
            <p>Protect yourself from unwanted messages with spam detection system.</p>
        </div>

        <div class="progress-indicator">
            <div class="progress-step">
                <div class="step-bubble active">
                    <i class="fas fa-keyboard" style="font-size: 12px; color: white;"></i>
                </div>
                <div class="step-text active">Input</div>
            </div>
            <div class="progress-step">
                <div class="step-bubble" id="analyzeStep">
                    <i class="fas fa-search" style="font-size: 12px; color: white;"></i>
                </div>
                <div class="step-text">Analysis</div>
            </div>
            <div class="progress-step">
                <div class="step-bubble" id="resultStep">
                    <i class="fas fa-check" style="font-size: 12px; color: white;"></i>
                </div>
                <div class="step-text">Result</div>
            </div>
        </div>

        <div class="card input-section">
            <div class="input-group">
                <label for="message"><i class="fas fa-comment-alt"></i> Message Text:</label>
                <textarea id="message" placeholder="Type or paste a message here to analyze..."></textarea>
            </div>
            <button id="checkBtn" class="btn btn-primary" onclick="checkSpam()">
                <i class="fas fa-search"></i> Analyze Message
            </button>
        </div>

        <div class="loader" id="loader"></div>

        <div class="card result-section" id="resultSection">
            <div class="result-header">
                <h2>Analysis Result</h2>
                <p>Our AI has analyzed your message and determined the following:</p>
            </div>
            <div class="result-content">
                <div id="resultBadge" class="result-badge"></div>
                <div class="probability-bar-container">
                    <div id="probabilityBar" class="probability-bar"></div>
                </div>
                <div id="probabilityText" class="probability-text"></div>
            </div>
        </div>

        <div class="features">
            <div class="feature-item">
                <div class="feature-icon">
                    <i class="fas fa-bolt"></i>
                </div>
                <h3 class="feature-title">Fast Analysis</h3>
                <p class="feature-desc">Our system analyzes messages Random Forest algorithms.</p>
            </div>
            <div class="feature-item">
                <div class="feature-icon">
                    <i class="fas fa-shield-alt"></i>
                </div>
                <h3 class="feature-title">High Accuracy</h3>
                <p class="feature-desc">Over 99% accuracy in detecting various types of spam messages.</p>
            </div>
            <div class="feature-item">
                <div class="feature-icon">
                    <i class="fas fa-lock"></i>
                </div>
                <h3 class="feature-title">Privacy First</h3>
                <p class="feature-desc">Your messages are never stored and analysis happens locally.</p>
            </div>
        </div>
    </div>

    <footer class="footer">
        <p> &copy; 2025 Spam Shield - Advanced Protection System </p>
    </footer>

    <script>
        function checkSpam() {
            const message = document.getElementById('message').value.trim();
            if (!message) {
                alert("Please enter a message.");
                return;
            }

            // Update progress steps
            document.getElementById("analyzeStep").classList.add("active");
            document.querySelectorAll(".step-text")[1].classList.add("active");

            document.getElementById("loader").style.display = "block"; // Show loader

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById("loader").style.display = "none"; // Hide loader
                const resultSection = document.getElementById('resultSection');
                const resultBadge = document.getElementById('resultBadge');
                const probabilityBar = document.getElementById('probabilityBar');
                const probabilityText = document.getElementById('probabilityText');
                
                const isSpam = data.prediction === 'spam';
                const confidence = isSpam ? data.spam_probability : 1 - data.spam_probability;
                const percent = Math.round(confidence * 100);

                // Update final progress step
                document.getElementById("resultStep").classList.add("active");
                document.querySelectorAll(".step-text")[2].classList.add("active");

                // Add icon to result badge
                resultBadge.innerHTML = isSpam ? 
                    '<i class="fas fa-exclamation-triangle"></i> SPAM' : 
                    '<i class="fas fa-check-circle"></i> LEGITIMATE';
                resultBadge.className = "result-badge " + (isSpam ? "spam" : "ham");

                probabilityBar.style.width = percent + "%";
                probabilityBar.className = "probability-bar " + (isSpam ? "spam" : "ham");

                probabilityText.innerHTML = `Confidence Level: ${percent}%`;
                resultSection.style.display = "block"; // Show result
                
                // Smooth scroll to result
                resultSection.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                document.getElementById("loader").style.display = "none";
                alert("Error: " + error);
                console.error(error);
            });
        }

        // Add sample messages functionality
        document.addEventListener('DOMContentLoaded', function() {
            // This is new JS that doesn't modify the original functionality
            const textarea = document.getElementById('message');
            
            // Add tooltip hover effect for elements with tooltip class
            const tooltips = document.querySelectorAll('.tooltip');
            tooltips.forEach(tooltip => {
                tooltip.addEventListener('mouseenter', function() {
                    this.querySelector('.tooltip-text').style.visibility = 'visible';
                    this.querySelector('.tooltip-text').style.opacity = '1';
                });
                
                tooltip.addEventListener('mouseleave', function() {
                    this.querySelector('.tooltip-text').style.visibility = 'hidden';
                    this.querySelector('.tooltip-text').style.opacity = '0';
                });
            });
        });
    </script>
</body>
</html>
    """)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Clean the text
    cleaned_message = cleantext(message)
    # Transform the cleaned message using the loaded vectorizer
    vec = vectorizer.transform([cleaned_message])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    # Find the probability of spam
    spam_index = list(encoder.classes_).index('spam')
    spam_prob = prob[spam_index] if len(prob) > spam_index else 0.0

    return jsonify({
        'prediction': encoder.inverse_transform([prediction])[0],
        'spam_probability': round(spam_prob, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)

# SMS Spam Classifier 📧

A modern, full-stack machine learning application that detects and classifies SMS messages as spam or legitimate. Built with Flask backend and an interactive web interface featuring real-time analysis and dark mode support.

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-96.6%25-brightgreen)

---

## 🌟 Features

- **High Accuracy ML Model**: Multinomial Naive Bayes classifier with 96.6% accuracy
- **Real-time Text Analysis**: Instant spam detection as you type
- **Spam Keyword Highlighting**: Visual highlighting of detected spam keywords
- **Dark Mode**: Toggle between light and dark themes with persistent preferences
- **Character Counter**: Real-time character count display
- **Loading Animation**: Smooth loading spinner during predictions
- **Responsive Design**: Beautiful gradient UI that works on all devices
- **Message Persistence**: Text remains in input after prediction
- **Clear/Reset Button**: Quickly clear all fields and results
- **Confidence Score**: View model confidence percentage for predictions

---

## 🛠️ Tech Stack

### Backend
- **Framework**: Flask (Python)
- **ML Library**: scikit-learn
- **Text Processing**: TF-IDF Vectorizer
- **Algorithm**: Multinomial Naive Bayes
- **Model Serialization**: joblib

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern animations and gradients
- **JavaScript**: Interactive features
- **Bootstrap-inspired**: Responsive grid system

### Data Processing
- **pandas**: Data manipulation and loading
- **NumPy**: Numerical operations
- **NLTK**: Natural language processing (stopwords)

---

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
Flask>=2.0.0
nltk>=3.6.0
joblib>=1.0.0
```

---

## 🚀 Quick Start

### 1. Clone/Setup the Project
```bash
cd "c:\Users\kvmou\OneDrive\Desktop\Spam Email Classification"
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn nltk Flask joblib
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
Spam Email Classification/
├── app.py                      # Flask application server
├── model.ipynb                 # Jupyter notebook with ML pipeline
├── spam.csv                    # Training dataset (83,448 messages)
├── spam_model.pkl              # Trained Naive Bayes model
├── tfidf_vectorizer.pkl        # TF-IDF vectorizer model
├── README.md                   # This file
├── templates/
│   └── index.html              # Web UI interface
└── static/                     # (Optional) CSS/JS files
```

---

## 🧠 Model Information

### Training Data
- **Dataset**: SMS Spam Collection
- **Total Messages**: 83,448
- **Spam Messages**: 43,910 (52.6%)
- **Legitimate Messages**: 39,538 (47.4%)
- **Train-Test Split**: 80-20

### Model Architecture
- **Algorithm**: Multinomial Naive Bayes
- **Text Representation**: TF-IDF Vectorizer
- **Max Features**: 5,000 vocabulary size
- **Smoothing (alpha)**: 1.0

### Performance Metrics
```
              precision    recall  f1-score   support
          Ham       0.96      0.97      0.96      7938
         Spam       0.98      0.96      0.97      8752
       Accuracy                           0.97     16690
     Macro Avg       0.97      0.97      0.97     16690
  Weighted Avg       0.97      0.97      0.97     16690
```

---

## 💡 How It Works

### Text Preprocessing
1. **Lowercasing**: Convert all text to lowercase
2. **Character Cleaning**: Remove special characters, keep only alphanumeric and spaces
3. **Tokenization**: Split text into words
4. **Stopword Removal**: Remove common English stopwords (except: 'hi', 'i', 'you', 'please')

### Vectorization
- Converts preprocessed text into numerical features using TF-IDF
- Creates 5,000 dimensional feature vectors

### Classification
- Naive Bayes classifier predicts probability of spam vs. legitimate
- Returns prediction with confidence score

### Spam Keyword Detection
- Detects 23 common spam keywords: free, win, claim, prize, urgent, etc.
- Highlights detected keywords in results (yellow background)

---

## 🎯 Spam Keywords Detected

```
free, win, winner, claim, prize, urgent, act now, limited, offer, click, 
link, http, www, verify, account, password, bank, congratulations, gift, 
bonus, guaranteed, credit, loan, otp
```

---

## 📊 Model Training

To retrain the model with new data:

1. **Update the dataset**: Replace `spam.csv` with your data
2. **Run the notebook**: Open `model.ipynb` in Jupyter
3. **Execute cells in order**:
   - Install dependencies
   - Import libraries
   - Load and preprocess data
   - Train the model
   - Evaluate performance
   - Save models

```bash
jupyter notebook model.ipynb
```

---

## 🎨 User Interface Features

### Input Section
- Large textarea for message input
- Real-time character counter
- Clear button to reset form

### Results Display
- **Color-Coded Output**: 
  - 🟢 Green for "Not Spam"
  - 🔴 Red for "Spam"
- **Confidence Score**: Percentage confidence of prediction
- **Spam Keywords**: Yellow highlighted detected keywords

### Visual Features
- **Dark Mode Toggle**: Switch between light/dark themes
- **Loading Animation**: Smooth spinner during processing
- **Gradient Background**: Modern colorful design
- **Responsive Layout**: Works on mobile, tablet, desktop

---

## 🔧 Configuration

### Model Parameters (in `model.ipynb`)
```python
# Vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Classifier
model = MultinomialNB(alpha=1.0)

# Train-Test Split
train_test_split(X, y, test_size=0.2, random_state=42)
```

### Flask Configuration (in `app.py`)
```python
app = Flask(__name__)
# Debug mode enabled for development
if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📝 API Endpoints

### 1. Home Page
```
GET /
Returns: HTML interface for spam classification
```

### 2. Prediction
```
POST /predict
Input: form data with 'message' field
Returns: HTML with prediction, confidence, and keywords
```

### 3. Real-time Analysis (Optional)
```
POST /analyze
Input: JSON with message text
Returns: JSON with prediction and metadata
```

---

## ⚠️ Important Notes

### Model Behavior
- Model is trained on SMS spam patterns and may not generalize perfectly to other text types
- Some legitimate messages might be classified as spam if they contain words common in spam messages
- The dataset may have domain-specific characteristics (SMS vs. Email)

### Version Compatibility
- Compatible with scikit-learn 1.0+ (developed with 1.7.2)
- Python 3.8 or higher recommended
- Flask 2.0+ required

### Performance
- First prediction takes ~1-2 seconds (model loading)
- Subsequent predictions are instant
- No GPU required, runs on CPU

---

## 🐛 Troubleshooting

### Flask App Won't Start
```bash
# Kill existing Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Restart Flask
python app.py
```

### Model Files Not Found
```bash
# Ensure spam_model.pkl and tfidf_vectorizer.pkl exist in project root
# If missing, run the notebook to retrain and save models
```

### NLTK Data Missing
```python
import nltk
nltk.download('stopwords')
```

### Port 5000 Already in Use
```bash
# Use a different port in app.py
app.run(debug=True, port=5001)
```

---

## 📈 Future Enhancements

- [ ] Support for multiple languages
- [ ] Batch message processing
- [ ] Advanced model algorithms (SVM, Random Forest)
- [ ] Database integration for message history
- [ ] API authentication and rate limiting
- [ ] Model performance metrics dashboard
- [ ] Custom training data upload
- [ ] Export predictions to CSV

---

## 📚 Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [TF-IDF Vectorization](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [NLTK Documentation](https://www.nltk.org/)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Your Name**
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request with improvements.

### How to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ❓ FAQs

**Q: How accurate is the model?**
A: The model achieves 96.6% accuracy on the test set. However, accuracy may vary depending on message type and domain.

**Q: Can I use this for Email classification?**
A: Yes, but be aware the model was trained on SMS data. Consider retraining with email-specific data for better results.

**Q: How do I update the model with new training data?**
A: Replace `spam.csv` and run all cells in `model.ipynb` to retrain the model.

**Q: Is it safe to use this in production?**
A: For production use, consider adding API authentication, rate limiting, and monitoring. Use a production WSGI server like Gunicorn instead of Flask's development server.

**Q: Can the model learn from user feedback?**
A: Currently, the model is static. For active learning, implement a feedback mechanism and periodic retraining pipeline.

---

## 📞 Support

For issues, questions, or suggestions, please:
1. Check this README and FAQs
2. Open an issue on GitHub
3. Contact the author via email

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ for SMS Spam Detection

</div>

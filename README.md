# CV MANAGIZER

**CV Managizer** is a smart, modular resume analysis system built to streamline the evaluation and classification of CVs. Designed for recruiters, HR teams, and developers, it provides intelligent scoring, filtering, and keyword-based evaluation to identify the most relevant candidates efficiently.

## Key Features

- Resume Classification  
  Automatically distinguishes between real CVs and irrelevant documents using a custom-trained model.

- Custom Keyword Evaluation  
  Supports user-defined keywords and skill sets to tailor scoring to specific job requirements.

- Scoring Engine  
  Assigns scores to each resume based on layout, skill match, education, and experience extracted from the content.

- Training Pipeline  
  Includes scripts and datasets to train or retrain the classification model as needed.

- Clean Web Interface  
  Built with Flask and Jinja templating, the frontend allows for easy interaction with the system via upload forms and result dashboards.

## Project Structure

CV MANAGIZER/

├── app.py                 # Main Flask app  
├── resume_analyzer.py     # Resume processing logic  
├── evaluator.py           # Custom scoring logic  
├── train/                 # Model training scripts and datasets  
├── models/                # Saved machine learning models  
├── Templates/             # HTML frontend pages  
├── Static/                # CSS, JS, and images  
├── config.py              # App configuration  
├── requirements.txt       # Python dependencies  
└── cvmanagizer.db         # SQLite database file

## Technologies Used

- Python 3.9+  
- Flask (Web framework)  
- SQLite (Lightweight database)  
- Scikit-learn / Pandas / NumPy (Machine Learning and Data Handling)  
- PDFMiner / PyMuPDF (PDF parsing and text extraction)  
- Jinja2 (HTML templating)


## Sample Code

def score_resume(text, required_keywords):  
    score = 0  
    total_keywords = len(required_keywords)  
    
    for keyword in required_keywords:  
        if keyword.lower() in text.lower():  
            score += 1  
    
    if total_keywords == 0:  
        return 0  
    
    return round((score / total_keywords) * 100, 2)

This function scores a resume based on the number of matched keywords and returns a percentage.


## Maintainer

**Atwa-a**  
For contributions, suggestions, or issues, feel free to submit a pull request or open an issue on the repository.

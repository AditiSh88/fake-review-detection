# Detection of Fake Reviews Using Hybrid Machine Learning

## Overview
Online reviews play a crucial role in influencing user decisions, but the rise of fake and manipulated reviews reduces trust and reliability. This project addresses the problem by building a hybrid machine learning system that classifies reviews as Likely Genuine or Likely Fake, while also explaining why a prediction was made.

## Tech Stack
- **Language**: Python  
- **Libraries**: Scikit-learn, XGBoost, Pandas, NumPy, NLTK, Matplotlib  
- **Techniques**: TF-IDF Vectorization  
- **Models**: Logistic Regression, XGBoost (Hybrid Model)  
- **Frontend**: Streamlit  

## Key Features

### Hybrid Machine Learning Model
- Combines Logistic Regression (interpretable) and XGBoost (high performance)  
- Produces more reliable and balanced predictions  
- Reduces limitations of individual models  

### Explainability
- Highlights important words directly within the review  
- Shows how specific words influence the prediction  
- Improves transparency and user trust
  
### Model Comparison & Analytics Dashboard
- Displays performance metrics:
  - Accuracy, Precision, Recall, F1 Score  
- Includes:
  - Confusion Matrix  
  - ROC Curve  
- Allows comparison between Logistic Regression and XGBoost outputs  

## Prediction Output
- **Final Verdict**: Likely Genuine / Likely Fake  
- **Prediction Score**: Probability of being fake  
- **Confidence Score**: Certainty of prediction  

## Project Structure
```
app/
│
├── Home.py
├── pages/
│ ├── 1_Analysis.py
│ ├── 2_Insights.py
│
├── utils/
│ ├── model_loader.py
│ ├── explain.py
│ ├── report.py
│
src/
├── train.py
├── preprocessing.py
│
models/
├── tfidf.pkl
├── logreg.pkl
├── xgb.pkl
│
data/
├── reviews.csv
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AditiSh88/fake-review-detection.git
   cd fake-review-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python src/train.py
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app/Home.py
   ```

## Future Scope
- Integration with deep learning models
- Real-time API deployment  
- Multilingual review detection  
- Larger dataset training for improved accuracy  

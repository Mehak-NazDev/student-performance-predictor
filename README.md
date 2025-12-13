# 🎓 Student Performance Predictor - EdTech ML Project

Predict student academic performance with a **98.8% R² score** using study habits, sleep patterns, and previous scores.
This project helps students understand factors affecting academic performance and provides insights for **study planning and optimization**.


## 📌 Quick Results

* **R² score:** 0.988 with Linear Regression
* **MAE (Mean Absolute Error):** 1.64
* **Dataset:** 9,873 students with 5 key features
* **Top Predictor:** Previous Scores (~0.9 correlation with final score)
* **Study Hours:** +7.37 points associated per additional hour


## 🚀 Get Started (Reproducing the Environment)

This guide uses Anaconda/Conda to guarantee a successful and conflict-free setup.

```bash
# 1. Clone the repository
git clone https://github.com/Mehak-NazDev/student-performance-predictor
cd student-performance-predictor

# 2. Setup Clean Environment & Install Dependencies
# Creates an isolated environment to prevent conflicts.
conda create -n student-predictor-env python=3.9 -y
conda activate student-predictor-env
pip install -r requirements.txt

# 3. Run the Dashboard
streamlit run dashboard_app/app.py
 ```
## 📁 Project Structure
```
student-performance-predictor/
├── data/                                              # Dataset
│   └── Student_Performance_cleaned.csv
├── notebooks/                                         # Complete ML analysis
│   └── Student-performance-predictor-ml-model.ipynb
├── models/                                            # Trained models
│   ├── linear_model.pkl                               # 🏆 Best model (R² = 0.988)
│   └── scaler.pkl
├── dashboard_app/                                     # Interactive web app
│   └── app.py
├── images/                                            # Screenshots or GIFs
├── README.md
└── requirements.txt                            

```

## 🔍 Model Validation & Feature Checks

 * Train-test split: 80/20
 * Scaling: StandardScaler applied on training data
 * Skewness check: Features normalized where necessary
 * Multicollinearity check: VIF analysis done 
 * Correlation insight: Previous Scores have ~0.9 correlation with final score → strongest predictor
 * Linearity check: Scatter plots confirm linear relationships between features and target


## 🧠 Why Linear Regression?
| Model | MAE | R² | Why Not Chosen |
|---|---|---|---|
| Linear Regression | 1.64 | 0.988 | ✅ Selected: Best R2 (highest) and lowest MAE |
| Random Forest | 1.89 | 0.985 | Good performance but higher MAE |
| Gradient Boosting | 1.69 | 0.987 | Similar R² score, higher MAE than Linear Regression |
> Sometimes the simplest model is the smartest choice.
>

## 📊 Key Insights
 * Past Performance = Future Success (strongest predictor)
 * Study Consistency > Sleep > Practice Papers (relative impact)
 * 6–8 Study Hours are associated with higher predicted performance
 * Extracurricular Balance shows a small but meaningful association

## 📄 Dataset Overview
 * Source: Public Kaggle dataset (~10,000 students)
 * Features:Hours Studied, Previous Scores, Sleep Hours, Sample Question Papers Practiced, Extracurricular Activities
 * Target: Performance index (Final score)

## 🛠 Tech stacks

 * Python, Scikit-learn, Pandas – ML pipeline
 * Streamlit – Interactive dashboard
 * Plotly, Matplotlib – Visualizations
 * Joblib – Model persistence
  
## ⚠️ Educational Use Only

 * Designed for study optimization & planning
 * Not intended for grading or official evaluation
 * Focus on growth and improvement insights
   
## 📈 My Learning Journey

 * Week 3: Healthcare visualization project (foundation in data analysis)
 * Week 6: Completed full ML pipeline with interactive dashboard
 * Growth: From data visualization to production-ready ML system
> Building tools that make education more data-informed and personalized.
>

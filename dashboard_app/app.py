#importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path



# Setting up the  PAGE Configiuration 
st.set_page_config(
    page_title="EduPredict Pro | Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)


#  SIMPLE CSS Styling
st.markdown("""
<style>
    /* Modern Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #4c51bf 100%);
        background-attachment: fixed;
    }
    
    /* Simple card style */
    .simple-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Big metric numbers */
    .big-metric {
        font-size: 3em;
        font-weight: bold;
        color: #2d3748;
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2d3748;
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #4a5568;
    }
</style>
""", unsafe_allow_html=True)

# Loading the MODELS 
@st.cache_resource
def load_models():
    try:
        # Multiple possible paths to load models
        possible_paths = [
            Path("models"),  # Same directory as app
            Path("..") / "models",  # Parent directory
            Path(__file__).parent.parent / "models" if "__file__" in globals() else None,
            Path.cwd() / "models",  # Current working directory
        ]
        
        scaler_path = None
        model_path = None
        
        # Find the models
        for base_path in possible_paths:
            if base_path and (base_path / "scaler.pkl").exists():
                scaler_path = base_path / "scaler.pkl"
                model_path = base_path / "linear_model.pkl"
                break
        
        if scaler_path is None or model_path is None:
            st.error("❌ Models not found! Please ensure:")
            st.info("1. Run the notebook to train and save models")
            st.info("2. Models should be in a folder called 'models'")
            st.info("3. The 'models' folder should contain: scaler.pkl and linear_model.pkl")
            return None, None
        
        
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        
        return scaler, model
        
    except Exception as e:
        st.error(f"⚠️ Model loading error: {str(e)}")
        st.info("Try placing models in the same folder as this app")
        return None, None

scaler, lr_model = load_models()

if scaler is None or lr_model is None:
    st.stop()


# SIDEBAR - SIMPLE & CLEAN
with st.sidebar:
   
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='color: white;'>📊 Student Inputs</h2>
        <p style='color: #e2e8f0;'>Enter your study details below</p>
    </div>
    """, unsafe_allow_html=True)
    # Inputs in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.slider("Study Hours", 0, 24, 7)
        previous_score = st.slider("Previous Score", 0, 100, 75)
    
    with col2:
        sleep_hours = st.slider("Sleep Hours", 0, 24, 8)
        sample_papers = st.slider("Practice Papers", 0, 10, 5)
    
    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["No", "Yes"]
    )
    
    st.markdown("---")
    
    # Quick insights based on inputs
    with st.expander("💡 Quick Insights", expanded=True):
        # Study hours insight
        if hours_studied < 4:
            st.warning("Study more: Aim for 4+ hours daily")
        elif hours_studied > 10:
            st.info("Good study hours! Remember to take breaks")
        else:
            st.success("Optimal study range: 4-10 hours")
        
        # Sleep insight
        if sleep_hours < 6:
            st.error("Insufficient sleep: Aim for 7+ hours")
        elif 7 <= sleep_hours <= 8:
            st.success("Perfect sleep range for learning")
        elif 8 <= sleep_hours <= 11 :
            st.info("Good sleep, but ensure it doesn't affect study time")    
        else:
            st.warning("Too much sleep can affect productivity")    
        
        # Practice papers insight
        if sample_papers < 3:
            st.warning("More practice needed: Aim for 3+ papers weekly")

#  MAIN DASHBOARD 
# Header
st.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h1 style='color: #2d3748;'>🎓 Student Performance Predictor</h1>
    <p style='color: #2d3748; font-size: 1.2em;'>
        Predict your academic performance using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Current Metrics Display
st.markdown("### 📊 Your Current Profile")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📚 Study Hours", hours_studied, 
              f"{'✅ Optimal' if 6 <= hours_studied <= 8 else '⚡ Adjust'}")
    
with col2:
    st.metric("🏆 Previous Score", previous_score, 
              f"{'Excellent' if previous_score >= 85 else 'Good' if previous_score >= 70 else 'Average'  if previous_score >=50  else 'Below Average' }")

with col3:
    st.metric("😴 Sleep Hours", sleep_hours, 
              f"{'Optimal' if 7 <= sleep_hours <= 8 else 'Adjust'}")

with col4:
    st.metric("📝 Practice Papers", sample_papers, 
              f"{'On Track' if sample_papers >= 5 else 'More Needed'}")

with col5:
    st.metric("⚽ Activities", extracurricular, 
              f"{'Balanced' if extracurricular == 'Yes' else 'Study Focused'}")

# Prediction Button (making it Centered)
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "🚀 **PREDICT MY PERFORMANCE**",
        use_container_width=True
    )

#  PREDICTION LOGIC 
if predict_button:
    if scaler is None or lr_model is None:
        st.error("Please train and save the models first!")
    else:
        # Prepare data
        extracurricular_numeric = 1 if extracurricular == "Yes" else 0
        new_student = pd.DataFrame({
            "Hours Studied": [hours_studied],
            "Previous Scores": [previous_score],
            "Sleep Hours": [sleep_hours],
            "Sample Question Papers Practiced": [sample_papers],
            "Extracurricular Activities_Yes": [extracurricular_numeric],
        })
        expected_cols = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Extracurricular Activities_Yes"
]

        new_student = new_student[expected_cols]
        
        # Scale features
        numerical_features = ["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced"]
        new_student_scaled = new_student.copy()
        new_student_scaled[numerical_features] = scaler.transform(new_student[numerical_features])
        
        # Make prediction
        prediction = lr_model.predict(new_student_scaled)[0]
        prediction = np.clip(prediction, 0, 100)
        
        # Show success
        st.balloons()
        
        # RESULTS section
        st.markdown("## 📈 Prediction Results")
        
        # Create two columns for results
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Performance Index", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#c0c8d7"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 75], 'color': "lightyellow"},
                        {'range': [75, 100], 'color': "lightgreen"}
                    ],
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with result_col2:
            # Performance level
            if prediction >= 85:
                level = "EXCELLENT 🏆"
                color = "#38a169"
                message = "Outstanding! You're in the top tier!"
            elif prediction >= 70:
                level = "GOOD 🚀"
                color = "#3182ce"
                message = "Solid performance with room to grow!"
            elif prediction >= 50:
                level = "AVERAGE 📈"
                color = "#dd6b20"
                message = "Good foundation. Small improvements can help!"
            elif prediction==0:
                level =  "Below Average ❌"
                color = "#e53e3e"
                message = "You will have to work hard,don't lose hope!"    
            else:
                level = "NEEDS IMPROVEMENT 💪"
                color = "#e53e3e"
                message = "Focus on key areas for better results!"
            
            st.markdown(f"""
            <div class='simple-card' style='text-align: center; border-left: 5px solid {color};'>
                <div style='font-size: 2.5em; margin: 10px 0;'>🎯</div>
                <h2 style='color: {color};'>{prediction:.1f}/100</h2>
                <h3 style='color: #2d3748;'>{level}</h3>
                <p style='color: #4a5568;'>{message}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show insights
        # Insight 1: Comparison with average
        dataset_average = 55.22 # From the dataset analysis
        difference = prediction - dataset_average
        
        if difference > 0:
            insight1 = f"📊 You're predicted to score **{difference:.1f} points above** the average student ({dataset_average})"
        else:
            insight1 = f"📊 You're predicted to score **{abs(difference):.1f} points below** the average student ({dataset_average})"
        
        # Insight 2: Strongest predictor
        insight2 = "🎯 **Previous scores** have the biggest impact on predictions"
        
        # Insight 3: Your strongest area
        strengths = []
        if hours_studied >= 6:
            strengths.append("consistent study hours")
        if sleep_hours >= 7:
            strengths.append("good sleep habits")
        if sample_papers >= 4:
            strengths.append("regular practice")
        if extracurricular == "Yes":
            strengths.append("balanced lifestyle")
        
        if strengths:
            insight3 = f"✅ Your strengths: {', '.join(strengths)}"
        else:
            insight3 = "💡 Focus on building consistent study habits"
        
        # Display insights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(insight1)
        with col2:
            st.info(insight2)
        with col3:
            st.info(insight3)
        
        #  RECOMMENDATIONs
        st.markdown("## 🎯 Recommendations")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        with rec_col1:
            st.markdown("""
            ### 📚 Study Optimization
            - Use Pomodoro technique (25 min study, 5 min break)
            - Review notes within 24 hours
            - Teach concepts to reinforce learning
            - Study in the morning for better focus
            """)
        
        with rec_col2:
            st.markdown("""
            ### 😴 Sleep & Health
            - Aim for 7-8 hours of sleep
            - Maintain consistent sleep schedule
            - Exercise 20 min daily for brain health
            - Stay hydrated (drink enough water)
            """)
        
        with rec_col3:
            st.markdown("""
            ### 📝 Practice Strategy
            - Solve 5+ practice papers weekly
            - Review mistakes thoroughly
            - Practice under time pressure
            - Focus on weak topics first
            """)
        
        
        st.markdown("---")
        st.markdown("""
        ### 🤖 How It Works
        This prediction is made by a **Linear Regression model** trained on student data.    
        The model achieves **95%+ R2 score** on test data.
        """)

# FOOTER 
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #e2e8f0; padding: 20px;'>
    <p><strong style='color: white;'>🎓 EduPredict Pro</strong> | Machine Learning Project</p>
    <p>Developed by Mehak Naz | Built with Python, Scikit-learn, Streamlit</p>
</div>
""", unsafe_allow_html=True)
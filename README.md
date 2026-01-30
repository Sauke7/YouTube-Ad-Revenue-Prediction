# 📺 YouTube Ad Revenue Prediction 💰

Predicting YouTube ad revenue isn’t just about views — it’s about understanding engagement, content impact, and data-driven decisions.  
This project uses **Machine Learning (ElasticNet Regression)** to estimate ad revenue for individual YouTube videos based on performance metrics.

---

## 🚀 Project Highlights
- 📊 End-to-end ML pipeline (Data → Model → Deployment)
- 🧠 ElasticNet Regression as the final optimized model
- ⚙️ Cleaned & preprocessed real-world dataset
- 🌐 Interactive **Streamlit web app** for live predictions
- 💾 Model persistence using Joblib

---

## 🧩 Problem Statement
Content creators and media companies rely heavily on YouTube ad revenue.  
This project helps **predict expected ad revenue** for a video using key indicators like:
- Views
- Likes
- Comments
- Video duration

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Model:** ElasticNet Regression  
- **Deployment:** Streamlit  
- **Model Saving:** Joblib  

---

## 🔁 Project Workflow
1. Data loading & cleaning  
2. Exploratory Data Analysis (EDA)  
3. Feature selection & scaling  
4. Model training using ElasticNet  
5. Model evaluation (R², RMSE, MAE)  
6. Model deployment via Streamlit  

---

## 🤖 Model Used
### ✅ ElasticNet Regression
- Combines **L1 (Lasso)** & **L2 (Ridge)** regularization  
- Handles multicollinearity (views, likes, comments)  
- Prevents overfitting  
- Delivered the best performance on test data  

---

## 📂 Project Structure
├── app.py # Streamlit web application
├── model_training.py # Model training & evaluation
├── data_cleaning.py # Data preprocessing
├── cleaned_data.csv # Final cleaned dataset
├── model.pkl # Trained ElasticNet model
├── README.md # Project documentation

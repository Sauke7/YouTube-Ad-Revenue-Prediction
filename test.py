import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
 
# -------------------------------
# Load resources
# -------------------------------
df = pd.read_csv("cleaned_data.csv")
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature engineering
df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"]

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📺 YouTube Revenue App")
page = st.sidebar.radio(
    "Go to",
    ["📊 Visual Analytics & Insights", "💰 Revenue Prediction"]
)

# =====================================================
# PAGE 1: VISUAL ANALYTICS & MODEL INSIGHTS
# =====================================================
if page == "📊 Visual Analytics & Insights":

    st.title("📊 Basic Visual Analytics & Model Insights")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Revenue Distribution
    # -------------------------------
    st.subheader("Ad Revenue Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["ad_revenue_usd"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("🔍 **Insight:** Revenue is right-skewed with few high-earning videos.")

    # -------------------------------
    # Views vs Revenue
    # -------------------------------
    st.subheader("Views vs Ad Revenue")

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["views"],
        y=df["ad_revenue_usd"],
        alpha=0.5,
        ax=ax
    )
    ax.set_xlabel("Views")
    ax.set_ylabel("Ad Revenue (USD)")
    st.pyplot(fig)

    st.markdown("🔍 **Insight:** Higher views generally result in higher ad revenue.")

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    st.subheader("Feature Correlation Heatmap")

    corr = df[
        ["views", "likes", "comments", "video_length_minutes",
         "engagement_rate", "ad_revenue_usd"]
    ].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.markdown("""
    🔍 **Key Insights:**
    - Views have the strongest correlation with revenue  
    - Engagement rate improves prediction accuracy  
    - Duration has moderate impact  
    """)

    # -------------------------------
    # Model Insights
    # -------------------------------
    st.subheader("Best Model Insights")

    st.success("""
    ✅ Best Model Selected: **ElasticNet Regressor**

    **Why ElasticNet?**
    - Combines **L1 (Lasso)** and **L2 (Ridge)** regularization
    - Handles **multicollinearity** effectively
    - Prevents **overfitting** on noisy features
    - Achieved the **highest R² score**
    - Delivered the **lowest RMSE and MAE** among all models
    - Suitable for datasets with **many correlated predictors**
    """)
# =====================================================
# PAGE 2: REVENUE PREDICTION
# =====================================================
elif page == "💰 Revenue Prediction":

    st.title("💰 YouTube Ad Revenue Prediction")

    st.markdown("Enter video performance details to predict ad revenue 👇")

    # -------------------------------
    # User Inputs
    # -------------------------------
    views = st.number_input("Views", min_value=1, step=100)
    likes = st.number_input("Likes", min_value=0, step=10)
    comments = st.number_input("Comments", min_value=0, step=5)
    duration = st.slider("Video Duration (minutes)", 1, 60, 10)

    # Feature engineering
    engagement_rate = (likes + comments) / views

    input_data = np.array([[
        views,
        likes,
        comments,
        duration,
        engagement_rate
    ]])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # -------------------------------
    # Prediction
    # -------------------------------
    if st.button("Predict Ad Revenue"):
        prediction = model.predict(input_scaled)

        st.success(
            f"💵 Estimated Ad Revenue: **${prediction[0]:.2f}**"
        )

     

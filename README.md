# TunnelVision 🏙️  
### **AI for Predictive Infrastructure Maintenance in the Bay Area**

TunnelVision is an AI-powered predictive maintenance system designed to help cities across the **San Francisco Bay Area** detect infrastructure risks *before* failures occur.

Using machine learning and realistic synthetic data, TunnelVision forecasts short-term failures and maintenance needs for assets like pipelines, roads, and drainage systems—supporting **construction workers, maintenance crews, and city officials** in making faster, safer, and more cost-effective decisions.

---

## 🚀 What It Does
- Predicts **asset failure risk within the next 30 days**
- Classifies **likely failure types**
- Generates a **0–100 risk score**
- Recommends **maintenance actions and priorities**
- Helps teams decide **what to fix first and why**

---

## 🧠 How It Works

### **1️⃣ Realistic Synthetic Infrastructure Data**
TunnelVision uses high-quality synthetic data generated with **mostly.ai**, modeled after real Bay Area infrastructure patterns.

Each asset snapshot includes features such as:
- Asset type, material, installation year, and length  
- Geographic data (latitude, longitude, region)  
- Environmental conditions (rainfall, temperature, soil moisture)  
- Terrain and surrounding context (soil type, slope, tree density, traffic)  
- Maintenance history (previous failures, last repair date)

This approach preserves privacy while still capturing realistic degradation and failure behavior.

---

### **2️⃣ Machine Learning Models**
TunnelVision uses **Random Forest models** to learn complex, non-linear relationships between environmental stressors, asset history, and failure risk.

The models predict:
- **Failure likelihood in the next 30 days**
- **Most likely failure type**
- **Maintenance recommendations and priority level**

---

### **3️⃣ Risk Score (0–100)**
Each asset is assigned a risk score based on:
| Factor | Weight |
|------|------|
| Remaining Useful Life | 40% |
| Failure Probability | 30% |
| Asset Criticality | 20% |
| Model Uncertainty | 10% |

This score allows teams to quickly compare assets and focus on the highest-risk areas.

---

### **4️⃣ System Outputs**
For every asset snapshot, TunnelVision outputs:
- `failure_next_30d`
- `failure_type_predicted`
- `risk_score`
- `recommended_action`
- `recommended_priority`

These outputs are designed to be **directly actionable** for real-world maintenance planning.

---

## 🌟 Why It Matters
Most infrastructure systems rely on **reactive maintenance**—fixing problems only after failure.

TunnelVision enables a **proactive approach**, helping reduce:
- Emergency repair costs  
- Service disruptions  
- Safety hazards  
- Long-term infrastructure degradation  

---

## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn (Random Forest)**
- **mostly.ai (synthetic data generation)**

---

## 🎯 What’s Next
- **Interactive Streamlit Dashboard**  
  Build a lightweight web interface that allows users to:
  - View assets ranked by risk
  - Filter by region, asset type, or priority
  - Inspect individual asset predictions and recommended actions
  - Visualize infrastructure risk across the Bay Area  

- **Expand Asset Coverage**  
  Add additional infrastructure types such as bridges and streetlights.

- **Model Explainability (SHAP)**  
  Integrate SHAP to show which features (e.g., rainfall, asset age, traffic) most influenced each prediction—improving transparency and trust for city officials and engineers.

---

# 🎉 TunnelVision — Smarter Cities, Safer Infrastructure

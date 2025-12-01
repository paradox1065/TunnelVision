# TunnelVision 🏙️  
### **AI for Predictive Maintenance in Cupertino, CA**

TunnelVision predicts infrastructure failures *before* they happen.  
Using machine learning and realistic synthetic data, it forecasts when roads, pipelines, pumps, and storm drains will need maintenance — helping cities prevent outages, reduce costs, and improve public safety.

---

## 🚀 What It Does
- Predicts **Remaining Useful Life (RUL)** of each asset  
- Sends **early failure alerts** (next 30 days)  
- Generates a **0–100 risk score**  
- Recommends **maintenance actions & priorities**  
- Visualizes asset health across the city  

---

## 🧠 How It Works

### **1️⃣ Realistic Synthetic City Data**
We built a dataset modeled after Cupertino infrastructure using 20+ engineered features:
- Asset age, material, soil type  
- Rainfall, temperature, traffic  
- Previous failures & repair history  
- Seasonal patterns (ex: storms → storm drain failures)  

> This allows the model to learn real degradation behavior without needing months of city data collection.

---

### **2️⃣ Machine Learning Models**
TunnelVision predicts:
- **RUL (regression)**  
- **Failure probability (classification)**  

Models used:
- Random Forest  
- Gradient Boosting  
- XGBoost  

---

### **3️⃣ Risk Score (0–100)**
Weighted by:
| Factor | Weight |
|--------|--------|
| Remaining Useful Life | 40% |
| Failure Probability | 30% |
| Asset Criticality | 20% |
| Model Uncertainty | 10% |

---

### **4️⃣ Recommendations**
The system outputs:
- Maintenance priority  
- Estimated cost  
- Predicted failure type  
- Urgency level  

---

## 🌟 Why It Matters
Cities usually fix infrastructure *after* it breaks.  
TunnelVision flips the model: fix things **before** they fail.

This reduces:
- Emergency repair costs  
- Service outages  
- Public safety risks  
- Long-term degradation  

---

## 🛠️ Tech Stack
- **Python**  
- **Pandas, NumPy**  
- **Scikit-learn / XGBoost**  
- **Synthetic Data Engine**  
- *(Optional)* Streamlit dashboard  

---

## 🧪 Current Status
Actively building:
- Synthetic dataset  
- RUL prediction model  
- Failure prediction model  
- Risk scoring engine  
- Asset health dashboard  

---

## 🎯 What’s Next
- Add more asset types (bridges, streetlights)  
- Deploy full dashboard UI  
- Integrate SHAP explainability  

---

# 🎉 TunnelVision — Smarter Cities, Safer Infrastructure

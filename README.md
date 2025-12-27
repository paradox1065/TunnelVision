# TunnelVision 🏙️  
### **AI for Predictive Infrastructure Maintenance in the Bay Area**

> **🔗 Live Demo:** [https://tunnelvision-jef5.onrender.com](https://tunnelvision-jef5.onrender.com)  

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

## 🎮 Try It Out

### **Option 1: Live Demo (Recommended)**
Visit the live site: **[https://tunnelvision-jef5.onrender.com](https://tunnelvision-jef5.onrender.com)**

1. Navigate to the **Form** page
2. Fill in asset details (type, material, location, etc.)
3. Click **"Analyze"** to get instant predictions
4. View risk score, failure type, and recommended actions

> **Note:** The free tier may take up to 5 minutes to wake up on first visit.

### **Option 2: Run Locally**

#### Prerequisites
- Python 3.9+
- pip

#### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/tunnelvision.git
cd tunnelvision

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn back_end.api:app --host 0.0.0.0 --port 8000

Open a new terminal

# Run the server
python -m http.server 5500
```
Make sure that both ports (8000 and 5500) are public. 
Wait until the API terminal says `Application startup complete` before entering port 5500.
Open port 5500 and click on the `front_end` directory. This will lead you to the website.

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
TunnelVision uses an **ensemble of ML models** (Random Forest, XGBoost, Gradient Boosting) to learn complex, non-linear relationships between environmental stressors, asset history, and failure risk.

**Five specialized models predict:**
- **Failure likelihood in the next 30 days**
- **Most likely failure type**
- **Remaining useful life (RUL)**
- **Maintenance recommendations**
- **Priority level**

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
- `failure_in_30_days` - Boolean flag for urgent attention
- `failure_type` - Predicted failure mode (e.g., "Corrosion", "Structural")
- `risk_score` - 0-100 composite risk assessment
- `recommended_action` - Specific maintenance recommendation
- `priority` - 1-5 urgency level

These outputs are designed to be **directly actionable** for real-world maintenance planning.

---

## 🏗️ Project Structure
```
tunnelvision/
├── back_end/
│   ├── api.py              # FastAPI server + endpoints
│   ├── model_utils.py      # Model loading & prediction
│   ├── preprocessing.py    # Feature engineering
│   └── models/             # Trained ML models (.pkl files)
├── front_end/
│   ├── index.html          # Landing page
│   ├── Form.html           # Prediction form
│   ├── About.html          # Project information
│   ├── style.css           # Styling
│   └── script.js           # Form logic & API calls
├── requirements.txt        # Python dependencies
└── render.yaml             # Deployment configuration
```

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

### **Backend**
- **Python 3**
- **FastAPI** - Modern Python web framework
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Random Forest models
- **XGBoost & Gradient Boosting** - Ensemble methods

### **Frontend**
- **HTML**
- **Vanilla JavaScript** - No framework overhead
- **Modern CSS** - Glassmorphism design
- **Responsive Design** - Mobile-friendly

### **Data**
- **mostly.ai** - Synthetic data generation

### **Deployment**
- **Render** - Cloud hosting (backend + frontend)

---

## 🎯 What's Next

### **Short Term**
- **Interactive Dashboard**  
  Build a lightweight web interface that allows users to:
  - View assets ranked by risk
  - Filter by region, asset type, or priority
  - Inspect individual asset predictions and recommended actions
  - Visualize infrastructure risk across the Bay Area  

- **Model Explainability (SHAP)**  
  Integrate SHAP to show which features (e.g., rainfall, asset age, traffic) most influenced each prediction—improving transparency and trust for city officials and engineers.

### **Long Term**
- **Expand Asset Coverage**  
  Add additional infrastructure types such as bridges, streetlights, and electrical grids.

- **Historical Trend Analysis**  
  Track risk score changes over time to identify degradation patterns.

- **Mobile App**  
  Field-ready mobile interface for maintenance crews.

---

## 👥 Team
- **Aarushi Upadhyayula**: I worked on the back-end of this project, specifically on the predictive models. I had never done any coding besides competitive, much less AIML. There were several roadblocks, but I learned so much about datasets, models, and metrics along the way, improving my Python skills as well.
- **Julia Sun**: I coded the front-end for this project. It was my first time doing anything with html, CSS, and JavaScript, so I had to learn the languages while coding. This was a little challenging, but it felt really rewarding after the website started to come together.
- **Kavya Thyagarajan**: I primarily worked on the API, with some front-end contributions. Building the API with FastAPI deepened my understanding of Python—especially classes. On the front-end, I learned JavaScript from scratch, which was a steep learning curve, but ultimately a rewarding one.

---

## 🙏 Acknowledgments
- **mostly.ai** for synthetic data generation capabilities
- **FastAPI** for excellent API framework
- **Render** for free hosting tier

---

# 🎉 TunnelVision — Smarter Cities, Safer Infrastructure

**Built with ❤️ for the 2025 Artificiathon**

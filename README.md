# NJ Transit Smart Journey Assistant 🚆

## 🏆 Hackathon Achievement

**First Place - HackRU Fall 2024 (NJ Transit Track)**
**https://devpost.com/software/ontrack-nj-transit**
- Deployed App
**https://onnjtransit.streamlit.app**

## 📱 Overview

Enhancing NJ Transit passenger experience with AI and machine learning:

- **11** commuter rail lines
- **3** light rail lines
- **253** bus routes
- **133M+** annual riders

## ✨ Key Features

- **Smart Delay Prediction**: Provides real-time delay estimates with MAE 1.39 and RMSE 1.84
- **Mechanical Analytics**: Predictive maintenance, system monitoring
- **AI Support Assistant**: 24/7 chat, schedule and fare assistance

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Streamlit
- Any LLM API Key or Local (Ollama)

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/chetanreddyv/nj_transit.git
   cd nj_transit
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment:
   ```bash
   echo "gemini_api=your_api_key_here" > .env
   ```
4. Launch the app:
   ```bash
   streamlit run ON_NJ_Transit.py
   ```

## 📁 Project Structure

```
nj_transit/
├── ON_NJ_Transit.py
├── pages/
├── assets/
├── data/
├── models/
└── utils/
```

## 🔒 Security & Best Practices

- Use environment variables
- Regular API key rotation
- Implement rate limiting and encryption

## 📊 Data Sources & Tech Stack

- **Data**: NJ Transit API, historical records, NJ Transit Data
- **Tech**: Streamlit, Python, scikit-learn, Pandas, Plotly

## 🎯 Impact & Results

- **RMSE 1.84** delay prediction accuracy
- **24/7** customer support
- improvement in maintenance efficiency

## 👥 Team

- **Chetan Valluru**
- **Vasant Saladi**

## 🏆 Awards & Recognition

- First Place - HackRU Fall 2024
- Best Transportation Solution
- Best Use of NJ Transit Data

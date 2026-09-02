<div align="center">
  <h1>🍛 PlateWise: AI-Powered Indian Food Macro Estimator</h1>
  <p>An end-to-end Machine Learning web application that identifies Indian dishes from photos and estimates their nutritional macros.</p>
  
  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://plate-wise.streamlit.app/)
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
</div>

---

## 📖 Overview
**PlateWise** is a full-stack applied machine learning project designed to make nutrition tracking for complex Indian cuisines effortless. Built as a portfolio ML project, it allows users to upload a photo of a dish, automatically identifies the dish using a deep learning image classifier, and calculates its estimated calories and macronutrients (protein, carbs, fat) based on the user's selected serving size. 

From data acquisition and transfer learning to a dynamic, minimalist UI and live deployment, this project demonstrates a complete, production-ready AI pipeline.

## ✨ Features
- **📸 AI Image Classification**: Accurately identifies 80 different Indian dishes using a fine-tuned MobileNetV2 architecture.
- **📊 Real-time Macro Estimation**: Calculates Calories, Protein, Carbs, and Fats based on a custom-built JSON nutrition database.
- **⚖️ Dynamic Portion Sizing**: Adjust macro calculations on the fly by selecting custom serving sizes (in grams or pieces).
- **🔍 Fallback Search**: If the AI is unsure, seamlessly fall back to manual search to calculate macros for any of the 80 supported dishes.
- **📱 Responsive UI**: A beautifully clean, native Streamlit interface optimized for both desktop and mobile with native dark mode support.

## 🧠 Model Architecture & Transfer Learning
This project utilizes **MobileNetV2**, a lightweight and highly efficient convolutional neural network. 

**Why transfer learning?** Instead of training a CNN from scratch on a limited dataset (~4,000 images), which would likely lead to severe overfitting, we leveraged the feature extraction capabilities of a model pre-trained on the massive ImageNet dataset. We froze the base layers and fine-tuned a custom classification head. MobileNetV2 was chosen specifically for its speed and low parameter count (under 10MB), making it ideal for free-tier cloud deployment without hitting memory limits.

### Dataset
- **Source**: [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images) on Kaggle.
- **Scope**: 80 classes of common Indian dishes.

### Performance Metrics
- **Top-1 Accuracy**: `64.62%`
- **Top-3 Accuracy**: `85.62%`

*(Note: Top-3 accuracy is a much fairer metric for this task, as many Indian dishes—e.g., Paneer Butter Masala vs. Chicken Tikka Masala, or different types of Dal—share identical base gravies and look visually indistinguishable without contextual clues.)*

## 🛠️ Tech Stack
- **Machine Learning**: `PyTorch`, `TorchVision` (MobileNetV2)
- **Web Application & UI**: `Streamlit`
- **Data & Nutrition**: `Python (Pandas, Requests)`
- **Deployment**: `Streamlit Community Cloud`

## 🚀 Local Setup & Installation

Want to run PlateWise locally? Follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/indian-food-macros-estimator.git
   cd indian-food-macros-estimator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

*(To train the model yourself or run data processing scripts, install `requirements-dev.txt` instead.)*

## 📂 Repository Structure
```text
├── .streamlit/             # Streamlit theme configuration (Dark Mode default)
├── data/                   # JSON nutrition database 
├── models/                 # Pre-trained MobileNetV2 weights and class mappings
├── scripts/                # Data acquisition and preprocessing utilities
├── src/                    # PyTorch model architecture and training scripts
├── app.py                  # Main Streamlit web application
├── requirements.txt        # Production dependencies (CPU-only PyTorch)
├── requirements-dev.txt    # Development/Training dependencies
└── README.md               # Project documentation
```

## ⚠️ Known Limitations
- **Vocabulary Limit**: The model only recognizes 80 specific Indian food classes. It will attempt to reject unknown foods via a confidence threshold.
- **Manual Sizing**: The app cannot automatically detect portion size from the image due to lack of depth/scale reference.
- **Single Subject**: It cannot detect multiple foods in a single photo (e.g., a full Thali).

## 🌐 Live Demo
Experience the live application here: **[https://plate-wise.streamlit.app/](https://plate-wise.streamlit.app/)**

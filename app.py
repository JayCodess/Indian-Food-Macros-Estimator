import streamlit as st
import json
from PIL import Image
from src.model_utils import FoodClassifier
import pandas as pd

st.set_page_config(page_title="Indian Food Macros Estimator", page_icon="🍛", layout="centered")

# --- Custom CSS for a premium look ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    .metric-label {
        font-size: 1rem;
        color: #AAAAAA;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background-color: #333;
        margin-top: 5px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #FF4B4B, #FF8E53);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_classifier():
    return FoodClassifier()

@st.cache_data
def load_nutrition_db():
    try:
        with open("data/nutrition_table.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def main():
    st.title("🍛 Indian Food Macro Estimator")
    st.markdown("Upload a photo of an Indian dish, and we'll estimate its calories and macros!")
    
    classifier = get_classifier()
    nutrition_db = load_nutrition_db()
    
    if classifier.model is None:
        st.error("Model not found! Please ensure the model is trained and saved in the `models/` directory.")
        st.stop()
        
    if not nutrition_db:
        st.warning("Nutrition database is missing or empty. Macro calculations will not work.")
        
    # Input options
    source = st.radio("Choose image source:", ["Upload File", "Take Picture"], horizontal=True)
    
    img_file = None
    if source == "Upload File":
        img_file = st.file_uploader("Upload food image...", type=["jpg", "jpeg", "png"])
    else:
        img_file = st.camera_input("Take a picture")
        
    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing your food..."):
            is_confident, predictions = classifier.predict(image)
            
        if not predictions:
            st.error("Could not generate predictions.")
            return
            
        st.subheader("Classification Results")
        
        if not is_confident:
            st.error(f"⚠️ I'm not confident this is a food I recognize. The top guess was {predictions[0]['dish'].title()} ({predictions[0]['confidence']*100:.1f}%), but this is too low.")
            st.info("Try uploading a clearer photo of a single Indian dish.")
            return
            
        # Display top 3
        st.write("Here are my top guesses:")
        for p in predictions:
            dish = p['dish'].replace('_', ' ').title()
            conf = p['confidence'] * 100
            st.markdown(f"**{dish}**: {conf:.1f}%")
            st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf}%"></div>
                </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # Serving Size & Macros (Use top-1 prediction)
        top_dish = predictions[0]['dish']
        top_dish_info = nutrition_db.get(top_dish)
        
        if top_dish_info:
            st.subheader("Portion Size")
            unit_type = top_dish_info.get("unit_type", "weight")
            
            entered_amount = 0
            
            if unit_type == "weight":
                st.write("Select a preset serving size or enter exact grams:")
                col1, col2, col3 = st.columns(3)
                if col1.button("Small (150g)"): st.session_state.amount = 150
                if col2.button("Medium (250g)"): st.session_state.amount = 250
                if col3.button("Large (350g)"): st.session_state.amount = 350
                
                entered_amount = st.number_input("Or enter exact grams:", min_value=1, value=st.session_state.get('amount', 250), step=10)
                multiplier = entered_amount / 100.0
                serving_text = f"{entered_amount}g"
                
            else:
                st.write("Select a preset number of pieces or enter exact count:")
                col1, col2, col3 = st.columns(3)
                if col1.button("1 piece"): st.session_state.amount = 1
                if col2.button("2 pieces"): st.session_state.amount = 2
                if col3.button("3 pieces"): st.session_state.amount = 3
                
                entered_amount = st.number_input("Or enter exact count:", min_value=1, value=st.session_state.get('amount', 1), step=1)
                multiplier = entered_amount
                serving_text = f"{entered_amount} pieces"
                
            st.subheader(f"Nutrition for {serving_text} of {top_dish.replace('_', ' ').title()}")
            
            cals = top_dish_info.get("calories", 0) * multiplier
            prot = top_dish_info.get("protein_g", 0) * multiplier
            carbs = top_dish_info.get("carbs_g", 0) * multiplier
            fat = top_dish_info.get("fat_g", 0) * multiplier
            
            # Display Macros in a beautiful card layout
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{int(cals)}</div><div class="metric-label">Calories</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{int(prot)}g</div><div class="metric-label">Protein</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{int(carbs)}g</div><div class="metric-label">Carbs</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{int(fat)}g</div><div class="metric-label">Fat</div></div>', unsafe_allow_html=True)
                
        else:
            st.warning("Nutrition information for this dish is not available in our database.")

if __name__ == "__main__":
    main()

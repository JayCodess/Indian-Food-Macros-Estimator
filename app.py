import streamlit as st
import json
from PIL import Image
from src.model_utils import FoodClassifier
import pandas as pd

st.set_page_config(page_title="Indian Food Macro Estimator", page_icon="🍛", layout="centered", initial_sidebar_state="collapsed")

# Minimal CSS to keep radio buttons horizontally spaced nicely
st.markdown("""
<style>
    div[role="radiogroup"] {
        gap: 0.5rem;
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
        st.error("Model not found! Please train the model first.")
        st.stop()
        
    # --- Upload Section ---
    with st.container(border=True):
        st.markdown("### 📸 Upload your food")
        source = st.radio("Source:", ["Upload Photo", "Take Picture"], horizontal=True, label_visibility="collapsed")
        
        img_file = None
        if source == "Upload Photo":
            img_file = st.file_uploader("Drop your image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        else:
            img_file = st.camera_input("Take a picture", label_visibility="collapsed")
        
    # --- Results Section ---
    if img_file is not None:
        image = Image.open(img_file)
        
        with st.expander("👁️ View Uploaded Image", expanded=False):
            sub_col1, sub_col2, sub_col3 = st.columns([1, 4, 1])
            with sub_col2:
                st.image(image, use_column_width=True)
                
        with st.spinner("Analyzing spices and textures..."):
            is_confident, predictions = classifier.predict(image)
            
        if not predictions:
            st.error("Could not generate predictions.")
            st.stop()
            
        # --- Classification Results ---
        with st.container(border=True):
            st.markdown("### 🎯 What is this?")
            
            if not is_confident:
                st.warning(f"I'm not entirely sure! My best guess is {predictions[0]['dish'].replace('_', ' ').title()}, but I could be wrong.")
                
            for p in predictions:
                dish = p['dish'].replace('_', ' ').title()
                conf = p['confidence']
                st.write(f"**{dish}** - {conf*100:.1f}%")
                st.progress(float(conf))
                
            st.divider()
            
            # Dish confirmation
            top3_options = [p['dish'].replace('_', ' ').title() for p in predictions]
            options = top3_options + ["None of the above (Search manually)"]
            
            selected_option = st.selectbox("Confirm the correct dish to calculate macros:", options)
            
            all_classes = sorted(list(nutrition_db.keys()))
            if selected_option == "None of the above (Search manually)":
                all_classes_formatted = [c.replace('_', ' ').title() for c in all_classes]
                manual_selection = st.selectbox("Search database:", all_classes_formatted)
                selected_dish = all_classes[all_classes_formatted.index(manual_selection)]
            else:
                idx = options.index(selected_option)
                selected_dish = predictions[idx]['dish']
                
        # --- Nutrition Calculator ---
        top_dish_info = nutrition_db.get(selected_dish)
    
        if top_dish_info:
            with st.container(border=True):
                st.markdown(f"### ⚖️ Portion Size: {selected_dish.replace('_', ' ').title()}")
                unit_type = top_dish_info.get("unit_type", "weight")
                
                if unit_type == "weight":
                    size_choice = st.radio("Select Serving Size:", ["Small (150g)", "Medium (250g)", "Large (350g)", "Custom"], horizontal=True)
                    
                    if size_choice == "Custom":
                        entered_amount = st.number_input("Enter exact grams:", min_value=1, value=250, step=10)
                    else:
                        entered_amount = int(size_choice.split("(")[1].replace("g)", ""))
                        
                    multiplier = entered_amount / 100.0
                    serving_text = f"{entered_amount}g"
                else:
                    size_choice = st.radio("Select Quantity:", ["1 piece", "2 pieces", "3 pieces", "Custom"], horizontal=True)
                    
                    if size_choice == "Custom":
                        entered_amount = st.number_input("Enter exact pieces:", min_value=1, value=1, step=1)
                    else:
                        entered_amount = int(size_choice.split(" ")[0])
                        
                    multiplier = entered_amount
                    serving_text = f"{entered_amount} piece{'s' if entered_amount > 1 else ''}"
                    
                cals = top_dish_info.get("calories", 0) * multiplier
                prot = top_dish_info.get("protein_g", 0) * multiplier
                carbs = top_dish_info.get("carbs_g", 0) * multiplier
                fat = top_dish_info.get("fat_g", 0) * multiplier
                
                st.divider()
                st.markdown(f"**Estimated Macros for {serving_text}**")
                
                # Use native Streamlit metrics
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Calories", f"{int(cals)}")
                m_col2.metric("Protein", f"{int(prot)}g")
                m_col3.metric("Carbs", f"{int(carbs)}g")
                m_col4.metric("Fat", f"{int(fat)}g")
                
        else:
            st.warning("Nutrition information for this dish is not available in our database.")

if __name__ == "__main__":
    main()

# 🍛 Indian Food Calorie & Macro Estimator

## Overview
The **Indian Food Calorie & Macro Estimator** is an end-to-end machine learning web application that allows users to upload a photo of an Indian dish, automatically identifies the dish using a deep learning image classifier, and calculates its estimated calories and macronutrients (protein, carbs, fat) based on the user's selected serving size. This project was built to demonstrate a full-stack applied ML pipeline—from data acquisition and transfer learning to a dynamic user interface and live deployment—making nutrition tracking for complex Indian cuisines much easier.

## Dataset
- **Source**: [Indian Food Images Dataset by iamsouravbanerjee](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images) on Kaggle.
- **Classes**: Started with 80 classes of Indian food. We kept all 80 classes as they each had more than 40 images, providing a sufficient baseline for transfer learning without extreme class imbalances.

## Model Architecture & Transfer Learning
This project utilizes **MobileNetV2**, a lightweight and highly efficient convolutional neural network. 
**Why transfer learning?** Instead of training a CNN from scratch on only ~4,000 images—which would likely lead to severe overfitting and poor generalization—we leveraged the feature extraction capabilities of a model pre-trained on the massive ImageNet dataset. We froze the base layers and fine-tuned a custom classification head to recognize our 80 specific Indian food classes. MobileNetV2 was chosen specifically for its speed and low parameter count, making it ideal for free-tier cloud deployment.

## Model Performance
- **Top-1 Accuracy**: 64.62%
- **Top-3 Accuracy**: 85.62%

*(Note: Top-3 accuracy is a much fairer metric for this task, as many Indian dishes (e.g., Paneer Butter Masala vs. Chicken Tikka Masala, or different types of Dal) look visually identical even to humans.)*

### Example Predictions
1. **[Success]**: The model correctly identified `biryani` with high confidence.
2. **[Success]**: The model correctly identified `samosa` as its first prediction.
3. **[Failure]**: The model predicted `kadai_paneer` instead of `paneer_butter_masala`. *Analysis: Both dishes feature a similar orange-red gravy and paneer cubes, making the visual distinction extremely difficult without contextual clues.*

## Tech Stack
- **Machine Learning**: PyTorch, TorchVision (MobileNetV2), Scikit-Learn
- **Web Application & UI**: Streamlit, Custom CSS
- **Data & Nutrition**: Python (Requests, Pandas), USDA FoodData Central API
- **Deployment**: Streamlit Community Cloud (or Hugging Face Spaces)

## Known Limitations
- **Limited Vocabulary**: The model cannot recognize general or non-Indian foods beyond the 80 trained classes. If an unknown food is uploaded, it uses a confidence threshold to reject the prediction.
- **Manual Portion Sizing**: The app does not automatically detect the portion size or weight from the image (due to lack of depth/scale reference). The user must manually input their serving size.
- **Single Dish Only**: It cannot detect multiple foods in a single photo (e.g., a full Thali). The image must be focused on a single dish.
- **No User Accounts**: There is no day-by-day tracking or history logging in this version.

## Live Demo
Check out the live application here: **[https://plate-wise.streamlit.app/]**

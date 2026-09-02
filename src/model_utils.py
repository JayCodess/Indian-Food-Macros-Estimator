import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import json

MODEL_PATH = "models/best_model.pth"
CLASSES_PATH = "models/classes.json"
CONFIDENCE_THRESHOLD = 0.05  # Lowered to 5% because 80 classes spread probabilities very thin after only 5 epochs.

class FoodClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self._load_classes()
        self.model = self._load_model()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def _load_classes(self):
        try:
            with open(CLASSES_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def _load_model(self):
        if not self.classes:
            return None
            
        model = models.mobilenet_v2(weights=None)
        # Replace classifier
        model.classifier[1] = nn.Linear(model.last_channel, len(self.classes))
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
        except FileNotFoundError:
            print("Model weights not found. Please train the model first.")
            return None
            
        model = model.to(self.device)
        model.eval()
        return model
        
    def predict(self, image):
        """
        Returns (is_confident, list of (class_name, confidence_percentage))
        """
        if self.model is None or not self.classes:
            return False, []
            
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_t)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # Get top 3
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        
        results = []
        for i in range(top3_prob.size(0)):
            results.append({
                "dish": self.classes[top3_catid[i].item()],
                "confidence": top3_prob[i].item()
            })
            
        is_confident = results[0]["confidence"] >= CONFIDENCE_THRESHOLD
        
        return is_confident, results

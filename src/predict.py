import torch
from torchvision import transforms
from PIL import Image
import os
from train import XRayCNN  # Make sure XRayCNN is properly imported

class XRayPredictor:
    def __init__(self, model_path):
        """Initialize the predictor with a trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and load weights
        self.model = XRayCNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except RuntimeError as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
            
        self.model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Make prediction on a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Prediction class (COVID19/NORMAL/PNEUMONIA)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = torch.max(outputs.data, 1)
                
            return ["COVID19", "NORMAL", "PNEUMONIA"][predicted.item()]
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    # Example usage
    try:
        predictor = XRayPredictor("models/best_model.pth")
        test_image = "data/test/COVID19/COVID19(460).jpg"
        
        if not os.path.exists(test_image):
            test_image = input("Enter path to test image: ")
            
        result = predictor.predict(test_image)
        print(f"Prediction: {result}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
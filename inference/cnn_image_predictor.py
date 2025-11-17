import numpy as np
import torch
import torchvision
import io
import base64
import mlflow.pyfunc
import cv2
from PIL import Image
from inference.gradcam import GradCAM


class CNNImagePredictor(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for MobileNetV3 and EfficientNet with GradCAM explainability.
    Handles 16-bit PNG and DICOM images via base64 encoding.
    """
    
    def load_context(self, context):
        """Load model and setup preprocessing pipeline."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load PyTorch model
        self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup GradCAM (target last convolutional layer before classifier)
        target_layer = self.model.features[-1]
        self.gradcam = GradCAM(self.model, target_layer)
        
        # Define ToFloatTensor16Bit inline
        class ToFloatTensor16Bit:
            def __call__(self, image):
                """Convert PIL Image to float32 tensor, preserving 16-bit range."""
                arr = np.array(image, dtype=np.float32)
                arr = arr / 65535.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=0)
                else:
                    arr = arr.transpose(2, 0, 1)
                return torch.from_numpy(arr)
        
        self.to_float_tensor = ToFloatTensor16Bit()
        
        # Preprocessing transforms
        self.resize_transform = torchvision.transforms.Resize((224, 224))
        self.normalize_transform = torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
        self.class_names = ['benign', 'benign_without_callback', 'malignant']
    
    def _decode_image(self, img_data_str, file_type='.png'):
        """Decode base64 image string to PIL Image."""
        from utils.dicom import load_image
        import tempfile
        import os
        import time
        
        if 'base64,' in img_data_str:
            img_data_str = img_data_str.split('base64,')[1]
        
        img_bytes = base64.b64decode(img_data_str)
        
        with tempfile.NamedTemporaryFile(suffix=file_type, delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        
        try:
            image = load_image(tmp_path)
            image.load()
            return image
        finally:
            for attempt in range(3):
                try:
                    os.unlink(tmp_path)
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(0.1)
    
    def _generate_gradcam_overlay(self, image_pil, heatmap):
        """
        Create overlay of GradCAM heatmap on original image.
        
        Args:
            image_pil: PIL Image (original, 16-bit)
            heatmap: GradCAM heatmap numpy array (H, W) in [0, 1]
        
        Returns:
            base64 encoded PNG image of the overlay
        """
        # Convert PIL to numpy (normalize to 8-bit for visualization)
        img_arr = np.array(image_pil, dtype=np.float32)
        img_arr = (img_arr / 65535.0 * 255).astype(np.uint8)
        
        # Resize to match model input size
        img_arr = cv2.resize(img_arr, (224, 224))
        
        # Convert grayscale to RGB
        if img_arr.ndim == 2:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        
        # Apply colormap to heatmap (jet colormap)
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Overlay: 60% original image + 40% heatmap
        overlay = cv2.addWeighted(img_arr, 0.6, heatmap_colored, 0.4, 0)
        
        # Convert to PIL and then base64
        overlay_pil = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format='PNG')
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return overlay_base64
    
    def predict(self, context, model_input):
        """
        Run inference with GradCAM explainability.
        
        Input (pandas DataFrame):
            - 'image': base64 encoded image string
            - 'file_type': optional, '.png' or '.dcm' (default: '.png')
            - 'include_gradcam': optional, bool (default: True)
        
        Output (list of dicts):
            - predicted_class: int (0=benign, 1=benign_without_callback, 2=malignant)
            - predicted_class_name: str
            - confidence: float (probability of predicted class)
            - probabilities: dict with all class probabilities
            - diagnosis: str (human-readable diagnosis)
            - gradcam_overlay: base64 encoded PNG (if include_gradcam=True)
            - gradcam_heatmap: base64 encoded heatmap only (if include_gradcam=True)
        """
        predictions = []
        
        for idx, row in model_input.iterrows():
            try:
                img_data = row['image']
                file_type = row.get('file_type', '.png')
                include_gradcam = row.get('include_gradcam', True)
                
                # Decode image
                image = self._decode_image(img_data, file_type)
                
                # Preprocess for model
                image_resized = self.resize_transform(image)
                img_tensor = self.to_float_tensor(image_resized)
                img_tensor = self.normalize_transform(img_tensor)
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                
                # Enable gradients for GradCAM
                img_tensor.requires_grad = True
                
                # Run inference
                with torch.set_grad_enabled(True):
                    output = self.model(img_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # Generate diagnosis text
                diagnosis_map = {
                    0: "BENIGN - No signs of malignancy detected",
                    1: "BENIGN WITHOUT CALLBACK - No abnormalities, no follow-up needed",
                    2: "MALIGNANT - Potential cancerous findings detected. Urgent follow-up recommended."
                }
                diagnosis = diagnosis_map[predicted_class]
                
                # Prepare response
                result = {
                    'predicted_class': predicted_class,
                    'predicted_class_name': self.class_names[predicted_class],
                    'confidence': float(confidence),
                    'probabilities': {
                        'benign': float(probabilities[0][0]),
                        'benign_without_callback': float(probabilities[0][1]),
                        'malignant': float(probabilities[0][2])
                    },
                    'diagnosis': diagnosis
                }
                
                # Generate GradCAM if requested
                if include_gradcam:
                    heatmap = self.gradcam.generate(img_tensor, target_class=predicted_class)
                    
                    # Generate overlay
                    overlay_base64 = self._generate_gradcam_overlay(image_resized, heatmap)
                    
                    # Generate heatmap only (for separate display)
                    heatmap_colored = cv2.applyColorMap(
                        (heatmap * 255).astype(np.uint8),
                        cv2.COLORMAP_JET
                    )
                    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
                    buffer = io.BytesIO()
                    heatmap_pil.save(buffer, format='PNG')
                    heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    result['gradcam_overlay'] = overlay_base64
                    result['gradcam_heatmap'] = heatmap_base64
                
                predictions.append(result)
                
            except Exception as e:
                import traceback
                predictions.append({
                    'predicted_class': -1,
                    'predicted_class_name': 'error',
                    'confidence': 0.0,
                    'probabilities': {
                        'benign': 0.0,
                        'benign_without_callback': 0.0,
                        'malignant': 0.0
                    },
                    'diagnosis': 'Error during prediction',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        return predictions

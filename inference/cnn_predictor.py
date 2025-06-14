from pathlib import Path
from typing import Union, List, Dict #se agregó List, Dict
from datetime import datetime

import numpy as np
import torch
import csv
from PIL import Image

from inference.predictor_interface import PredictorInterface
from utils.dicom import load_image


class CNNPredictor(PredictorInterface):
    """
    A CNN-based predictor that uses a trained model to make predictions on images.
    
    This class implements the PredictorInterface and provides functionality to:
    - Load a trained CNN model
    - Transform input images to match model requirements
    - Make predictions and return probabilities
    """
    def __init__(self, strategy, model_path, device=None, label_list=None):
        """
        Initialize the CNN predictor with a model strategy and trained weights.

        Args:
            strategy: Model strategy object that provides model architecture and transforms
            model_path (str): Path to the trained model weights file
            device (torch.device, optional): Device to run inference on. Defaults to CUDA if available, else CPU.
        """
        self.strategy = strategy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = strategy.get_model().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = strategy.get_transforms(train=False)
        self.label_list = label_list or ["CLASS_0", "CLASS_1", "CLASS_2"]

    def predict(self, image: Image.Image) -> dict:
        """
        Make a prediction on the input image.

        Args:
            image (PIL.Image): Input image to classify

        Returns:
            dict: Dictionary containing:
                - prediction (int): Predicted class index
                - confidence (float): Confidence score for the prediction
                - all_probs (numpy.ndarray): Probability distribution over all classes
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred = int(np.argmax(probs))
            class_label = self.label_list[pred] if self.label_list and pred < len(self.label_list) else str(pred)
        return {
            "prediction": pred,
            "class_label": class_label,
            "confidence": round(float(probs[pred]), 3),
            "all_probs": np.round(probs, 3)
        }


    def predict_from_path(self, image_path: Union[str, Image.Image]) -> dict:
        """
        Load and preprocess image from path, then run prediction.
        Should return the same output as `predict()`, with optional label mapping.

        Args:
            image_path (Union[str, Image.Image]): Path to image file or PIL Image object

        Returns:
            dict: Dictionary containing:
                - prediction (int): Predicted class index
                - confidence (float): Confidence score for the prediction
                - all_probs (numpy.ndarray): Probability distribution over all classes
                - class_label (str): Human-readable class label
        """
        if isinstance(image_path, str):
            image = load_image(image_path)
        else:
            image = image_path

        result = self.predict(image)
        if self.label_list:
            result["class_label"] = self.label_list[result["prediction"]]
        return result

    def predict_directory(
        self,
        dir_path: str,
        correlation_id: str,
        output_path: str,
        output_file_name: str = "inference.csv",
        recursive: bool = False,
        allowed_exts={".dcm", ".jpg", ".png"}
    ) -> List[Dict]: # se modificó el original era list[dict]
        """
        Predict on all images in a directory and write results to a timestamped CSV.

        Args:
            dir_path (str): Directory containing images.
            correlation_id (str): Unique ID for experiment folder.
            output_path (str): Base output directory.
            output_file_name (str): Base name of output file (timestamp will be appended).
            recursive (bool): Whether to include subfolders.
            allowed_exts (set): Allowed image extensions.

        Returns:
            List[dict]: List of prediction result dicts.
        """
        results = []
        dir_path = Path(dir_path)
        out_dir = Path(output_path) / correlation_id
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = output_file_name.rsplit(".", 1)[0]
        final_name = f"{base_name}_{timestamp}.csv"
        csv_path = out_dir / final_name

        files = dir_path.rglob("*") if recursive else dir_path.glob("*")

        with open(csv_path, "w", newline="") as f:
            fieldnames = ["filename", "filepath", "prediction", "class_label", "confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for file in files:
                if file.suffix.lower() in allowed_exts:
                    try:
                        result = self.predict_from_path(str(file))
                        result["filename"] = file.name
                        result["filepath"] = str(file.resolve())

                        writer.writerow({
                            "filename": result["filename"],
                            "filepath": result["filepath"],
                            "prediction": result["prediction"],
                            "class_label": result.get("class_label", "N/A"),
                            "confidence": result["confidence"]
                        })

                        results.append(result)
                    except Exception as e:
                        print(f"[WARN] Failed to predict {file.name}: {e}")

        print(f"[INFO] Results saved to {csv_path}")
        return results
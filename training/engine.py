import os
import time
from collections import defaultdict
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from tqdm.notebook import tqdm


from training.early_stopping import EarlyStopping


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Run one epoch of training.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

def evaluate_ml(model, X_test, y_test, view_info=None, class_names=None):
    """
    Evaluate sklearn-compatible models (SVM, XGBoost, RandomForest, etc.)
    
    Args:
        model: Trained sklearn-compatible model (must have predict and predict_proba)
        X_test (np.ndarray or pd.DataFrame): Test features, shape (n_samples, n_features)
        y_test (np.ndarray or pd.Series): Test labels, shape (n_samples,)
        view_info (list or np.ndarray, optional): List of view names corresponding to each sample.
                                                   Must have length n_samples.
        class_names (list, optional): List of class names. If None, uses class_0, class_1, etc.
        
    Returns:
        tuple: (accuracy, avg_loss, recall, precision, f1, auc, 
                per_view_metrics, per_view_predictions, per_class_metrics)
            - accuracy (float): Classification accuracy
            - avg_loss (None): Not applicable for sklearn models
            - recall (float): Macro-averaged recall score
            - precision (float): Macro-averaged precision score
            - f1 (float): Macro-averaged F1 score
            - auc (float or None): Macro-averaged ROC AUC score if possible, None otherwise
            - per_view_metrics (dict): Dictionary containing per-view metrics
            - per_view_predictions (dict): Dictionary containing per-view predictions
            - per_class_metrics (dict): Dictionary containing per-class metrics
    """
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities (handle models that don't support it)
    try:
        y_prob = model.predict_proba(X_test)
    except AttributeError:
        y_prob = None
    
    # Convert to numpy if needed
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    # Calculate global metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # AUC calculation
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            auc = None
    else:
        auc = None
    
    # Loss approximation (sklearn models don't have a direct loss)
    avg_loss = None
    
    # Per-view metrics (if view information provided)
    per_view_metrics = {}
    per_view_predictions = {}
    
    if view_info is not None:
        per_view = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_prob": []})
        
        for i, view in enumerate(view_info):
            per_view[view]["y_true"].append(y_test[i])
            per_view[view]["y_pred"].append(y_pred[i])
            if y_prob is not None:
                per_view[view]["y_prob"].append(y_prob[i].tolist())
        
        for view, d in per_view.items():
            if d["y_true"]:
                view_auc = None
                if y_prob is not None and d["y_prob"]:
                    try:
                        view_auc = roc_auc_score(
                            d["y_true"], d["y_prob"], 
                            multi_class='ovr', average='macro'
                        )
                    except ValueError:
                        pass
                
                per_view_metrics[view] = {
                    "recall": recall_score(d["y_true"], d["y_pred"], average='macro', zero_division=0),
                    "precision": precision_score(d["y_true"], d["y_pred"], average='macro', zero_division=0),
                    "f1": f1_score(d["y_true"], d["y_pred"], average='macro', zero_division=0),
                    "accuracy": np.mean(np.array(d["y_true"]) == np.array(d["y_pred"])),
                    "auc": view_auc
                }
                
                per_view_predictions[view] = {
                    "y_true": d["y_true"],
                    "y_pred": d["y_pred"]
                }
    
    # Per-class metrics
    model_classes = getattr(model, "classes_", None)
    class_values = list(model_classes) if model_classes is not None else list(np.unique(y_test))

    if class_names is not None and len(class_names) == len(class_values):
        class_labels = list(class_names)
    else:
        class_labels = [str(cls) for cls in class_values]

    per_class_metrics = {}
    for i, (class_value, label) in enumerate(zip(class_values, class_labels)):
        y_true_bin = (y_test == class_value).astype(int)
        y_pred_bin = (y_pred == class_value).astype(int)
        
        # Binary AUC for this class
        class_auc = None
        if y_prob is not None:
            try:
                prob_bin = y_prob[:, i]
                class_auc = roc_auc_score(y_true_bin, prob_bin)
            except (ValueError, IndexError):
                pass
        
        acc = np.mean(y_true_bin == y_pred_bin)
        
        per_class_metrics[label] = {
            "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
            "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
            "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            "accuracy": acc,
            "auc_roc": class_auc
        }
    
    return (
        accuracy,
        avg_loss,
        recall,
        precision,
        f1,
        auc,
        per_view_metrics,
        per_view_predictions,
        per_class_metrics
    )

def evaluate_epoch(model, loader, criterion, device):
    """
    Run one epoch of evaluation.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Evaluation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        tuple: (average_loss, accuracy, recall, precision, f1, auc, per_view_metrics)
            - average_loss (float): Average evaluation loss
            - accuracy (float): Classification accuracy
            - recall (float): Macro-averaged recall score
            - precision (float): Macro-averaged precision score
            - f1 (float): Macro-averaged F1 score
            - auc (float or None): Macro-averaged ROC AUC score if possible, None otherwise
            - per_view_metrics (dict): Dictionary containing per-view metrics
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds, all_labels, all_probs = [], [], []

    per_view = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_prob": []})

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if hasattr(loader.dataset, "samples"):
                for i, (_, view) in enumerate(loader.dataset.samples[batch_idx * loader.batch_size: batch_idx * loader.batch_size + images.size(0)]):
                    per_view[view]["y_true"].append(labels[i].item())
                    per_view[view]["y_pred"].append(preds[i].item())
                    per_view[view]["y_prob"].append(probs[i].cpu().tolist())

    # Global metrics
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = None

    per_view_metrics = {}
    for view, d in per_view.items():
        if d["y_true"]:
            try:
                auc_score = roc_auc_score(d["y_true"], d["y_prob"], multi_class='ovr', average='macro')
            except ValueError:
                auc_score = None

            per_view_metrics[view] = {
                "recall": recall_score(d["y_true"], d["y_pred"], average='macro', zero_division=0),
                "precision": precision_score(d["y_true"], d["y_pred"], average='macro', zero_division=0),
                "f1": f1_score(d["y_true"], d["y_pred"], average='macro', zero_division=0),
                "accuracy": np.mean(np.array(d["y_true"]) == np.array(d["y_pred"])),
                "auc": auc_score
            }

    per_view_predictions = {
        view: {
            "y_true": d["y_true"],
            "y_pred": d["y_pred"]
        }
        for view, d in per_view.items()
    }

    # Per-class metrics
    n_classes = len(set(all_labels))
    class_labels = loader.dataset.classes if hasattr(loader.dataset, "classes") else [f"class_{i}" for i in range(n_classes)]

    per_class_metrics = {}
    for i, label in enumerate(class_labels):
        y_true_bin = np.array([1 if y == i else 0 for y in all_labels])
        y_pred_bin = np.array([1 if y == i else 0 for y in all_preds])
        prob_bin = np.array([p[i] for p in all_probs])

        try:
            auc_roc = roc_auc_score(y_true_bin, prob_bin)
        except ValueError as e:
            tqdm.write(f"AUC computation failed: {repr(e)}")
            raise  # don't skip — stop execution

        acc = np.mean(y_true_bin == y_pred_bin)

        per_class_metrics[label] = {
            "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
            "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
            "f1": f1_score(y_true_bin, y_pred_bin, zero_division=0),
            "accuracy": acc,
            "auc_roc": auc_roc
        }

    return (
        accuracy,
        avg_loss,
        recall,
        precision,
        f1,
        auc,
        per_view_metrics,
        per_view_predictions,
        per_class_metrics
    )


class Trainer:
    """
    Encapsulates the training and evaluation logic for a PyTorch model like ResNet, EfficientNet, etc.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 optimizer: Optimizer, criterion: nn.Module,
                 scheduler: ReduceLROnPlateau = None,
                 early_stopping: EarlyStopping = None):
        """
        Initialize Trainer.

        Args:
            model (nn.Module): Model to train.
            device (torch.device): CUDA or CPU device.
            optimizer (Optimizer): Optimizer to use (e.g., Adam).
            criterion (nn.Module): Loss function.
            scheduler (ReduceLROnPlateau, optional): Learning rate scheduler.
            early_stopping: EarlyStopping class to stop before overfitting.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_recall": [],
            "val_precision": [],
            "val_f1": [],
            "val_auc": [],
            "val_views": defaultdict(list),
            "val_view_predictions": [],
            "val_class_metrics": [],
            "epoch_time": [],
            "max_memory_mb": [],
        }

        self.best_model_wts = None
        self.best_val_recall = float("-inf")
        self.save_path = None
        self.early_stopping = early_stopping


    def fit(self, train_loader, val_loader, epochs=20):
        """
        Train the model for a given number of epochs.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Number of training epochs.
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            start_time = time.time()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            train_loss = train_epoch(self.model, train_loader, self.criterion, self.optimizer, self.device)

            val_acc, val_loss, val_recall, val_precision, val_f1, val_auc, val_views, val_view_predictions, val_class_metrics  = evaluate_epoch(
                self.model, val_loader, self.criterion, self.device
            )

            elapsed_time = time.time() - start_time
            self.history["epoch_time"].append(elapsed_time)

            if torch.cuda.is_available():
                mem_bytes = torch.cuda.max_memory_allocated(self.device)
                self.history["max_memory_mb"].append(mem_bytes / 1024 ** 2)  # in MB
            else:
                process = psutil.Process(os.getpid())
                mem_bytes = process.memory_info().rss
                self.history["max_memory_mb"].append(mem_bytes / 1024 ** 2)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_recall"].append(val_recall)
            self.history["val_precision"].append(val_precision)
            self.history["val_f1"].append(val_f1)
            self.history["val_auc"].append(val_auc)

            # Flatten per-class metrics into individual history keys for plotting
            self.history.setdefault("val_class_metrics", []).append(val_class_metrics)
            for class_name, metrics in val_class_metrics.items():
                for metric_name, value in metrics.items():
                    key = f"{class_name}_{metric_name}"
                    self.history.setdefault(key, []).append(value)

            self.history.setdefault("val_view_predictions", []).append(val_view_predictions)
            for view, metrics in val_views.items():
                for k, v in metrics.items():
                    self.history["val_views"][(view, k)].append(v)

            if self.scheduler:
                self.scheduler.step(val_recall)

            if val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
                self.best_model_wts = self.model.state_dict()

            auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
            print(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Recall: {val_recall:.4f} | "
                f"Precision: {val_precision:.4f} | F1: {val_f1:.4f} | AUC: {auc_str} | "
                f"Time: {elapsed_time:.2f}s | Max Mem: {self.history['max_memory_mb'][-1]:.2f} MB"
            )

            if self.early_stopping:
                monitored_value = {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_recall": val_recall,
                    "val_precision": val_precision,
                    "val_f1": val_f1,
                    "val_auc": val_auc
                }.get(self.early_stopping.monitor)

                if monitored_value is None:
                    raise ValueError(
                        f"Unknown metric: {self.early_stopping.monitor}. "
                        f"Valid metrics are: val_loss, val_acc, val_recall, val_precision, val_f1, val_auc"
                    )

                stop = self.early_stopping.step(monitored_value)
                if stop:
                    print(f"Early stopping triggered at epoch {epoch + 1} (no improvement in {self.early_stopping.monitor}).")
                    break


    def evaluate(self, test_loader):
        """
        Evaluate the trained model on a test set.

        Args:
            test_loader (DataLoader): Test data.

        Returns:
            Tuple[float, float, float, float, float]:
            (accuracy %, avg loss, recall, precision, f1)
        """
        return evaluate_epoch(self.model, test_loader, self.criterion, self.device)

    def get_history(self):
        """
        Get training history.

        Returns:
            dict: Training metrics over all epochs.
        """
        return self.history

    def get_model(self):
        """
        Get the trained model.

        Returns:
            nn.Module: Trained PyTorch model.
        """
        return self.model

    def save_best_model(self, save_path: str = "models/best_model.pth"):
        """
        Save the best model weights (based on validation accuracy).

        Args:
            save_path (str): File path to save the model weights.
        """
        self.save_path = save_path
        if self.best_model_wts:
            torch.save(self.best_model_wts, self.save_path)
            print(f"Best model saved to: {self.save_path} (val_recall: {self.best_val_recall:.2f}%)")
        else:
            print("No best model weights available to save.")

    def load_model(self, path: str):
        """
        Load model weights from a file.

        Args:
            path (str): File path of the saved weights.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"Model loaded from: {path}")

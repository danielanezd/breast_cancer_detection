# Model endpoints
MODEL_ENDPOINTS = {
    "MobileNet": "http://localhost:5001/invocations",
    "EfficientNet": "http://localhost:5002/invocations",
    "ResNet": "http://localhost:5003/invocations",
    # "SVM": "http://localhost:5004/invocations",
    # "XGBoost": "http://localhost:5005/invocations"
}

# Class labels
CLASS_NAMES = ['benign', 'benign_without_callback', 'malignant']

# UI Settings
APP_TITLE = "Breast Cancer Screening Tool"
SUPPORTED_FORMATS = ['png', 'dcm']
MAX_FILE_SIZE_MB = 50
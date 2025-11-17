import requests
from config import MODEL_ENDPOINTS


def predict_all_models(img_b64, file_type):
    """
    Send image to all CNN model endpoints and collect results.
    
    Args:
        img_b64: Base64 encoded image string
        file_type: '.png' or '.dcm'
    
    Returns:
        dict: {model_name: prediction_result}
    """
    results = {}
    
    payload = {
        "dataframe_split": {
            "columns": ["image", "file_type", "include_gradcam"],
            "data": [[img_b64, file_type, True]]
        }
    }
    
    for model_name, endpoint in MODEL_ENDPOINTS.items():
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30  # 30 seconds timeout
            )
            response.raise_for_status()
            
            # Extract first prediction from response
            results[model_name] = response.json()['predictions'][0]
            
        except requests.exceptions.Timeout:
            results[model_name] = {
                'error': 'Request timeout',
                'predicted_class': -1,
                'predicted_class_name': 'error'
            }
        except requests.exceptions.RequestException as e:
            results[model_name] = {
                'error': f'Connection error: {str(e)}',
                'predicted_class': -1,
                'predicted_class_name': 'error'
            }
        except (KeyError, ValueError) as e:
            results[model_name] = {
                'error': f'Invalid response: {str(e)}',
                'predicted_class': -1,
                'predicted_class_name': 'error'
            }
    
    return results


def predict_single_model(img_b64, file_type, model_name):
    """
    Send image to a single model endpoint.
    
    Args:
        img_b64: Base64 encoded image string
        file_type: '.png' or '.dcm'
        model_name: Name of model from MODEL_ENDPOINTS
    
    Returns:
        dict: Prediction result
    """
    if model_name not in MODEL_ENDPOINTS:
        return {
            'error': f'Unknown model: {model_name}',
            'predicted_class': -1,
            'predicted_class_name': 'error'
        }
    
    endpoint = MODEL_ENDPOINTS[model_name]
    payload = {
        "dataframe_split": {
            "columns": ["image", "file_type", "include_gradcam"],
            "data": [[img_b64, file_type, True]]
        }
    }
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        response.raise_for_status()
        return response.json()['predictions'][0]
        
    except Exception as e:
        return {
            'error': str(e),
            'predicted_class': -1,
            'predicted_class_name': 'error'
        }
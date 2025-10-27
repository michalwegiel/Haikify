import cv2
import numpy as np
from keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input
)
from PIL import Image

MODEL = MobileNetV2(weights="imagenet")


def preprocess_image(image: Image) -> np.ndarray:
    """
    Convert a PIL image to a preprocessed NumPy array ready for MobileNetV2 inference.

    Steps
    -----
        1. Convert the PIL Image to a NumPy array (RGB).
        2. Resize to (224, 224) pixels, the MobileNetV2 input size.
        3. Apply the MobileNetV2-specific preprocessing function.

    Parameters
    ----------
    image: Image
        Input PIL image.

    Returns
    -------
    np.ndarray
        Preprocessed image array of shape (224, 224, 3).
    """
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    return img


def classify(image: Image) -> list:
    """
    Classify a single image using the pretrained MobileNetV2 model.

    Parameters
    ----------
    image: Image
        Input PIL image to classify.

    Returns
    -------
    list[tuple[str, str, float]]
        Top-3 predictions as tuples of (class_id, class_name, confidence_score).
    """
    processed_image = preprocess_image(image)
    predictions = MODEL.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions


def classify_images(image_list: list[Image]) -> list[str]:
    """
    Classify multiple images and return the top-1 class name for each.

    Parameters
    ----------
    image_list: list[Image]
        A list of PIL images to classify.

    Returns
    -------
    list[str]
        A list of predicted class names (human-readable) for each image.
    """
    processed_images = [preprocess_image(img) for img in image_list]
    batch = np.array(processed_images)

    predictions = MODEL.predict(batch)

    decoded = decode_predictions(predictions, top=1)
    predictions = []
    for item in decoded:
        _, prediction, _ = item[0]
        prediction = prediction.replace("_", " ")
        predictions.append(prediction)
    return predictions

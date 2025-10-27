import cv2
import numpy as np
from keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

MODEL = MobileNetV2(weights="imagenet")


def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    # img = np.expand_dims(img, axis=0)
    return img


def classify(image):
    processed_image = preprocess_image(image)
    predictions = MODEL.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions


def classify_images(image_list):
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

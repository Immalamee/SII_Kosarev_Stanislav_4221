import sys
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def predict(img_path):
    model = mobilenet_v2.MobileNetV2(weights="imagenet")
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=5)[0]

    print(f"\nTop predictions for '{img_path}':")
    found = False
    for i, (imagenetID, label, prob) in enumerate(decoded):
        print(f"{i+1}. {label} ({prob:.4f})")
        if "toilet" in label.lower():
            found = True
            conf = prob
    if found:
        print(f"\n Toilet detected! Confidence: {conf:.2f}")
    else:
        print("\n Toilet not detected.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_toilet.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])

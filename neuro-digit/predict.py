import sys
import numpy as np
from tensorflow import keras
from preprocess_paint import preprocess_image

def main(image_path):
    model = keras.models.load_model('model.h5')
    x = preprocess_image(image_path) 
    x = np.expand_dims(x, 0) 
    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    probs = preds[0]
    print(f"Predicted digit: {pred_class}")
    print("Top probabilities:")
    for i, p in enumerate(probs):
        print(f"  {i}: {p:.4f}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.png")
        sys.exit(1)
    main(sys.argv[1])

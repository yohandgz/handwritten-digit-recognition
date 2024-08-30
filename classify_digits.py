import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def classify_digits():
    # loading the model
    model = tf.keras.models.load_model("digit-recognition.keras")

    # starting at 0, referencing name of png files
    image_number = 0
    while os.path.isfile(f"assets/digits/{image_number}.png"):
        try:
            img = cv2.imread(f"assets/digits/{image_number}.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = np.invert(img)
            img = img.reshape(1, 28, 28, 1) / 255.0
            prediction = model.predict(img)
            print(f"This digit is a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print(f"Error! {e}")
        finally:
            image_number += 1

if __name__ == "__main__":
    classify_digits()

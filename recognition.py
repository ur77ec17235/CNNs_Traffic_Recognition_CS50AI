import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from traffic import load_model, traffic_sign_names


def load_test_data(test_dir, img_width, img_height):
    test_images = []

    for file_name in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file_name)
        image = cv2.imread(file_path)

        if image is not None:
            try:
                image = cv2.resize(image, (img_width, img_height))
                image = image / 255.0
                test_images.append(image)
            except cv2.error as e:
                print(f"Error processing file {file_path}: {e}")
        else:
            print(f"Skipping invalid file {file_path}")

    return test_images

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python recognize.py model.h5 test_directory")

    model_path = sys.argv[1]
    test_dir = sys.argv[2]

    model = load_model(model_path)
    x_test = load_test_data(test_dir, model.input_shape[1], model.input_shape[2])
    predictions = model.predict(np.array(x_test))

    for i, image in enumerate(x_test):
        predicted_label = np.argmax(predictions[i])
        traffic_sign_name = traffic_sign_names.get(predicted_label, 'Unknown')

        image = cv2.resize(image, (300, 300))
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imshow(f"Traffic Sign: {traffic_sign_name}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
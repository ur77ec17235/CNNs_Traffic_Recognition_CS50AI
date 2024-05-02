import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


traffic_sign_names = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)',      
            2:'Speed limit (50km/h)',       
            3:'Speed limit (60km/h)',      
            4:'Speed limit (70km/h)',    
            5:'Speed limit (80km/h)',      
            6:'End of speed limit (80km/h)',     
            7:'Speed limit (100km/h)',    
            8:'Speed limit (120km/h)',     
           9:'No passing',   
           10:'No passing veh over 3.5 tons',     
           11:'Right-of-way at intersection',     
           12:'Priority road',    
           13:'Yield',     
           14:'Stop',       
           15:'No vehicles',       
           16:'Veh > 3.5 tons prohibited',       
           17:'No entry',       
           18:'General caution',     
           19:'Dangerous curve left',      
           20:'Dangerous curve right',   
           21:'Double curve',      
           22:'Bumpy road',     
           23:'Slippery road',       
           24:'Road narrows on the right',  
           25:'Road work',    
           26:'Traffic signals',      
           27:'Pedestrians',     
           28:'Children crossing',     
           29:'Bicycles crossing',       
           30:'Beware of ice/snow',
           31:'Wild animals crossing',      
           32:'End speed + passing limits',      
           33:'Turn right ahead',     
           34:'Turn left ahead',       
           35:'Ahead only',      
           36:'Go straight or right',      
           37:'Go straight or left',      
           38:'Keep right',     
           49:'Keep left',      
           40:'Roundabout mandatory',     
           41:'End of no passing',      
           42:'End no passing veh > 3.5 tons'
}



def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python traffic.py data_directory")

    data_dir = sys.argv[1]
    images, labels = load_data(data_dir)
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(np.array(images), np.array(labels), test_size=TEST_SIZE)

    datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    datagen.fit(x_train)

    model = get_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
    model.fit(datagen.flow(x_train, y_train, batch_size=32),
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping, model_checkpoint])

    model.evaluate(x_test, y_test, verbose=2)



def load_data(data_dir):
    """Load image data from directory `data_dir`.
    Assume `data_dir` has one directory named after each category, numbered 0 through NUM_CATEGORIES - 1.
    Inside each category directory will be some number of image files.
    Return tuple `(images, labels)`.
    `images` should be a list of all of the images in the data directory, where each image is formatted as a numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3.
    `labels` should be a list of integer labels, representing the categories for each of the corresponding `images`.
    """
    images = []
    labels = []

    # Loop through each category directory
    for category_idx in range(43):
        category_dir = os.path.join(data_dir, str(category_idx))

        # Loop through each image file in the category directory
        for file_name in os.listdir(category_dir):
            file_path = os.path.join(category_dir, file_name)

            # Load the image and preprocess it
            image = cv2.imread(file_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = image / 255.0  # Normalize pixel values to [0, 1]

            images.append(image)
            labels.append(category_idx)

    return images, labels


def get_model():
    """Returns a compiled convolutional neural network model.
    Assume that the `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def load_model(model_path):
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    main()


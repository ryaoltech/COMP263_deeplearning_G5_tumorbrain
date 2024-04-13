# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:32:29 2024

@author: Jbaru
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16
import seaborn as sns
import matplotlib.image as mpimg
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


class BrainTumorClassifier:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.tumor_dir = os.path.join(base_dir, 'yes')
        self.notumor_dir = os.path.join(base_dir, 'no')
        self.images = []
        self.labels = []

    def load_images(self, directory, label):
        for image_path in glob.glob(os.path.join(directory, "*")):
            image = mpimg.imread(image_path)
            # Ensure the image is in float32 format, and drop alpha channel if present
            if image.ndim == 3 and image.shape[2] == 4:
                image = image[:, :, :3]  # Keep only the first three channels
            if image.dtype != np.float32:
                image = (image / 255.0).astype(np.float32)  # Normalize and convert to float32
            if image is not None:
                self.images.append((image, label))
        print(f"Loaded {len(self.images)} images from {directory}")
    
    def preprocess_data(self):
        data = []
        labels = []
        for img, label in self.images:
            img = Image.fromarray((img * 255).astype('uint8'))  # Convert back to uint8 for PIL
            if img.mode != 'RGB':
                img = img.convert('RGB')  # Convert grayscale to RGB if needed
            img = img.resize((64, 64))
            img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize data
            data.append(img_array)
            labels.append(label)
            # Optional: Check image shape
            if img_array.shape != (64, 64, 3):
                print(f"Unexpected shape {img_array.shape}")
        return np.array(data), np.array(labels)

    def augment_data(self):
        self.datagen = ImageDataGenerator(
            rotation_range=10,       # Increased rotation range
            width_shift_range=0.1,   # Increased shift range
            height_shift_range=0.1,  # Increased shift range
            shear_range=0.1,         # Increased shear range
            zoom_range=0.1,          # Increased zoom range
            horizontal_flip=True,
            fill_mode='nearest'
        )


    def create_model(self, input_shape=(64, 64, 3), conv_layers_details=[], dense_layers_details=[]):
        model = Sequential()
        
        # Primero añadir el modelo base VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False
        model.add(base_model)
        
        # Añadir las capas convolucionales personalizadas encima del modelo base
        for filters, kernel_size, pool_size in conv_layers_details:
            model.add(Conv2D(filters, kernel_size, activation='relu', padding='same'))
            model.add(BatchNormalization())
            if pool_size:
                model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        
        # Aplanar antes de pasar a las capas densas
        model.add(Flatten())
    
        # Añadir las capas densas personalizadas
        for neurons, dropout_rate, l2_rate in dense_layers_details:
            model.add(Dense(neurons, activation='relu', kernel_regularizer=l2(l2_rate)))
            model.add(Dropout(dropout_rate))
    
        # Capa de salida
        model.add(Dense(1, activation='sigmoid'))
    
        # Compilar el modelo
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, class_weight):
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            verbose=1,
            min_delta=0.0001,
            min_lr=0.00001
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )
        history = model.fit(
            self.datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=30,
            validation_data=(X_val, y_val),
            validation_steps=len(X_val) // 32,
            callbacks=[reduce_lr, early_stopping],
            class_weight=class_weight  # Now this parameter is correctly expected
        )
        
        return history
    

    def plot_results(self, history):
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()
        
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.show()

    def evaluate_model(self, model, X_test, y_test):
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(f"Test Accuracy: {test_acc}")
        return test_loss, test_acc

    def predict(self, model, X_test):
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary output
        return y_pred

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

def run_experiment(classifier, conv_details, dense_details):
    model = classifier.create_model(
        input_shape=(64, 64, 3),
        conv_layers_details=conv_details,
        dense_layers_details=dense_details
    )
    history = classifier.train_model(model, X_train, y_train, X_val, y_val, class_weight_dict)
    classifier.plot_results(history)
    test_loss, test_acc = classifier.evaluate_model(model, X_val, y_val)
    y_pred = classifier.predict(model, X_val)
    classifier.plot_confusion_matrix(y_val, y_pred)
    print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")

    # Plotting the images and their predictions
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        for j in range(4):
            index = np.random.randint(0, len(X_val))
            axes[i, j].imshow(X_val[index].reshape(64, 64, 3))  # Corrected for RGB images
            axes[i, j].set_title('Actual: {} \n Predicted: {}'.format(y_val[index], y_pred[index]))
            axes[i, j].axis('off')
    plt.show()

    # Saving the model
    model.save('brain_tumor_model.h5')

# Ensure you define X_train, y_train, X_val, y_val, class_weight_dict appropriately before calling run_experiment



if __name__ == '__main__':
    # Path to the dataset
    directory_path = r"C:\Users\Jbaru\DeepLearning\Project\brain_tumor_dataset"

    # Instantiate the classifier
    classifier = BrainTumorClassifier(directory_path)
    
    # Load images
    classifier.load_images(classifier.tumor_dir, 1)  # 1 for tumor
    classifier.load_images(classifier.notumor_dir, 0)  # 0 for no tumor

    # Preprocess data
    data, labels = classifier.preprocess_data()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Compute class weights manually
    class_weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Initialize data augmentation
    classifier.augment_data()
    
    # Experimentation hyperparameters

    # Define various hyperparameters sets for experimentation
    hyperparameters_sets = [
        # Set 1: Basic setup
        {'conv_layers_details': [(32, (3, 3), (2, 2)), (64, (3, 3), (2, 2))],
         'dense_layers_details': [(128, 0.5, 0.01), (64, 0.3, 0.01)]},
    
        # Set 2: More filters and layers
        {'conv_layers_details': [(64, (3, 3), (2, 2)), (128, (3, 3), (2, 2)), (256, (3, 3), (2, 2))],
         'dense_layers_details': [(256, 0.4, 0.02), (128, 0.4, 0.02)]},
    
        # Set 3: Larger kernels and no pooling
        {'conv_layers_details': [(32, (5, 5), None), (64, (5, 5), None)],
         'dense_layers_details': [(516, 0.3, 0.01), (128, 0.4, 0.02)]},
        # Set 4: Your new configuration with no pooling
        {'conv_layers_details': [(32, (3, 3), None), (64, (3, 3), None)],  # No pooling is explicitly stated
         'dense_layers_details': [(516, 0.5, 0.01), (64, 0.3, 0.01)]},
    ]


    for hp_set in hyperparameters_sets:
        run_experiment(classifier, hp_set['conv_layers_details'], hp_set['dense_layers_details']) 





    

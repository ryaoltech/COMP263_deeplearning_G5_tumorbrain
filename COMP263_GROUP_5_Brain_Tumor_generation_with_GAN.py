# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 02:15:22 2024

@author: saman
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math 
import cv2
from tqdm import tqdm
import os 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, Flatten, Dropout

# Define the path to the directory containing the dataset
dir_path = './brain_tumor_dataset'
# List subdirectories within the main directory
dirs = [os.path.join(dir_path, d) for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

# Create a list of paths for a subset of images from each subdirectory
imshow_dirs = []
for d in dirs:
    step = os.listdir(d)
    for x in range(min(8, len(step))):
        try:
            imshow_dirs.append(os.path.join(d, step[x]))
        except:
            continue
# Create a grid of images for visualization
fig, ax = plt.subplots(4,4,figsize=(8,8))
for n in range(4):
    for m in range(4):
        path =imshow_dirs[m + 4*n]
        image = cv2.imread(str(path))
        ax[n,m].imshow(image)
        ax[n,m].grid(False)
        
# Read the shape of the first image
cv2.imread(imshow_dirs[0]).shape

# Initialize lists to store images and labels
X = []
Y = []
# Define directories for the two classes: "no" and "yes"
dirs = ['./brain_tumor_dataset/no','./brain_tumor_dataset/yes']
# Iterate over directories
for i in dirs:
    path = i
    # Iterate over images in each directory
    for x in os.listdir(path):
        # Read, resize, and convert images to grayscale
        img = cv2.imread(path + '/'+ x)
        img = cv2.resize(img,(128, 128), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(128,128,1)
        X.append(img)
        Y.append(i)
# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)
# Reshape and normalize images
X = (X.astype(np.float32) - 127.5)/127.5
X = X.reshape(253, 128*128)

#Building the generator model of the GAN
generator_model = Sequential([
    Dense(256, input_shape=(100,)),
    BatchNormalization(),
    LeakyReLU(0.2),
    Reshape((16, 16, 1)),
    Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    BatchNormalization(),
    LeakyReLU(0.2),
    Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
])
# Print summary of the generator model
generator_model.summary()

#sample untrained generator
sample_vector = tf.random.normal([1, 100])
sample_image = generator_model(sample_vector, training=False)
plt.imshow(sample_image[0, :, :, 0], cmap='gray')

#Building the discriminator model of the GAN
discriminator_model = Sequential([
    Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 1]),
    LeakyReLU(),
    Dropout(0.3),
    Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    LeakyReLU(),
    Dropout(0.3),
    Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
    BatchNormalization(),
    LeakyReLU(),
    Flatten(),
    Dense(1)
])
# Print summary of the discriminator model
discriminator_model.summary()

cross_entropy_ruben = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer_ruben = tf.keras.optimizers.Adam()
discriminator_optimizer_ruben = tf.keras.optimizers.Adam()

def training_step(images):
    noise = tf.random.normal([tf.shape(images)[0], 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model(noise, training=True)
        real_output = discriminator_model(images, training=True)
        fake_output = discriminator_model(generated_images, training=True)
        gen_loss = cross_entropy_ruben(tf.ones_like(fake_output), fake_output)
        real_loss = cross_entropy_ruben(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy_ruben(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss
    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
    generator_optimizer_ruben.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer_ruben.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))
    return gen_loss, disc_loss

import time

epochs = 3000
batch_size = 32
gen_loss_history = []
disc_loss_history = []

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the image generator
datagen = ImageDataGenerator(
rotation_range=20, # Random rotation range (in degrees)
width_shift_range=0.1, # Random horizontal shift range (as a fraction of total width)
height_shift_range=0.1, # Random vertical shift range (as a fraction of total height)
shear_range=0.2, # Random shear range
zoom_range=0.2, # Random zoom range
horizontal_flip=True, # Random horizontal flip
fill_mode='nearest' # Fill strategy for pixels outside the original image boundaries
)

# Reshape the data so that ImageDataGenerator can accept it
X = X.reshape(-1, 128, 128, 1)

# Create an iterator to generate augmented images
augmented_images = datagen.flow(X, batch_size=batch_size)

# Training with augmented images
for epoch in range(epochs):
    start_time = time.time()
    for i in range(0, len(X), batch_size):
        batch_images = next(augmented_images)
        gen_loss, disc_loss = training_step(batch_images)
        gen_loss_history.append(gen_loss)
        disc_loss_history.append(disc_loss)
    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start_time))
    
# Generating 16 sample vectors
sample_vectors = tf.random.normal(shape=(16, 100))

# Generating images from the generator
generated_images = generator_model(sample_vectors, training=False)

# Normalizing the pixels in the generated images
generated_images = (generated_images * 127.5) + 127.5  
    
# Plotting the generated images
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()

#plotting the loss history of the generator and discriminator
plt.plot(gen_loss_history, label='Generator Loss')
plt.plot(disc_loss_history, label='Discriminator Loss')
plt.legend()
plt.show()

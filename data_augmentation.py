import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import top_k_categorical_accuracy
from PIL import Image
import matplotlib.pyplot as plt


# Define paths
csv_path = r'D:\Nitin\pythonProject1moviegenre\cleaned_data.csv'
image_dir = r'D:\Nitin\pythonProject1moviegenre\posters'

# Load the dataset
df = pd.read_csv(csv_path)
imdbid = df["imdbId"]

y = df.drop(columns="imdbId").values  # Genres (multi-label) for each movie

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(imdbid, y, test_size=0.2, random_state=42)

# Create ImageDataGenerators for training and testing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Rescaling pixel values to [0, 1]
    rotation_range=40,  # Random rotations
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill any pixels that are shifted out of the image
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255  #rescale for validation set, no augmentation
)


#a generator function to read and process images and calculate sample weights
def image_gen(imdbid_list, labels, batch_size):
    while True:
        for i in range(0, len(imdbid_list), batch_size):
            batch_imdbid = imdbid_list[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            images = []
            sample_weights = []  # List to hold sample weights
            for imdbid, label in zip(batch_imdbid, batch_labels):
                img_path = os.path.join(image_dir, f'{imdbid}.jpg')
                img = Image.open(img_path).resize((224, 224))  # Resize to match input size for VGG16
                img = np.array(img)  # Convert image to numpy array

                # Ensure the image is 3 channels (RGB)
                if img.ndim == 2:  # Grayscale image (no color channels)
                    img = np.stack([img] * 3, axis=-1)  # Convert to RGB by duplicating the grayscale channel
                elif img.shape[2] == 4:  # If image has 4 channels (RGBA), convert to RGB
                    img = img[:, :, :3]

                images.append(img)

                # Calculate sample weight based on number of non-zero labels
                sample_weight = np.sum(label)  # Sum of 1s in the multi-label vector
                sample_weights.append(sample_weight)

            images = np.array(images)  # Ensure that images are stacked correctly
            sample_weights = np.array(sample_weights)
            yield images, np.array(batch_labels), sample_weights  # Yield sample weights as well


# Load pre-trained VGG16 model (without the top fully connected layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of the pre-trained model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Replace Flatten with Global Average Pooling
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(y.shape[1], activation='sigmoid')(x)  # Multi-label classification

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with top-3 accuracy metric
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', top_k_categorical_accuracy])  # Added top_k_categorical_accuracy

# Callbacks: Learning rate scheduler and early stopping
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 patience=3,
                                 verbose=1,
                                 factor=0.5,
                                 min_lr=0.00001)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model using the image generator for training data
history = model.fit(
    image_gen(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    epochs=30,
    validation_data=image_gen(x_test, y_test, batch_size=32),
    validation_steps=len(x_test) // 32,
    callbacks=[lr_reduction, early_stopping]
)

# Save the trained model
model.save('movie_genre_model_transfer_learning_with_weights.h5')  # Save the model after training

# Plot the training progress
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()

# Optionally, plot top-3 accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['top_k_categorical_accuracy'], label='Train Top-3 Accuracy')
plt.plot(history.history['val_top_k_categorical_accuracy'], label='Validation Top-3 Accuracy')
plt.title('Training and Validation Top-3 Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Top-3 Accuracy')
plt.legend()
plt.show()
# Evaluate the model on the test set and print top-3 accuracy
test_loss, test_accuracy, test_top3_accuracy = model.evaluate(image_gen(x_test, y_test, batch_size=32),
                                                              steps=len(x_test) // 32)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
print(f"Test top-3 accuracy: {test_top3_accuracy}")

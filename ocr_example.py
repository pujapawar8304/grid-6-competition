import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Preprocessing data (assuming labeled dataset of fresh and spoiled produce)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory('produce_images', 
                                         target_size=(224, 224),
                                         batch_size=32,
                                         class_mode='binary',
                                         subset='training')

val_data = datagen.flow_from_directory('produce_images', 
                                       target_size=(224, 224),
                                       batch_size=32,
                                       class_mode='binary',
                                       subset='validation')

# Build a simple CNN model for freshness detection
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (Fresh or Spoiled)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_data, epochs=10, validation_data=val_data)

# Save the model for later use
model.save('freshness_detection_model.h5')

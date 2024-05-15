import requests
import json
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical

# Load JSON data
with open('4252890.json') as f:
    data = json.load(f)

# Assuming UML is indicated in the description
is_UML = "UML" in data['metadata']['description']

# Use the "files" link to access the actual image files
files_link = data['links']['files']
files_response = requests.get(files_link)

if files_response.status_code == 200:
    files_data = files_response.json()
    file_entries = files_data.get('entries', [])

    # Extract image paths and labels
    image_paths = [entry['links']['self'] for entry in file_entries]
    labels = [1 if is_UML else 0] * len(image_paths)  # Assuming UML is indicated in the description

    # Load and preprocess images
    images = [img_to_array(Image.open(BytesIO(requests.get(path).content)).convert("RGB").resize((224, 224))) for path in image_paths]
    images = np.array(images) / 255.0  # Normalize pixel values

    # Convert labels to numpy array or use one-hot encoding
    labels = np.array(labels)
    num_classes = 2  # Assuming binary classification (UML, non-UML)
    labels = to_categorical(labels, num_classes)

    # Split data into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build a simple CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

    # Save the model
    model.save('uml_class_diagram_classifier.h5')
else:
    print(f"Error accessing files link: {files_response.status_code}")

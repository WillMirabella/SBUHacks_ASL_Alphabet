import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

NUM_EPOCHS = 3
BATCH_SIZE = 64

def load_data(data_dir):
    train_features, train_labels = load_dataset(os.path.join(data_dir, 'asl_alphabet_train'))
    test_features, test_labels = load_dataset(os.path.join(data_dir, 'asl_test'))
    return (train_features, train_labels), (test_features, test_labels)

def load_dataset(dataset_dir, target_size=(150, 150)):
    features = []
    labels = []
    label_encoder = LabelEncoder()
    
    # Get list of subdirectories (each subdirectory represents a class)
    classes = sorted(os.listdir(dataset_dir))
    
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):  # Skip non-directory files
            continue
        for image_name in os.listdir(class_dir):
            # Read and preprocess image
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            if image is None or image.size == 0:  # Skip empty or invalid images
                print(f"Error: Failed to load image: {image_path}")
                continue
            image = preprocess_image(image, target_size)  # Resize image
            
            # Append image to features list
            features.append(image)
            
            # Append label to labels list
            labels.append(class_name)  # Use class name as label
    
    # Encode labels
    labels = label_encoder.fit_transform(labels)
        
    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels


def preprocess_image(image, target_size):
    # Resize image to target size
    image = cv2.resize(image, target_size)
    
    # Convert image to grayscale (if needed)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to range [0, 1]
    image = image / 255.0
    
    # Add any other preprocessing steps here (e.g., data augmentation, noise removal, contrast enhancement)
    
    return image 




def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # Dropout layer with 50% dropout rate
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model




# Train the model
def train_model(model, train_data):
    train_features, train_labels = train_data
    
    # Train the model
    model.fit(train_features, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    return model


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Evaluate the trained model on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

def main():
    # Load data
    data_dir = "data/asl_train3/"
    (train_features, train_labels), (test_features, test_labels) = load_data(data_dir)
    
    # Check the size of the datasets
    print("Train samples:", len(train_features))
    print("Test samples:", len(test_features))
    
    # Build the model
    input_shape = input_shape = (*train_features[0].shape, 1)
    num_classes = len(np.unique(train_labels))
    model = build_model(input_shape, num_classes)
    
    trained_model = train_model(model, (train_features, train_labels))


    # Evaluate the model on the test set
    evaluate_model(trained_model, test_features, test_labels)

    trained_model.save("models/model1")

if __name__ == "__main__":
    main()


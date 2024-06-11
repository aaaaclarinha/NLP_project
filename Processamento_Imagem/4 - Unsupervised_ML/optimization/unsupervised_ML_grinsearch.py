import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import pandas as pd
from skimage.filters import sobel
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor

# Dataset path
train_path = "images/cats_dogs_light/train/*"
test_path = "images/cats_dogs_light/test/*"

# Resize images to
SIZE = 128

# Function to count the number of images
def count_images(path):
    count = 0
    for directory_path in glob.glob(path):
        count += len(glob.glob(os.path.join(directory_path, "*.jpg")))
    return count

# Function to load images and labels
def load_images_and_labels(path):
    num_images = count_images(path)
    images = np.zeros((num_images, SIZE, SIZE), dtype=np.uint8)
    labels = np.empty(num_images, dtype=object)
    
    idx = 0
    for directory_path in glob.glob(path):
        label = os.path.basename(directory_path)
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images[idx] = img
            labels[idx] = label
            idx += 1

    return images, labels

# Load training and test datasets
print("Loading training and test datasets...")
train_images, train_labels = load_images_and_labels(train_path)
test_images, test_labels = load_images_and_labels(test_path)
print("Datasets loaded.")

# Encode labels
le = preprocessing.LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
test_labels_encoded = le.transform(test_labels)

# Normalize pixel values
x_train, x_test = train_images / 255.0, test_images / 255.0
y_train, y_test = train_labels_encoded, test_labels_encoded

# Feature extractor function
def feature_extractor(images):
    num_images = images.shape[0]
    image_dataset = []

    def process_image(image_idx):
        img = images[image_idx, :, :]
        df = pd.DataFrame()

        # Pixel values
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values

        # Gabor filters
        num = 1
        for theta in range(2):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                lamda = np.pi/4
                gamma = 0.5
                gabor_label = f'Gabor{num}'
                kernel = cv2.getGaborKernel((9, 9), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel).reshape(-1)
                df[gabor_label] = fimg
                num += 1

        # Sobel filter
        edge_sobel = sobel(img).reshape(-1)
        df['Sobel'] = edge_sobel

        return df

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, range(num_images)))

    image_dataset = pd.concat(results, ignore_index=True)
    return image_dataset

print("Extracting features...")
train_features = feature_extractor(x_train)
test_features = feature_extractor(x_test)
print("Features extracted.")

# Reshape to a vector for Random Forest training
X_for_RF = train_features.values.reshape((x_train.shape[0], -1))
test_for_RF = test_features.values.reshape((x_test.shape[0], -1))

# GridSearch setup
param_grid = {
    'n_estimators': [50, 60 ,100, 150, 200, 250, 300],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80],
    'min_samples_split': [2, 4, 5, 6 , 8, 10, 20, 25],
    'min_samples_leaf': [1, 2, 4, 8, 10, 20, 40, 50]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

def run_grid_search():
    grid_search.fit(X_for_RF, y_train)
    best_rf = grid_search.best_estimator_
    test_prediction = best_rf.predict(test_for_RF)
    accuracy = metrics.accuracy_score(y_test, test_prediction)
    return best_rf, accuracy

print("Starting GridSearch for Random Forest hyperparameters...")
desired_accuracy = 0.8
best_model = None
current_accuracy = 0.0

# Calculate the total number of combinations in the grid search
total_combinations = np.prod([len(param_grid[key]) for key in param_grid])
num_combinations_done = 0

while current_accuracy < desired_accuracy and num_combinations_done < total_combinations:
    best_model, current_accuracy = run_grid_search()
    num_combinations_done += 1
    print(f"Current best accuracy: {current_accuracy}")

    # Write the current best accuracy and hyperparameters to a .txt file
    with open("best_model_params.txt", "w") as file:
        file.write(f"Current best accuracy: {current_accuracy}\n")
        file.write("Best model hyperparameters:\n")
        for param, value in best_model.get_params().items():
            file.write(f"{param}: {value}\n")

    # Stop if all combinations have been tried
    if grid_search.best_score_ >= desired_accuracy or num_combinations_done >= total_combinations:
        break

print(f"Best model parameters: {best_model.get_params()}")
print(f"Achieved accuracy: {current_accuracy}")
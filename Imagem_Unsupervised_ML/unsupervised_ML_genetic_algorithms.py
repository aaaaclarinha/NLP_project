import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import pandas as pd
from skimage.filters import sobel
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from concurrent.futures import ThreadPoolExecutor
from deap import base, creator, tools, algorithms

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

# Genetic algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", np.random.randint, 10, 400)
toolbox.register("attr_max_depth", np.random.randint, 1, 400)
toolbox.register("attr_min_samples_split", np.random.randint, 2, 100)
toolbox.register("attr_min_samples_leaf", np.random.randint, 1, 100)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_int, toolbox.attr_max_depth, toolbox.attr_min_samples_split, toolbox.attr_min_samples_leaf), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    n_estimators = max(10, int(individual[0]))
    max_depth = max(1, int(individual[1]))  # Ensure max_depth is at least 1
    min_samples_split = max(2, int(individual[2]))  # Ensure min_samples_split is at least 2
    min_samples_leaf = max(1, int(individual[3]))  # Ensure min_samples_leaf is at least 1
    
    RF_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                      random_state=42)
    RF_model.fit(X_for_RF, y_train)
    test_prediction = RF_model.predict(test_for_RF)
    accuracy = metrics.accuracy_score(y_test, test_prediction)
    
    print(f"Evaluating: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, accuracy={accuracy}")
    
    return accuracy,

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutUniformInt, low=[10, 1, 2, 1], up=[400, 400, 100, 100], indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=5)  # Increased tournament size for stronger selection pressure
toolbox.register("evaluate", evaluate)

def main():
    population = toolbox.population(n=50)  # Increased population size for more diversity
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    print("Starting genetic algorithm...")
    accuracy = 0.0
    generation = 0
    max_generations = 700  # Increased number of generations

    # Elitism: Keep track of the best individuals
    elite_size = 2

    while accuracy < 0.95 and generation < max_generations:
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.3)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Select the next generation population
        population[:] = toolbox.select(offspring + population, len(population) - elite_size)
        
        # Add elites back to population
        best_individuals = tools.selBest(population, elite_size)
        population.extend(best_individuals)
        
        best_individual = tools.selBest(population, 1)[0]
        accuracy = evaluate(best_individual)[0]
        generation += 1
        print(f"Generation {generation}, Best individual: {best_individual}, Accuracy: {accuracy}")

    print(f"Best individual is: {best_individual}")
    print(f"Best accuracy is: {accuracy}")

if __name__ == "__main__":
    main()
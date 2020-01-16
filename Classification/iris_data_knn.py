import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


file_name = '/Users/ashutosh/Downloads/iris_data.csv'


def load_iris_data(file):
    return pd.read_csv(file)


def plot_hist_graph(data):
    data.hist(bins=50)
    plt.figure(figsize=(15, 10))
    plt.show()


def plot_parallel_coordinates(data, attr):
    plt.figure(figsize=(15, 10))
    parallel_coordinates(data[attr], "Species")
    plt.title('Iris Parallel Coordinates Plot', fontsize=20, fontweight='bold')
    plt.xlabel('Attributes', fontsize=15)
    plt.ylabel('Values', fontsize=15)
    plt.legend(loc=1, prop={'size': 15}, frameon=True, facecolor="white", edgecolor="black")
    plt.show()


def set_label_encoding(data_species):
    le = LabelEncoder()
    return le.fit_transform(data_species)


def test_train_data_split(data, data_species, test_ratio, state):
    return train_test_split(data, data_species, test_size=0.33, random_state=42)


def get_knn_classifier(k, x_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=k)
    return classifier.fit(x_train, y_train)


# Main Program starts from here
dataset = load_iris_data(file_name)

print(len(dataset))

# Only shows First 5 Lines
print(dataset.head())

# Print the info
print('---------- Info -------------')
print(dataset.info())
print('---------- Info Ends Here -------------')

# Describe dataset
print("\n----- Describe ------\n")
print(dataset.describe())
print('-------------- Describe Ends Here ----------')

# Results
print('\n----- Unique Species ------\n')
print(dataset['Species'].unique())

# Handling missing values
attributes = ['Sepal_Length(CM)', 'Sepal_Width(CM)', 'Petal_Length (CM)', 'Petal_Width (CM)', 'Species']

features = ['Sepal_Length(CM)', 'Sepal_Width(CM)', 'Petal_Length (CM)', 'Petal_Width (CM)']

data_values = dataset[features].values

plot_hist_graph(dataset)
plot_parallel_coordinates(dataset, attributes)

"""
Labels are categorical. KNN algorithm does not accept string labels. We need to use LabelEncoder to transform them into 
numbers. Iris-setosa correspond to 0, Iris-versicolor correspond to 1 and Iris-virginica correspond to 2
"""
feature_values = set_label_encoding(dataset['Species'].values)

x_train_set, x_test_set, y_train_set, y_test_set = test_train_data_split(data_values, feature_values, 0.2, 0)

# KNN Classification
# K = 3
knn_classifier = get_knn_classifier(3, x_train_set, y_train_set)

# Predicting the test result
prediction = knn_classifier.predict(x_test_set)

print('--- Prediction ---')
print(prediction)

# Confusion Matrix
c_matrix = confusion_matrix(y_test_set, prediction)

print(c_matrix)

accuracy = accuracy_score(y_test_set, prediction) * 100

print(accuracy)

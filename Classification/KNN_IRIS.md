**What is KNN Algorithm?**

Equally known as **K-Nearest Neighbour**, is one of the most common algorithms in Machine Learning and is broadly used in regression and classification problems.



This article assumes you have some familiarity with **supervised learning,** 
if not then please visit [here](https://ashutosh.dev/blog/post/2019/12/what-is-machine-learning). 



To be more precise, KNN falls under Instance-based learning. Consequently,
there is one more key question to be asked: "What is **Instance-based learning**"?



Instance-based learning or lazy learning or memory-based learning or by heart learning is one of the most common algorithms used in Machine Learning.
In Instance-based learning, the system learns from the models and promptly
using the similarity pattern, positively identifies the possible solution for the
new data set.



Let's get back our focus on KNN.



KNN uses similarity to predict the result of new data points. It indicates the
 data will be assigned a value based on how closely it relates the points
in the training set.





It's ok if you don't get the complete understanding of KNN, we'll understand
it more with the help of an iris dataset. Iris data is available [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data). It's accessed
 several times by the Machine Learning beginners and enthusiasts.





**Implementation of KNN (Python)**



I am using Pycharm to write the code but can use Jupyter too.



```python
import numpy as np
import pandas as pd

file_name = '/Users/ashutosh/Downloads/iris_data.csv'

dataset = pd.read_csv(file_name)
print(len(dataset))

# Only shows First 5 Lines
print(dataset.head())
```



Execute the above program and you will get the following output


```
> 150
>   Sepal_Length(CM) Sepal_Width(CM) ... Petal_Width (CM)   Species
> 0        5.1       3.5 ...        0.2 Iris-setosa
> 1        4.9       3.0 ...        0.2 Iris-setosa
> 2        4.7       3.2 ...        0.2 Iris-setosa
> 3        4.6       3.1 ...        0.2 Iris-setosa
> 4        5.0       3.6 ...        0.2 Iris-setosa
> [5 rows x 5 columns]
```



To print the info



```
# Print the info
print('---------- Info -------------')
print(dataset.info())
print('---------- Info Ends Here -------------')
```



Executing the above code will give the following output:
```

> ---------- Info -------------
> <class 'pandas.core.frame.DataFrame'>
> RangeIndex: **150 entries**, 0 to 149
> Data columns (total 5 columns):
> Sepal_Length(CM)   150 non-null float64
> Sepal_Width(CM)   150 non-null float64
> Petal_Length (CM)  150 non-null float64
> Petal_Width (CM)   150 non-null float64
> Species       150 non-null object
> dtypes: float64(4), object(1)
> memory usage: 6.0+ KB
> None
> --------------- Info Ends Here ----------
```

In the iris database, we have 150 entries and the index starts with 0.

According to the Python documentation, **describe()** function in pandas
generate statistics that summarize the central tendency, dispersion and
shape of a dataset's distribution, excluding ``NaN`` values. Analyzes both
numeric and object series, as well as ``DataFrame`` column sets of mixed
data types. The output will vary depending on what is provided.

```python
# Describe dataset
print("\n----- Describe ------\n")
print(dataset.describe())
print('-------------- Describe Ends Here ----------')

-- Output --

----- Describe ------
```

> ​    Sepal_Length(CM) Sepal_Width(CM) Petal_Length (CM) Petal_Width (CM)
> count    150.000000    150.000000     150.000000    150.000000
> mean      5.843333     3.054000      3.758667     1.198667
> std      0.828066     0.433594      1.764420     0.763161
> min      4.300000     2.000000      1.000000     0.100000
> 25%      5.100000     2.800000      1.600000     0.300000
> 50%      5.800000     3.000000      4.350000     1.300000
> 75%      6.400000     3.300000      5.100000     1.800000
> max      7.900000     4.400000      6.900000     2.500000
> -------------- Describe Ends Here ----------

In order to check the unique species in the dataset

```python
print(dataset['Species'].unique())
```

After executing the command, if you receive the following output 
*['Iris-setosa' 'Iris-versicolor' 'Iris-virginica'],* you are on the right track.

Next step is to import the following functions from the **sklearn** library.

import matplotlib.pyplot as plt

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
attributes = ['Sepal_Length(CM)', 'Sepal_Width(CM)', 
    'Petal_Length (CM)', 'Petal_Width (CM)', 'Species'
]
features = ['Sepal_Length(CM)', 'Sepal_Width(CM)', 
    'Petal_Length (CM)', 'Petal_Width (CM)'
]
```

"attributes' is the python list, consist of all the headers in the CSV file.
If there are **no headers in the CSV file**. Please add it.
"features" is the python list consist of Iris parameters.

```python
def plot_hist_graph(data):
    data.hist(bins=50)
    plt.figure(figsize=(15, 10))
    plt.show()


def plot_parallel_coordinates(data, attr):
    plt.figure(figsize=(15, 10))
    parallel_coordinates(data[attr], "Species")
    plt.title(
        'Iris Parallel Coordinates Plot', 
        fontsize=20, fontweight='bold'
    )
    plt.xlabel('Attributes', fontsize=15)
    plt.ylabel('Values', fontsize=15)
    plt.legend(
        loc=1, 
        prop={'size': 15}, 
        frameon=True, 
        facecolor="white", 
        edgecolor="black")
    plt.show()
data_values = dataset[features].values

plot_hist_graph(dataset)
```


However we want to start with the implementation of the KNN algorithm
but there is one hinderance, KNN does not validate or allow *string* labels.
Hence we need to convert string into integer labels.

We remember, we only have three unique species in the dataset, so we
can easily labelled them as "0", "1" and "2". To set labels we have
LabelEncoder() from *sklearn* library**.** Here is the implementation.

```
def set_label_encoding(data_species):
    le = LabelEncoder()
    return le.fit_transform(data_species)
feature_values = set_label_encoding(dataset['Species'].values)
```

Once the data is labelled, Now it's time to implement the KNN algorithm. 

```python
def test_train_data_split(data, data_species, test_ratio, state):
    return train_test_split(
        data, data_species, test_size=0.33, random_state=42
    )
def get_knn_classifier(k, x_train, y_train):
    classifier = KNeighborsClassifier(n_neighbors=k)
    return classifier.fit(x_train, y_train)
# Train the dataset
x_train_set, x_test_set, y_train_set, y_test_set = 
    test_train_data_split(
        data_values, feature_values, 0.2, 0
    )
# KNN Classification
# K = 3
knn_classifier = get_knn_classifier(3, x_train_set, y_train_set)

# Predicting the test result
prediction = knn_classifier.predict(x_test_set)

print('--- Prediction ---')
print(prediction)
To check the model accuracy, we need to build the confusion matrix.# Confusion Matrix
c_matrix = confusion_matrix(y_test_set, prediction)

print(c_matrix)

accuracy = accuracy_score(y_test_set, prediction) * 100

print(accuracy)
```

Execute the above program. By implementing the above, we get the accuracy of about 96.67 %. 

Important link I followed:

Iris dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

KNN Algorithm: https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

KNN Algo Introduction: [https://www.analyticsvidhya.com/blog/2018/03/
introduction-k-neighbours-algorithm-clustering/](

# data science essentials
import pandas as pd
import numpy as np

# sklearn - for machine learning tasks
import sklearn
from sklearn import linear_model, preprocessing, datasets, svm
from sklearn.neighbors import KNeighborsClassifier

# Tensorflow - to build neural networks
import tensorflow as tf
from tensorflow import keras

# matplotlib - for the plots
import matplotlib.pyplot as plt
from matplotlib import style

# libraries to save/load models to the system
import pickle
import os

def linear_showcase():
    # https://archive.ics.uci.edu/ml/datasets/Student+Performance
    # load the data into the workspace
    data = pd.read_csv("data/student-mat.csv", sep=";")

    # print(data.head()) # print the first 5 rows to evaluate the data structure if needed

    # select only the relevant columns/variables
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    # isolate the target variable (label)
    y = np.array(data["G3"])

    # isolate the features
    x = np.array(data.drop("G3", axis=1))

    # store the highest accuracy score in a variable to iterate and find the best fitting model
    high_score = 0

    # pickle is used to save/load machine learning models, but it does not work well with relative paths
    # as such, a variable is created to store the absolute path to where the models are to be saved
    here = os.path.dirname(os.path.abspath(__file__))
    print(here)

    # loop to find the best model
    print("Creating model...")
    for _ in range(50):
        # train-test split (90% training, 10% testing)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

        # create the model
        model = linear_model.LinearRegression()

        # fit the data to the model
        model.fit(x_train, y_train)

        # evaluate the model's accuracy
        acc = model.score(x_test, y_test)

        # save the model if the accuracy is higher than the previous highest
        if acc > high_score:
            high_score = acc
            print("Saving model...")
            with open(os.path.join(here, "models/linear_model.pickle"), "wb") as f:
                pickle.dump(model, f)
        

    # load the best performing model
    print("Loading best performing model...")
    pickle_in = open(os.path.join(here, "models/linear_model.pickle"), "rb")
    model = pickle.load(pickle_in)
    print(f"Accuracy: {model.score(x_test, y_test)}")

    # evaluate the linear model itself
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Note that there are {len(model.coef_)} dimensions since there were {len(model.coef_)} features") 

    # evaluate the model's predictions
    y_hat = model.predict(x_test)
    print("Predictions:")
    for i in range(len(y_hat)):
        print(f"Predicted grade: {round(y_hat[i])}, Actual grade: {y_test[i]}")
    
    # plot
    x_label = "G1"
    style.use("ggplot")
    plt.scatter(data[x_label], data["G3"])
    plt.xlabel(x_label)
    plt.ylabel("Final Grade")
    plt.show()
    plt.savefig(here + "/imgs/linear_model/scatterplot.png")

def knn_vs_svm_1():
    # https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    # load the data into the workspace
    column_names = ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
    data = pd.read_csv("data/car.data", names = column_names)

    # preprocessing is required to transform strings to float values
    # at the same time, features and labels are separated
    features = {}
    for i in range(len(column_names)):
        if i != len(column_names) - 1:
            features[i] = preprocessing.LabelEncoder().fit_transform(list(data[column_names[i]]))
        else:
            y = list(preprocessing.LabelEncoder().fit_transform(list(data[column_names[i]])))

    # features are taken from the dictionary and flattened to use with ML algorithms
    x = list(zip(features[0], features[1], features[2], features[3], features[4], features[5]))

    # store the highest accuracy score in a variable to iterate and find the best fitting model
    high_score = 0

    # pickle is used to save/load machine learning models, but it does not work well with relative paths
    # as such, a variable is created to store the absolute path to where the models are to be saved
    here = os.path.dirname(os.path.abspath(__file__))
    print(here)

    # loop to find the best model
    print("Creating KNN model...")
    for _ in range(5):
        # train-test split (90% training, 10% testing)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
        
        # optimizing for neighbors
        for k in range(4):
            # create the model
            model = KNeighborsClassifier(n_neighbors = 3*k + 1)

            # fit the data to the model
            model.fit(x_train, y_train)

            # evaluate the model's accuracy
            acc = model.score(x_test, y_test)

            # save the model if the accuracy is higher than the previous highest
            if acc > high_score["knn"]:
                high_score["knn"] = acc
                print("Saving model...")
                with open(os.path.join(here, "models/car_knn_model.pickle"), "wb") as f:
                    pickle.dump(model, f)
    
    print("Creating SVM model...")
    # loop to find the best model
    for _ in range(20):
        # train-test split (90% training, 10% testing)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
        
        # optimizing for neighbors
        for k in range(5):
            # create the model
            model = svm.SVC(kernel="linear")

            # fit the data to the model
            model.fit(x_train, y_train)

            # evaluate the model's accuracy
            acc = model.score(x_test, y_test)

            # save the model if the accuracy is higher than the previous highest
            if acc > high_score["svm"]:
                high_score["svm"] = acc
                print("Saving model...")
                with open(os.path.join(here, "models/car_svm_model.pickle"), "wb") as f:
                    pickle.dump(model, f)   

    # load the best performing model
    print("Loading best performing models...")
    pickle_in = open(os.path.join(here, "models/car_knn_model.pickle"), "rb")
    knn_model = pickle.load(pickle_in)
    pickle_in = open(os.path.join(here, "models/car_svm_model.pickle"), "rb")
    svm_model = pickle.load(pickle_in)
    print(f"KNN Accuracy: {knn_model.score(x_test, y_test)}")
    print(f"SVM Accuracy: {svm_model.score(x_test, y_test)}")

def knn_vs_svm_2():
    # load the data into the workspace
    cancer = datasets.load_breast_cancer()
    x = cancer.data
    y = cancer.target

    # store the highest accuracy score in a variable to iterate and find the best fitting model
    high_score = 0

    # pickle is used to save/load machine learning models, but it does not work well with relative paths
    # as such, a variable is created to store the absolute path to where the models are to be saved
    here = os.path.dirname(os.path.abspath(__file__))
    print(here)

    # loop to find the best model
    print("Creating KNN model...")
    for _ in range(5):
        # train-test split (90% training, 10% testing)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
        
        # optimizing for neighbors
        for k in range(4):
            # create the model
            model = KNeighborsClassifier(n_neighbors = 3*k + 1)

            # fit the data to the model
            model.fit(x_train, y_train)

            # evaluate the model's accuracy
            acc = model.score(x_test, y_test)

            # save the model if the accuracy is higher than the previous highest
            if acc > high_score["knn"]:
                high_score["knn"] = acc
                print("Saving model...")
                with open(os.path.join(here, "models/cancer_knn_model.pickle"), "wb") as f:
                    pickle.dump(model, f)

    # loop to find the best model
    print("Creating SVM model...")
    for _ in range(20):
        # train-test split (90% training, 10% testing)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
        
        # optimizing for neighbors
        for k in range(5):
            # create the model
            model = svm.SVC(kernel="linear")

            # fit the data to the model
            model.fit(x_train, y_train)

            # evaluate the model's accuracy
            acc = model.score(x_test, y_test)

            # save the model if the accuracy is higher than the previous highest
            if acc > high_score["svm"]:
                high_score["svm"] = acc
                print("Saving model...")
                with open(os.path.join(here, "models/cancer_svm_model.pickle"), "wb") as f:
                    pickle.dump(model, f)
    
    # load the best performing model
    print("Loading best performing models...")
    pickle_in = open(os.path.join(here, "models/cancer_knn_model.pickle"), "rb")
    knn_model = pickle.load(pickle_in)
    pickle_in = open(os.path.join(here, "models/cancer_svm_model.pickle"), "rb")
    svm_model = pickle.load(pickle_in)
    print(f"KNN Accuracy: {knn_model.score(x_test, y_test)}")
    print(f"SVM Accuracy: {svm_model.score(x_test, y_test)}")    

def neural_net():
    # https://keras.io/api/datasets/fashion_mnist/
    # load the data into the workspace
    data = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # data is regularized to be between 0 and 1
    train_images = train_images/255.0
    test_images = test_images/255.0

    # store the highest accuracy score in a variable to iterate and find the best fitting model
    high_score = 0

    # pickle is used to save/load machine learning models, but it does not work well with relative paths
    # as such, a variable is created to store the absolute path to where the models are to be saved
    here = os.path.dirname(os.path.abspath(__file__))
    print(here)

    """
    # loop to find the best model
    for _ in range(3):
        # the model is a sequence of layers
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)), # ml works better with flattened arrays/tensors
            keras.layers.Dense(128, activation="relu"), # middle hidden layer adds complexity (extra weights, bias...) in the hope of id patterns
            keras.layers.Dense(10, activation="softmax") # softmax means 1 to the most likely neuron, 0 otherwise
	    ])

        # optimizer and loss function are kind of arbitrary, the metrics determines which value to work around to test the model and optimize/minimize loss
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # optimize for epochs
        for i in range(3):
            # the model is trained
            model.fit(train_images, train_labels, epochs=i*6) # epochs is the number of times the model is trained on the data (BEWARE OVERFITTING)
            loss, acc = model.evaluate(test_images, test_labels)
            # save the model if the accuracy is higher than the previous highest
            if acc > high_score:
                high_score = acc
                print("Saving model...")
                with open(os.path.join(here, "models/neural_net.pickle"), "wb") as f:
                    pickle.dump(model, f)"""
        
    # load the best performing model
    print("Loading best performing model...")
    pickle_in = open(os.path.join(here, "models/neural_net.pickle"), "rb")
    model = pickle.load(pickle_in)
    loss, acc = model.evaluate(test_images, test_labels)

    # y_hat (predictions) are stored within a variable to then evaluate against real values/labels
    predictions = model.predict(test_images)

    # plot of the first predictions
    style.use("ggplot")
    plt.figure(figsize=(5,5))
    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[test_labels[i]])
        plt.title(class_names[np.argmax(predictions[i])])
        plt.show()
        plt.savefig(f"{here}/imgs/neural_net/{i}.png")

def main():
    # linear_showcase()
    # knn_vs_svm_1()
    # knn_vs_svm_2()
    neural_net()

if __name__ == '__main__':
    main()
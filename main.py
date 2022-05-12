import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import os

def linear_test():
    # load the data into the workspace
    data = pd.read_csv("data/student-mat.csv", sep=";")

    # print(data.head()) # print the first 5 rows to evaluate the data structure if needed

    # select only the relevant columns/variables
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

    # isolate the target variable (label)
    y = np.array(data["G3"])

    # isolate the features
    x = np.array(data.drop("G3", axis=1))

    # store the highest accuracy score in a variable
    high_score = 0

    # pickle does not work well with relative paths, so before saving/loading the model a variable is created to store the pickle path
    here = os.path.dirname(os.path.abspath(__file__))
    print(here)

    # loop to find the best model
    for _ in range(50):
        # train-test split (90% training, 10% testing)
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        # create the model
        print("Creating model...")
        linear = linear_model.LinearRegression()

        # fit the data to the model
        linear.fit(x_train, y_train)

        # evaluate the model's accuracy
        acc = linear.score(x_test, y_test)

        # save the model if the accuracy is higher than the previous highest
        if acc > high_score:
            print("Saving model...")
            with open(os.path.join(here, "models/linear_model.pickle"), "wb") as f:
                pickle.dump(linear, f)
            high_score = acc
        

    # load the best performing model
    print("Loading best performing model...")
    pickle_in = open(os.path.join(here, "models/linear_model.pickle"), "rb")
    linear = pickle.load(pickle_in)
    print(f"Accuracy: {linear.score(x_test, y_test)}")

    # evaluate the linear model itself
    print(f"Coefficients: {linear.coef_}")
    print(f"Intercept: {linear.intercept_}")
    print(f"Note that there are {len(linear.coef_)} dimensions since there were {len(linear.coef_)} features") 

    # evaluate the model's predictions
    y_hat = linear.predict(x_test)
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

def neural_net():
    # https://keras.io/api/datasets/fashion_mnist/
    data = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # data is regularized to be between 0 and 1
    train_images = train_images/255.0
    test_images = test_images/255.0

    # pickle does not work well with relative paths, so before saving/loading the model a variable is created to store the pickle path
    here = os.path.dirname(os.path.abspath(__file__))
    print(here)

    # store the highest accuracy score in a variable
    high_score = 0
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
                print("Saving model...")
                with open(os.path.join(here, "models/neural_net.pickle"), "wb") as f:
                    pickle.dump(model, f)
                high_score = acc"""
        
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
    neural_net()

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import collections

train_1 = pd.read_csv('Assig1-Dataset/train_1.csv')
valid_1 = pd.read_csv('Assig1-Dataset/val_1.csv')
test_labels_1 = pd.read_csv('Assig1-Dataset/test_with_label_1.csv')
test_no_labels_1 = pd.read_csv('Assig1-Dataset/test_no_label_1.csv')

train_2 = pd.read_csv('Assig1-Dataset/train_2.csv')
valid_2 = pd.read_csv('Assig1-Dataset/val_2.csv')
test_labels_2 = pd.read_csv('Assig1-Dataset/test_with_label_2.csv')
test_no_labels_2 = pd.read_csv('Assig1-Dataset/test_no_label_2.csv')

info1 = pd.read_csv('Assig1-Dataset/info_1.csv')
info2 = pd.read_csv('Assig1-Dataset/info_2.csv')


def unpack(data):
    X = data.values[:,:-1]
    Y = data.values[:,-1:].ravel()
    return X,Y

def Score(y_test,y_predict):
    return sklearn.metrics.accuracy_score(y_test, y_predict)

def Save(filename,file,mode = 'w'):
    file.to_csv("output file/" + filename + ".csv", mode = mode)

def confusionmatrix(DS, model, output_file):
    X, y = unpack(DS)
    y_predict = model.predict(X)
    plot_confusion_matrix(model, X, y)
    plt.savefig('output file/' + output_file + '.png')
    report = classification_report(y, y_predict, output_dict = True, zero_division = 1)
    df = pd.DataFrame(report).transpose()
    return df

def prediction(model, test, filename):
    X = test.values[:, :]
    y_predict = model.predict(X)
    df = pd.DataFrame(y_predict)
    Save(filename, df)

def PER():
    X,y = unpack(train_1)
    model1 = Perceptron().fit(X, y)

    X_valid, y_valid = unpack(valid_1)
    y_predict = model1.predict(X_valid)
    score = Score(y_valid, y_predict)
    print(f'PER DS1 Score: {score}')

    prediction(model1, test_no_labels_1, "PER-DS1")

    df = confusionmatrix(test_labels_1, model1, "PER-DS1")
    Save("PER-DS1", df, mode = 'a')


    X, y = unpack(train_2)
    model2 = Perceptron().fit(X, y)

    X_valid, y_valid = unpack(valid_2)
    y_predict = model2.predict(X_valid)
    score = Score(y_valid, y_predict)
    print(f'PER DS2 Score: {score}')

    prediction(model1, test_no_labels_2, "PER-DS2")

    df = confusionmatrix(test_labels_2, model2, "PER-DS2")
    Save("PER-DS2", df, mode = 'a')

def Base_MLP():
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), activation = 'logistic', solver = 'sgd', max_iter=5000)

    X, y = unpack(train_1)
    model1 = mlp.fit(X, y)

    X_valid, y_valid = unpack(valid_1)
    y_predict = model1.predict(X_valid)
    score = Score(y_valid, y_predict)
    print(f'Base-MLP DS1 Score: {score}')

    prediction(model1, test_no_labels_1, "Base-MLP-DS1")

    df = confusionmatrix(test_labels_1, model1, "Base-MLP-DS1")
    Save("Base-MLP-DS1", df, mode='a')

    X, y = unpack(train_2)
    model2 = mlp.fit(X, y)

    X_valid, y_valid = unpack(valid_2)
    y_predict = model2.predict(X_valid)
    score = Score(y_valid, y_predict)
    print(f'Base-MLP DS2 Score: {score}')

    prediction(model2, test_no_labels_2, "Base-MLP-DS2")

    df = confusionmatrix(test_labels_2, model2, "Base-MLP-DS2")
    Save("Base-MLP-DS2", df, mode='a')

def Best_MLP():
    mlp = MLPClassifier(max_iter=200)
    parameter_space = {
        'hidden_layer_sizes': [(10, 10, 50), (30, 50)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
    }
    X, y = unpack(train_1)
    model1 = GridSearchCV(mlp, parameter_space, n_jobs=-1)
    model1.fit(X,y)
    print(model1.best_params_)

    X_valid, y_valid = unpack(valid_1)
    y_predict = model1.predict(X_valid)
    score = Score(y_valid, y_predict)
    print(f'Best-MLP DS1 Score: {score}')

    prediction(model1, test_no_labels_1, "Best-MLP-DS1")

    df = confusionmatrix(test_labels_1, model1, "Best-MLP-DS1")
    Save("Best-MLP-DS1", df, mode='a')

    X, y = unpack(train_2)
    model2 = GridSearchCV(mlp, parameter_space, n_jobs=-1)
    model2.fit(X, y)
    print(model2.best_params_)

    X_valid, y_valid = unpack(valid_2)
    y_predict = model2.predict(X_valid)
    score = Score(y_valid, y_predict)
    print(f'Best-MLP DS2 Score: {score}')

    prediction(model2, test_no_labels_2, "Best-MLP-DS2")

    df = confusionmatrix(test_labels_2, model2, "Best-MLP-DS2")
    Save("Best-MLP-DS2", df, mode='a')

#PER()
#Base_MLP()
#Best_MLP()


def distribution_plot(dataset, filename):
    X,y = unpack(dataset)
    counter = collections.Counter(y)
    labels = list(counter.keys())
    frequency = list(counter.values())
    plt.pie(frequency, labels = labels)
    plt.title(filename)
    plt.savefig('distribution plot/' + filename + '.png')


#distribution_plot(valid_2,"validation2")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.svm             # For SVC class
import sklearn.preprocessing   # For scale function
import sklearn.metrics         # for accuracy_score

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
    Y = data.values[:,-1:]
    return X,Y


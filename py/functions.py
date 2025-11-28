import kagglehub #used to get the data from kaggle.com
from kagglehub import KaggleDatasetAdapter #used to fetch the specific dataset from kaggle
import pandas as pd #used for viewing and manipulating the data
import matplotlib.pyplot as plt #used for data visulisation
import sklearn as skl
from sklearn.model_selection import train_test_split as skl_tts
from sklearn.tree import DecisionTreeClassifier as skl_dtc
from sklearn.metrics import classification_report as skl_cr
from sklearn.metrics import accuracy_score as skl_as
from sklearn.tree import plot_tree as plt_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#all helper functions used below
def dataGrabber(): #fetches data from kaggle.com and returns it
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "bryancimo/sdss-galaxy-classification-dr18", "sdss_100k_galaxy_form_burst.csv", pandas_kwargs={'skiprows': 1})
    return df

def dataSplitting(features, target, testSize, randomState): #splits the data and training set
    xTrain, xTest, yTrain, yTest = skl_tts(features, target, test_size = testSize, random_state = randomState)
    return xTrain, xTest, yTrain, yTest

def complexityTreeController(xTrain, yTrain, max_depth, random_state):
    clf = skl_dtc(max_depth, random_state)
    clf.fit(xTrain, yTrain)
    return clf

def modelEvaluation(clf, xTest, yTest):
    yPred = clf.predict(xTest)
    accuracy = skl_as(yTest, yPred)
    print(f"Accuracy of the model: {accuracy:.3f}")
    classReport = skl_cr(yTest, yPred)
    print("\nClassification Report:")
    print(classReport)
    return yPred

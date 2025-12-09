import kagglehub #used to get the data from kaggle.com
from kagglehub import KaggleDatasetAdapter #used to fetch the specific dataset from kaggle
import pandas as pd #used for viewing and manipulating the data
import matplotlib.pyplot as plt #used for data visulisation
from sklearn.model_selection import train_test_split as skl_tts #used to split the training and testing data
from sklearn.tree import DecisionTreeClassifier as skl_dtc #used to do the decision tree
from sklearn.metrics import classification_report as skl_cr #used to create a classification report to analyse the model
from sklearn.metrics import accuracy_score as skl_as #used to determine the accuracy of the model
from sklearn.tree import plot_tree as plt_tree #used to plot the decision tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #used to display the confusion matrix to compare the test data to the true values

#all helper functions used below
def dataGrabber(): #fetches data from kaggle.com and returns it
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "diraf0/sloan-digital-sky-survey-dr18", "SDSS_DR18.csv")
    return df

def dataSplitting(features, target, testSize, randomState): #splits the data and training set
    xTrain, xTest, yTrain, yTest = skl_tts(features, target, test_size = testSize, random_state = randomState)
    return xTrain, xTest, yTrain, yTest

def decisionTreeController(xTrain, yTrain, depth, state): #performs the decision tree classifier
    clf = skl_dtc(max_depth = depth, random_state = state)
    clf.fit(xTrain, yTrain)
    return clf

def modelEvaluation(clf, xTest, yTest): #evaluates the model used
    yPred = clf.predict(xTest)
    accuracy = skl_as(yTest, yPred)
    print(f"Accuracy of the model: {accuracy:.3f}")
    classReport = skl_cr(yTest, yPred)
    print("\nClassification Report:")
    print(classReport)
    return yPred

def printParameters(clf): #prints the parameters of the model used
    n_nodes = clf.get_depth()
    n_leaves = clf.get_n_leaves()
    model_params = clf.get_params()
    model_params_df_dt_clf = pd.DataFrame([model_params]).T
    print(f"Nodes: {n_nodes}")
    print(f"Leaves: {n_leaves}") 
    print(f"Model Parameters:")
    display(model_params_df_dt_clf)
    print("\n")
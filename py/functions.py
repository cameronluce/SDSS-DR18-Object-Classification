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

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#functions used in notebook 1
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

#functions used in notebook 2
class NeuralNetworkClassifier(nn.Module):
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        z = self.relu(self.fc2(z))
        z = self.fc3(z)
        return z

def modelTraining(model, trainLoader, criteria, optimiser, epochs):
    model.train()
    lossHistory = []
    
    for i in range(epochs):
        runningLoss = 0.0
        for inputs, labels in trainLoader:
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criteria(outputs, labels)
            loss.backward()
            optimiser.step()
            runningLoss += loss.item()
        
        epochLoss = runningLoss / len(trainLoader)
        lossHistory.append(epochLoss)
        print(f"Epoch {i+1}/{epochs}, loss = {epochLoss:.3f}")
    return lossHistory

def modelEvaluationNN(model, testLoader, labelEncoder):
    model.eval()
    predictionList = []
    labelList = []
    correctValues = 0
    totalValues = 0
    
    with torch.no_grad():
        for inputs, labels in testLoader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictionList.extend(predicted.cpu().numpy())
            labelList.extend(labels.cpu().numpy())
            correctValues += (predicted == labels).sum().item()
            totalValues += labels.size(0)
    
    accuracy = correctValues / totalValues
    print(f"Accuracy Score: {accuracy:.2f}")
    
    print("Classification Report:")
    print(classification_report(labelList, predictionList, target_names = labelEncoder.classes_), "\n")
    
    conf_matrix = confusion_matrix(labelList, predictionList)
    sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Greens", xticklabels = labelEncoder.classes_, yticklabels = labelEncoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()
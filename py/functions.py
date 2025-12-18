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

import torch #pytorch used for neural networks
import torch.nn as nn #neural network used from pytorch
import torch.optim as optim #optimisation algorithm used from pytorch

from torch.utils.data import DataLoader, TensorDataset #used to load data and create tensors from the data
from sklearn.preprocessing import LabelEncoder, StandardScaler #used to encode labels of data as numerical values, and standardises features by removing mean and scaling
import seaborn as sns #used to visualise the data as a confusion matrix


#functions used
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

class NeuralNetworkClassifier(nn.Module): #defines the neural network being used
    def __init__(self):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 64) #creates the first layer, with 5 input features and 64 output neurons
        self.fc2 = nn.Linear(64, 32) #creates second layer, with 64 inputs and 32 outputs
        self.fc3 = nn.Linear(32, 3) #creates third layer, with 32 inputs and 3 outputs
        self.relu = nn.ReLU() #defines the ReLu activation function, this replaces negative values with zero
        self.dropout = nn.Dropout(0.3) #defines the dropout layer, with a probability of setting 30% of neurons to 0
    
    def forward(self, z):
        z = self.relu(self.fc1(z)) #passes an input z through the first layer and applies relu
        z = self.dropout(z) #applies droputout to the output
        z = self.relu(self.fc2(z)) #passes data through second layer and applies relu
        z = self.fc3(z) #passes data through the final layer
        return z

def modelTraining(model, trainLoader, criteria, optimiser, epochs): #trains the neural network on our dataset
    model.train()
    lossHistory = []
    
    for i in range(epochs): #trains model over each epoch, refining it with each epoch
        runningLoss = 0.0
        for inputs, labels in trainLoader:
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criteria(outputs, labels)
            loss.backward()
            optimiser.step()
            runningLoss += loss.item()
        
        epochLoss = runningLoss / len(trainLoader)
        lossHistory.append(epochLoss) #stores loss at this epoch
        print(f"Epoch {i+1}/{epochs}, loss = {epochLoss:.3f}") #prints loss at epoch
    return lossHistory

def modelEvaluationNN(model, testLoader, labelEncoder): #evaluates the model compared to the dataset
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
    print(skl_cr(labelList, predictionList, target_names = labelEncoder.classes_), "\n")
    
    conf_matrix = confusion_matrix(labelList, predictionList) #creates confusion matrix
    sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Greens", xticklabels = labelEncoder.classes_, yticklabels = labelEncoder.classes_) #plots confusion matrix
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

def trainingDataSizeAffect(model, trainData, testLoader, fractions, criteria, optimiser, epochs):
    trainAccuracy = [] #stores accuracy of training
    testAccuracy = [] #stores accuracy of testing
    trainLoss = [] #stores loss of training
    testLoss = [] #stores loss of testing
    confMatrices = [] #stores confusion matrices
    
    for i in fractions: #runs through model training for each fraction of the training data
        subsetSize = int(len(trainData) * i)
        subsetData, _ = torch.utils.data.random_split(trainData, [subsetSize, len(trainData) - subsetSize])
        subsetLoad = DataLoader(subsetData, batch_size = 64, shuffle = True)
    
        model.apply(lambda layer: layer.reset_parameters() if hasattr(layer, 'reset_parameters') else None)
        
        model.train()
        for j in range(epochs):
            for inputs, labels in subsetLoad:
                optimiser.zero_grad()
                outputs = model(inputs)
                loss = criteria(outputs, labels)
                loss.backward()
                optimiser.step()
                
        trainCorrect = 0
        trainTotal = 0
        runningTrainLoss = 0.0
        model.eval()
        
        with torch.no_grad():
            for inputs, labels in subsetLoad:
                outputs = model(inputs)
                loss = criteria(outputs, labels)
                runningTrainLoss += loss.item()
                _, predicted = torch.max(outputs, 1)
                trainCorrect += (predicted == labels).sum().item()
                trainTotal += labels.size(0)
        
        trainAccuracy.append(trainCorrect / trainTotal)
        trainLoss.append(runningTrainLoss / len(subsetLoad))
        
        testCorrect = 0
        testTotal = 0
        runningTestLoss = 0.0
        allLabels = []
        allPredictions = []
        
        with torch.no_grad():
            for inputs, labels in testLoader:
                outputs = model(inputs)
                loss = criteria(outputs, labels)
                runningTestLoss += loss.item()
                _, predicted = torch.max(outputs, 1)
                testCorrect += (predicted == labels).sum().item()
                testTotal += labels.size(0)
                allLabels.extend(labels.cpu().numpy())
                allPredictions.extend(predicted.cpu().numpy())
        
        testAccuracy.append(testCorrect / testTotal)
        testLoss.append(runningTestLoss / len(testLoader))
        
        confMatrix = confusion_matrix(allLabels, allPredictions)
        confMatrices.append((i, confMatrix))
    
    return confMatrices, trainAccuracy, trainLoss, testAccuracy, testLoss #returns all the data gathered from training of the fractions of training data

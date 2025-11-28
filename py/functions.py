import kagglehub #used to get the data from kaggle.com
from kagglehub import KaggleDatasetAdapter #used to fetch the specific dataset from kaggle
import pandas as pd #used for viewing and manipulating the data
import matplotlib.pyplot as plt #used for data visulisation
import sklearn as skl
from sklearn.model_selection import train_test_split as skl_tts
from sklearn.tree import DecisionTreeClassifier as skl_dtc
from sklearn.metrics import classification_report as skl_cr
from sklearn.metrics import accuracy_score as skl_as

#all helper functions used below
def dataGrabber(): #fetches data from kaggle.com and returns it
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "bryancimo/sdss-galaxy-classification-dr18", "sdss_100k_galaxy_form_burst.csv", pandas_kwargs={'skiprows': 1})
    return df

def dataGrabber1(): #fetches data from kaggle.com and returns it
    dfFull = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "realkiller69/gaia-stars-dataset-from-dr3-data-release-3", "dataGaia2.csv")
    df = dfFull.loc[:, dfFull.columns != 'Unnamed: 0'] #removes unused first column

    def conditions(df): #sets conditions for applying the class of each object
        if (df['PGal'] > df['Pstar']) and (df['PGal'] > df['PWD']) and (df['PGal'] > df['Pbin']):
            return "Galaxy" #Galaxy
        elif (df['Pstar'] > df['PGal']) and (df['Pstar'] > df['PWD']) and (df['Pstar'] > df['Pbin']):
            return "Star" #Star
        elif (df['PWD'] > df['Pstar']) and (df['PWD'] > df['PGal']) and (df['PWD'] > df['Pbin']):
            return "White Dwarf" #White Dwarf
        elif (df['Pbin'] > df['Pstar']) and (df['Pbin'] > df['PGal']) and (df['Pbin'] > df['PGal']):
            return "Binary System" #Binary System
        else:
            return 4 #Undefined
    
    df['Class'] = df.apply(conditions, axis = 1) #adds class of each object to the dataset
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

#def treeVisualisation():
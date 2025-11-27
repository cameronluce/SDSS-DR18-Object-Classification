import kagglehub #used to get the data from kaggle.com
from kagglehub import KaggleDatasetAdapter #used to fetch the specific dataset from kaggle
import pandas as pd #used for viewing and manipulating the data
import matplotlib.pyplot as plt #used for data visulisation

#all helper functions used below
def dataGrabber(): #fetches data from kaggle.com and returns it
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "bryancimo/sdss-galaxy-classification-dr18", "sdss_100k_galaxy_form_burst.csv", pandas_kwargs={'skiprows': 1})
    return df

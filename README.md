## SDSS DR18 Object Classification by Neural and Non-Neural Methods

In this repository, we classify Stars, Galaxies and Quasars using data from the Sloan Digital Sky Survey (SDSS) data-release 18 (DR18), which can be found [here](https://www.kaggle.com/datasets/diraf0/sloan-digital-sky-survey-dr18?select=SDSS_DR18.csv).

## Overview of our Project:
The aim for this project is to use non-neural network (non-NN) and neural network (NN) approaches to the classification of stars, galaxies and quasars from SDSS DR18 to compare the two approaches and find optimal solutions. We aim to use machine learning (ML) for this problem, since there is plenty of data that needs classifying and the use of ML is a vital aid in this cause. By utilising ML and NN, we can find ways of classifying data more effectively than a manual approach which are significantly faster.

We split the project into 3 sections, as follows:

- Q1_notebook: Use of non-neural networks for classification
- Q2_notebook: Use of neural networks for classification
- Q3_notebook: Investigating affects of factors on performance of NN

In Q1_notebook, we use a decision tree to classify our dataset, looking at how it performs and finding out ways we can improve its performance. In Q2_notebook, we utilise a NN model to perform classification, utilising the NN built into PyTorch, from which we can evaluate its performance and compare it to the decision tree. Finally, in Q3_notebook, we look at how factors within our dataset and NN can affect the performance of our model, specifically the amount of training data and the balance of classes.

## Goals of our Project:

Our primary goals for this project are to:

- **Find a comparison between our techniques:** we want to see how our approaches perform, how they differ between the two, and if non-NN still have a use.
- **See how NN's perform:** we want to look into how chosen parameters affect the performance of our NN.
- **Perform a research question:** we want to address how choices about our data affect the performance of our NN, specifically the amount of training data chosen and the balance of our classes.

## Our chosen dataset:
We are using the eighteenth data release from the Sloan Digital Sky Survey (SDSS), containing ~100,000 astronomical objects, with each being classed as either a `STAR`, `GALAXY` or `QSO` also known as a Quasar. This data relase is the first data released from the fifth phase of SDSS's lifecycle (SDSS-V), released on the 19th of January 2023. Our specific dataset contains 43 features we can choose from, from which we narrow down to 7. These being the `u, g, r, i, z` bands of light, the `redshift` and the `class` of each object. These are further defined below:

|Feature|Description|
|---|---|
|`u`|The magnitude in ultraviolet band|
|`g`|The magnitude in the green optical band|
|`r`|The magnitude in the red optical band|
|`i`|The magnitude in the near-infrared band|
|`z`|The magnitude in the infrared band|
|`redshift`|Redshift of the object|
|`class`|Class of the object|

Our first rows of the dataset are seen below:



## Dependencies:


## License:
This project is under the MIT license, see `LICENSE` for more.

## Acknowledgements:

- Coleman Krawczyk; understanding of concepts utilised and assistance with debugging
- Dan Tassie; provided assistance with understanding the coursework structure and format
- GitHub Copilot; provided by Microsoft, used for debugging
- Chat GPT; provided by OpenAI, used for debugging

## Contact Details:
Cameron Luce\
cameronaluce@outlook.com\
Mphys Physics, Astrophysics and Cosmology\
University of Portsmouth

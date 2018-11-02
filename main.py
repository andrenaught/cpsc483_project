#Importing the essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#1. Import dataset
cols_to_use = [1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,21] #we'll ignore some columns
dataset = pd.read_csv("drug_consumption.data.txt", sep=",", header=None, usecols=cols_to_use)
dataset.columns = ["Age", "Gender", "Education", "Country", "Ethnicity", #1,2,3,4,5
				   "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "Sensation_seeking", #6,7,8,9,10,11,12
				   "Alcohol_cons", "Cannabis_cons", "Chocolate_cons","Crack_cons"] #13, 18, 19, 21
#print (dataset)

#It looks like 0 - 12 is already normalized and feature encoded - so don't have to worry about that.
#TODO: feature encode + normalize 13, 18, 19, 21
#TODO: check for missing values - replace with average

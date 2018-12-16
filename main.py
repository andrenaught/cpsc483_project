#Importing the essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Main Idea: 
1.	Predict someone's alcohol consumption based on their Age, Gender, Education, Country, Ethnicity,
	Nscore, Escore, Oscore, Ascore, Cscore, Impulsivity, Sensation Seeking
2.	Let's try predicting other drugs as well like Cannabis and Crack
'''

print("\n***** Preprocessing + Feature Selection + Data Separation ******\n")

#1. Import dataset
#header = ["ID", "Age", "Gender", "Education", "Country", "Ethnicity", "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impuslive", "Sensation_seeking", "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc", "Coke", "Crack", "Ecstacy", "Heroin", "Ketamine", "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "Semer", "VSA"]
#cols_to_use = [1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,21] #we'll ignore some columns
dataset = pd.read_csv("drug_consumption.data.txt", sep=",", header=None)
dataset.columns = ["ID", "Age", "Gender", "Education", "Country", "Ethnicity", "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "Sensation_seeking", "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis", "Choc", "Coke", "Crack", "Ecstacy", "Heroin", "Ketamine", "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "Semer", "VSA"]
#print (dataset)

#picking features set to choose from (this is not feature selection)
personality_only = [5,6,7,8,9,10,11]
all_x = [1,2,3,4,5,6,7,8,9,10,11,12]
all_except_alcohol = [1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
independent_cols = all_x

#choosing to do binary classification or multi-class classification
normal = {'CL0':0,'CL1':1,'CL2':2,'CL3':3,'CL4':4,'CL5':5,'CL6':6}
binary = {'CL0':0,'CL1':1,'CL2':1,'CL3':1,'CL4':1,'CL5':1,'CL6':1}
label_encoding = binary

x_col_names = dataset.columns.values[independent_cols]
dataset = dataset.replace(label_encoding)
X = dataset.iloc[:, independent_cols].values
y = dataset.iloc[:, [13]].values #lets choose Alcohol Consumption


#2. Missing data
#Description says there are no missing data

#3. Categorical data encoding
#There is no need for the X, all values are already numerical
from sklearn import preprocessing
#labels = y
#encoder = preprocessing.LabelEncoder()
#encoder.fit(labels)
#y = encoder.transform(y.ravel())
#print(X[0][])
#encoder.fit(X[16])

#4. Normalize from 0 - 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

#5. Split Dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) #20% of the dataset is put into test set, 80% into training set

#6. Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


#For classification: chi2, f_classif, mutual_info_classif. We pick chi2
feature_selector = SelectKBest(chi2, k=6) # get 4 best features
X_train = feature_selector.fit_transform(X_train, y_train)
cols_to_keep = []
for index, val in enumerate(feature_selector.get_support()):
	if(val == True):
		cols_to_keep.append(index)

print("Features selected: ")
print(x_col_names[cols_to_keep])
print(feature_selector.scores_[cols_to_keep])

#remove the features from the test data as well
mask = np.array(feature_selector.get_support())
X_test = X_test[:, mask]


#It looks like 0 - 12 is already normalized and feature encoded - so don't have to worry about that.


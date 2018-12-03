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

#1. Import dataset
cols_to_use = [1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,21] #we'll ignore some columns
dataset = pd.read_csv("drug_consumption.data.txt", sep=",", header=None, usecols=cols_to_use)
dataset.columns = ["Age", "Gender", "Education", "Country", "Ethnicity", #1,2,3,4,5
				   "Nscore", "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "Sensation_seeking", #6,7,8,9,10,11,12
				   "Alcohol_cons", "Cannabis_cons", "Chocolate_cons","Crack_cons"] #13, 18, 19, 21
#print (dataset)
personality_only = [5,6,7,8,9,10,11]
all_x = [0,1,2,3,4,5,6,7,8,9,10,11]
independent_cols = personality_only

y_col_names = dataset.columns.values[independent_cols]
print(y_col_names)
X = dataset.iloc[:, independent_cols].values
y = dataset.iloc[:, [12]].values #lets choose Alcohol Consumption

#2. Missing data
#Description says there are no missing data

#3. Categorical data encoding
#There is no need for the X, all values are already numerical
from sklearn import preprocessing
labels = ["CL0", "CL1", "CL2", "CL3", "CL4", "CL5", "CL6"]
encoder = preprocessing.LabelEncoder()
encoder.fit(labels)
y = encoder.transform(y.ravel())
#print('More labels encoded =', list(y))


#4. Normalize from 0 - 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

#5. Split Dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) #20% of the dataset is put into test set, 80% into training set

#6. Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#For classification: chi2, f_classif, mutual_info_classif. We pick chi2
feature_selector = SelectKBest(chi2, k=4) # get 4 best features
X_train = feature_selector.fit_transform(X_train, y_train)
cols_to_keep = []
for index, val in enumerate(feature_selector.get_support()):
	if(val == True):
		cols_to_keep.append(index)

print("features selected: ")
print(y_col_names[cols_to_keep])
#print(feature_selector.scores_)

#remove the features from the test data as well
mask = np.array(feature_selector.get_support())
X_test = X_test[:, mask]

##### Start The Model #####

#1. SVM
print("\n\nStarting SVM Model: ")
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(X_test)

#identify result
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
print("\nAccuracy Score (SVM):")
print(accuracy_score(y_test,y_pred))

#TODO: plot it

#It looks like 0 - 12 is already normalized and feature encoded - so don't have to worry about that.
#TODO: feature encode + normalize 13, 18, 19, 21
#TODO: check for missing values - replace with average

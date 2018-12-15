##### Get Preprocessed Data, already separated #####
from main import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

##### Start The Model #####

#1. Random Forest

#Hyper Parameter Tuning
param_grid = {'criterion': ['gini', 'entropy'],
   'max_depth': [5, 10, 25, 100, None],
   'n_estimators': [5, 10, 25, 50, 500], #number of trees in forest
   'max_features': ['auto', 'sqrt'],
   'bootstrap': [True, False]} 

'''
param_grid = [{'criterion': ['gini', 'entropy'],
   'max_depth': [3, 4, 5, 6, 7],
   'n_estimators': [400 ,450, 500, 550, 600], #number of trees in forest
   'max_features': ['auto', 'sqrt'],
   'bootstrap': [True, False]} 
]

param_grid = [{'criterion': ['gini', 'entropy'],
   'max_depth': [5],
   'n_estimators': [550], #number of trees in forest
   'max_features': ['auto', 'sqrt'],
   'bootstrap': [True, False]} 
]
'''



#Build + Train Classifier
print("\n***** Starting Random Forest Model ******")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
clf = RandomForestClassifier(random_state=0)
#clf = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1) #n_iter = # of different combinations to try
#clf = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Best Parameters: ")
#print(clf.best_params_)
#estimator = clf.estimators_[0]
#0.4005305039787798


#identify result
print("\n***** RESULT *****\n")
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
print("Mean Accuracy Score: " + str(clf.score(X_test, y_test)))
print("Mean Accuracy Score: " + str(accuracy_score(y_test,y_pred)))
#print("Feature Importance Scores:")
#print(clf.feature_importances_)


#print(y_pred)

print("PREDICTED RESULTS")
print("[", end="")
for i in y_pred:
  print(i, end=" ")
print("]")

print("ACTUAL RESULTS")
print("[", end="")
for i in y_test:
  print(i[0], end=" ")
print("]")
#printArray(y_test)

#TODO: plot it

from sklearn.tree import export_graphviz
# Export as dot file
'''
export_graphviz(estimator, out_file='tree.dot',
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')
'''

print("\n***** END *****\n")

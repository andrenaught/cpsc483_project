##### Get Preprocessed Data, already separated #####
from main import *

##### Start The Model #####

#1. SVM

#Hyper Parameter Tuning
param_grid = [
  {'C': [1, 10, 100, 1000],
   'loss': ['hinge', 'squared_hinge'],
   'max_iter': [100, 500, 1000]}
 ]

#Build + Train Classifier
print("\n***** Starting SVM Model ******")
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV


clf = LinearSVC(random_state=0, multi_class='ovr')
#rf_random = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

clf = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Best Parameters: ")
print(clf.best_params_)
print("\nParameters:")
print(clf.get_params())
for param, score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
    print (param, score)

#identify result
print("\n***** RESULT *****\n")
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
print("Accuracy Score: " + str(clf.score(X_test, y_test)))
print("Accuracy Score: " + str(accuracy_score(y_test,y_pred)))
print("Intercepts: ")
#print(clf.intercept_)
#print(clf.coef_)
#TODO: plot it

print("\n***** END *****\n")

print(y_pred)
print(y_test)


#plot
#MAKE SURE ONLY 2 FEATURES IF YOU WANT GRAPH TO DISPLAY

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train, y_train, clf=clf)
plt.legend(loc='upper left')
plt.show()

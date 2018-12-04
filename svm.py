##### Get Preprocessed Data, already separated #####
from main import *

##### Start The Model #####

#1. SVM
print("\n***** Starting SVM Model ******")
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
clf = LinearSVC(random_state=0, multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#identify result
print("\n***** RESULT *****\n")
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
print("Accuracy Score: " + str(clf.score(X_test, y_test)))
print("Accuracy Score: " + str(accuracy_score(y_test,y_pred)))
print("Intercepts: ")
print(clf.intercept_)
#print(clf.coef_)
#TODO: plot it

print("\n***** END *****\n")

print(y_pred)
print(y_test)

#plot
#MAKE SURE ONLY 2 FEATURES IF YOU WANT GRAPH TO DISPLAY
'''
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_train, y_train, clf=clf)
plt.legend(loc='upper left')
plt.show()
'''
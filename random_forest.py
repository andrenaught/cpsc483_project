##### Get Preprocessed Data, already separated #####
from main import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

##### Start The Model #####

#1. Random Forest
print("\n***** Starting Random Forest Model ******")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#identify result
print("\n***** RESULT *****\n")
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
print("Accuracy Score: " + str(accuracy_score(y_test,y_pred)))
print("Feature Importance Scores:")
print(clf.feature_importances_)


print(y_pred)
print(y_test)

#TODO: plot it


print("\n***** END *****\n")

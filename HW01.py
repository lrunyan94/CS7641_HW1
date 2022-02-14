#HW01.py
#Author: Luke Runyan
#Class: CS7641
#Date: 20220212

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import plot_tree

#oneHot Encode an attribute
def OHE_Helper(df, target):
	# Get one hot encoding of atr
	one_hot = pd.get_dummies(df[target])
	# Drop column B as it is now encoded
	df = df.drop(target,axis = 1)
	# Join the encoded df
	df = df.join(one_hot)
	#print(df.head())
	return df  

def encode_Helper(df, target):
	le = LabelEncoder()
	labels = le.fit_transform(df[target])
	df = df.drop(target, axis=1)
	df[target] = labels
	return df

#
#==========================================================================
#
################################### MAIN ###################################
#
#==========================================================================

##########========== PRE-PROCESSING DS1==========##########
#CSV NAME
csv = 'aw_fb_data.csv'
target = 'activity'
x_drop = [target]
df = pd.read_csv(csv)
#pre processing
df = OHE_Helper(df, 'device')
df = encode_Helper(df, 'activity')

# #parse and split
Y = df[target]
X = df.drop(x_drop, axis=1)

# #Normalize Input Data
min_max = MinMaxScaler()
X_Norm = min_max.fit_transform(X)
X = pd.DataFrame(X_Norm)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.1, random_state = 42)

##########========== DECISION TREE ==========##########
clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=.00068)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________ DECISION TREE __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

##########========== BOOSTED DECISION TREE ==========##########
dec_tree = DecisionTreeClassifier(max_leaf_nodes = None, random_state=0, ccp_alpha=.00045)
path = dec_tree.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clf = AdaBoostClassifier(dec_tree, n_estimators=20, learning_rate = 1)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________ BOOSTED DECISION TREE __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

##########========== NEURAL NETWORK ==========##########
clf = MLPClassifier(hidden_layer_sizes=(18,200,), activation='relu', solver='adam', 
						alpha=0.0001, batch_size=100,
						learning_rate_init=0.001, max_iter=1000, shuffle=True, 
						random_state=None, tol=0.0001, verbose=True, warm_start=False, 
						early_stopping=False, 
						validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
						n_iter_no_change=10)

##########========== TEST SCORE ==========##########
start_time = time.time()
dec_tree.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________ NEURAL NETWORK __________")
print("Training Time", training_time)
print("Train Accuracy: ",dec_tree.score(x_train, y_train))
print("Test Accuracy: ",dec_tree.score(x_test,y_test))

##########========== KNN ==========##########
clf = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, 
							p=2, metric='minkowski', metric_params=None, n_jobs=None)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________ KNN __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

##########========== VSM ==========##########
clf = SVC(gamma='scale', kernel = 'poly', degree = 14)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________VSM __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))



###############################################################

##############    DATA SET 2 ######################

###############################################################


##########========== PRE-PROCESSING DS2==========##########
csv = 'heart_disease_health_indicators_BRFSS2015.csv'
df = pd.read_csv(csv)
target = 'HeartDiseaseorAttack'
x_drop = [target]

# #parse and split
Y = df[target]
X = df.drop(x_drop, axis=1)

# #Normalize Input Data
min_max = MinMaxScaler()
X_Norm = min_max.fit_transform(X)
X = pd.DataFrame(X_Norm)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)


# ##########========== DECISION TREE ==========##########
clf = DecisionTreeClassifier(criterion='entropy',ccp_alpha=.00125)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________DECISION TREE __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

 # # ##########========== BOOSTED DECISION TREE ==========##########
dec_tree = DecisionTreeClassifier(max_leaf_nodes = None, random_state=0, ccp_alpha=.00159)
path = dec_tree.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clf = AdaBoostClassifier(dec_tree, n_estimators=20, learning_rate = 1)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________ BOOSTED DECISION TREE __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

# ##########========== NEURAL NET ==========##########
clf = MLPClassifier(hidden_layer_sizes=(18,30,), activation='relu', solver='adam', 
						alpha=0.0001, batch_size=100,
						learning_rate_init=0.001, max_iter=1000, shuffle=True, 
						random_state=None, tol=0.0001, verbose=False, warm_start=False, 
						early_stopping=False, 
						validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
						n_iter_no_change=10)
#########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________NEURAL NET __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

# ##########========== KNN ==========##########
clf = KNeighborsClassifier(n_neighbors=12, weights='uniform', algorithm='auto', leaf_size=30, 
							p=2, metric='minkowski', metric_params=None, n_jobs=None)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________ KNN __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

# ##########========== SVM ==========##########
clf = SVC(gamma='scale', kernel = 'poly', degree=3)

##########========== TEST SCORE ==========##########
start_time = time.time()
dec_tree.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________ SVM __________")
print("Training Time", training_time)
print("Train Accuracy: ",dec_tree.score(x_train, y_train))
print("Test Accuracy: ",dec_tree.score(x_test,y_test))



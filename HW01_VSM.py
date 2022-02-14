#

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
import time

#oneHot Encode An attribute
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

##########========== PRE-PROCESSING ==========##########
#CSV NAME
# csv = 'aw_fb_data.csv'
# target = 'activity'
# x_drop = [target]
# df = pd.read_csv(csv)

csv = 'heart_disease_health_indicators_BRFSS2015.csv'
df = pd.read_csv(csv)
target = 'HeartDiseaseorAttack'
x_drop = [target]

# #pre processing
# df = OHE_Helper(df, 'device')
# df = encode_Helper(df, 'activity')

# #parse and split
Y = df[target]
X = df.drop(x_drop, axis=1)


# #Normalize Input Data
min_max = MinMaxScaler()
X_Norm = min_max.fit_transform(X)
X = pd.DataFrame(X_Norm)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)

##########========== MODEL ==========##########
clf = SVC(gamma='scale', kernel = 'poly')


##########========== KERNEL OPTIMIZATION ==========##########
# scores = []
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# for k in kernels:
# 	model = SVC(kernel=k)
# 	model.fit(x_train,y_train)
# 	scores.append(model.score(x_train,y_train))

# plt.bar(kernels,scores)
# plt.title('Accuracy v. Kernel Type')
# plt.ylabel('Accuracy')
# plt.show()

##########========== LEARNING CURVE ==========##########
# train_sizes, train_score, test_score = learning_curve(clf , x_train,y_train, cv=5, 
#  														scoring = 'accuracy', n_jobs=-1, 
#  														train_sizes = np.linspace(0.01,1.0, 10),
#  														verbose = True)
# train_mean = np.mean(train_score, axis=1)
# train_std = np.std(train_score, axis=1)
# test_mean = np.mean(test_score, axis=1)
# test_std = np.std(test_score, axis=1)
# plt.plot(train_sizes, train_mean, label="Training Score")
# plt.plot(train_sizes, test_mean, label = 'Validation Scores')
# plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, color='#DDDDDD', label="Cross Validation Set Standard Deviation")
# plt.fill_between(train_sizes, test_mean-test_std, test_mean+test_std, color='#DDDDDD')
# plt.xlabel('Training Size')
# plt.ylabel('Accuracy')
# plt.title(' Learning Curve')
# plt.legend()
# plt.show()


#########========== DEGREE OPTIMIZATION ==========##########
scores = []
train_times = []
#degrees= [14,15,]
degrees= [1,2,3,4,5,6]
for d in degrees:
	model = SVC(gamma='scale', kernel = 'poly', degree=d)
	cv_results = cross_validate(model, x_train, y_train, cv=3)
	train_mean = np.mean(cv_results['test_score'])
	train_times.append(np.mean(cv_results['fit_time']))
	scores.append(train_mean)


fig, ax = plt.subplots(2, 1)
ax[0].plot(degrees, scores)
ax[0].set_xlabel("Polynomial Degree")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Accuracy v. Polynomial Degree")
ax[1].plot(degrees, train_times)
ax[1].set_xlabel("Polynomial Degree")
ax[1].set_ylabel("Time to Train Model")
ax[1].set_title("Training Time v. Polynomial Degree")
fig.tight_layout()
plt.show()

# scores = []
# model = SVC(gamma='scale', kernel = 'poly', degree=4)
# cv_results = cross_validate(model, x_train, y_train, cv=3)

# print(cv_results['test_score'])
# print(cv_results['fit_time'])
# print(np.mean(cv_results['test_score']))
# print(np.mean(cv_results['fit_time']))









# model = SVC(C=10.0, kernel='poly', degree=9, gamma='scale', coef0=0.0, shrinking=True, 
# 			probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, 
# 			max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)
# start_time = time.time()
# model.fit(x_train,y_train)


# training_time = time.time()-start_time
# print("__________RESULTS__________")
# print('Training Time: ', training_time)
# print("Train Accuracy ",model.score(x_train,y_train))
# print("Test Accuracy: ",model.score(x_test,y_test))


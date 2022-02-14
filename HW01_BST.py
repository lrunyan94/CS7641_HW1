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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.compose import make_column_transformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import plot_tree

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
# ==========================================================================

# ################################## MAIN ###################################

# ==========================================================================

#########========== PRE-PROCESSING DS1==========##########
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

##########========== MODEL ==========##########
dec_tree = DecisionTreeClassifier(max_leaf_nodes = None, random_state=0, )#ccp_alpha=.00045
path = dec_tree.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clf = AdaBoostClassifier(dec_tree, n_estimators=20, learning_rate = 1)

##########========== TEST SCORE ==========##########
# start_time = time.time()
# clf.fit(x_train,y_train)
# training_time = time.time()-start_time
# print("__________DECISION TREE __________")
# print("Training Time", training_time)
# print("Train Accuracy: ",clf.score(x_train, y_train))
# print("Test Accuracy: ",clf.score(x_test,y_test))

# ##########========== PRE-PROCESSING DS2==========##########
# csv = 'heart_disease_health_indicators_BRFSS2015.csv'
# df = pd.read_csv(csv)
# target = 'HeartDiseaseorAttack'
# x_drop = [target]

# # #parse and split
# Y = df[target]
# X = df.drop(x_drop, axis=1)

# # #Normalize Input Data
# min_max = MinMaxScaler()
# X_Norm = min_max.fit_transform(X)
# X = pd.DataFrame(X_Norm)

# x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.5, random_state = 42)


# # # ##########========== MODEL ==========##########
# dec_tree = DecisionTreeClassifier(max_leaf_nodes = None, random_state=0, ccp_alpha=.00159)
# path = dec_tree.cost_complexity_pruning_path(x_train,y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
# clf = AdaBoostClassifier(dec_tree, n_estimators=20, learning_rate = 1)

# ##########========== TEST SCORE ==========##########
# start_time = time.time()
# clf.fit(x_train,y_train)
# training_time = time.time()-start_time
# print("__________DECISION TREE __________")
# print("Training Time", training_time)
# print("Train Accuracy: ",clf.score(x_train, y_train))
# print("Test Accuracy: ",clf.score(x_test,y_test))


##########========== LEARNING CURVE ==========##########
# train_sizes, train_score, test_score = learning_curve(clf , x_train,y_train, cv=5, 
#  														scoring = 'accuracy', n_jobs=-1, 
#  														train_sizes = np.linspace(0.01,1.0, 50),
#  														verbose = False)
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
# plt.title('Boosted Decision Tree Learning Curve')
# plt.legend()
# plt.show()

##########========== LEARNING RATE OPTIMIZATION ==========##########
# scores = []
# train_times = []
# learning_rates = np.linspace(.1,.1,10)
# for lr in learning_rates:
# 	model = AdaBoostClassifier(dec_tree, n_estimators=100, learning_rate = lr)
# 	start = time.time()
# 	model.fit(x_train,y_train)
# 	end = time.time()
# 	train_times.append(end-start)
# 	scores.append(model.score(x_train,y_train))

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(learning_rates, scores)
# ax[0].set_xlabel("Learning Rate")
# ax[0].set_ylabel("Accuracy")
# ax[0].set_title("Accuracy v. Learning Rate")
# ax[1].plot(learning_rates, train_times)
# ax[1].set_xlabel("Learning Rate")
# ax[1].set_ylabel("Time to Train")
# ax[1].set_title("Training Time v. Learning Rate")
# fig.tight_layout()
# plt.show()

##########========== PRUNING OPTIMIZATION ==========##########
scores = []
train_times = []
ccp_alphas = ccp_alphas[100:250]

for alpha in ccp_alphas:
	dec_tree = DecisionTreeClassifier(max_leaf_nodes = None, random_state=0, ccp_alpha=alpha)
	model = AdaBoostClassifier(dec_tree, n_estimators=20, learning_rate = 1)
	cv_results = cross_validate(model, x_train, y_train, cv=3)
	train_mean = np.mean(cv_results['test_score'])
	train_times.append(np.mean(cv_results['fit_time']))
	scores.append(train_mean)

fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, scores)
ax[0].set_xlabel("Alpha")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Accuracy v. Alpha")
ax[1].plot(ccp_alphas, train_times)
ax[1].set_xlabel("Alpha")
ax[1].set_ylabel("Time to Train")
ax[1].set_title("Training Time v. Alpha")
fig.tight_layout()
plt.show()
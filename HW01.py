#

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
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

########## MAIN ##########
#CSV NAME
csv = 'aw_fb_data.csv'
target = 'activity'
x_drop =[target] 

# Read Data and parse into test and train data
df = pd.read_csv(csv)

#pre processing
df = OHE_Helper(df, 'device')
#df = encode_Helper(df, 'activity')
activity_dict = {'Lying':0, 'Sitting':1, 'Self Pace walk':2, 'Running 3 METs':3, 'Running 5 METs':3, 'Running 7 METs':3 }
df['activity'] = df.activity.apply(lambda x: activity_dict[x])


# #Discard Outliers
# clf = LocalOutlierFactor()
# y_pred = clf.fit_predict(df) 
# x_score = clf.negative_outlier_factor_
# outlier_score = pd.DataFrame()
# outlier_score["score"] = x_score

# #threshold
# threshold2 = -1.5                                            
# filtre2 = outlier_score["score"] < threshold2
# outlier_index = outlier_score[filtre2].index.tolist()

# df.drop(outlier_index, inplace=True)


#parse and split
Y = df[target]
X = df.drop(x_drop, axis=1)
min_max = MinMaxScaler()
X = min_max.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3, random_state = 42)


#create decision tree
dec_tree = DecisionTreeClassifier(max_leaf_nodes = None, class_weight = None, random_state=0)
start_time = time.time()
dec_tree.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________NO PRUNING__________")
print('Training Time: ', training_time)
print("Train Accuracy ",dec_tree.score(x_train,y_train))
print("Test Accuracy: ",dec_tree.score(x_test,y_test))

path = dec_tree.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# fig, ax = plt.subplots()
# ax.plot(ccp_alphas[:-1],impurities[:-1], marker="o", drawstyle="steps-post")
# ax.set_xlabel("effective alpha")
# ax.set_ylabel("Totatl Impurity of Leaves")
# ax.set_title("Total Impurity vs Effective Alpha for Training Set")
# plt.show()

# clfs = []
# for ccp_alpha in ccp_alphas:
# 	clf = DecisionTreeClassifier(max_leaf_nodes = None, random_state=0, ccp_alpha=ccp_alpha)
# 	clf.fit(x_train, y_train)
# 	clfs.append(clf)
# print(
# 	"Number of nodes in the last tree is: {} with ccp_Alpha: {}".format(clfs[-1].tree_.node_count, ccp_alphas[-1]
# 	)
# )

# clfs = clfs[:-1]
# ccp_alphas = ccp_alphas[:-1]

# node_counts = [clf.tree_.node_count for clf in clfs]
# depth = [clf.tree_.max_depth for clf in clfs]
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
# ax[0].set_xlabel("alpha")
# ax[0].set_ylabel("number of nodes")
# ax[0].set_title("Number of nodes vs alpha")
# ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
# ax[1].set_xlabel("alpha")
# ax[1].set_ylabel("depth of tree")
# ax[1].set_title("Depth vs alpha")
# fig.tight_layout()
# plt.show()

# train_scores = [clf.score(x_train, y_train) for clf in clfs]
# test_scores = [clf.score(x_test, y_test) for clf in clfs]

# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
# ax.legend()
# plt.show()

dec_tree = DecisionTreeClassifier(max_leaf_nodes = None, random_state=0, ccp_alpha=.0002)
start_time = time.time()
dec_tree.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________AFTER PRUNING__________")
print("Training Time", training_time)
print("Train Accuracy: ",dec_tree.score(x_train, y_train))
print("Test Accuracy: ",dec_tree.score(x_test,y_test))

# plt.figure(figsize = (20,10))
# plot_tree(dec_tree, 
# 	feature_names = X.columns,
# 	class_names = ["0", "1"],
# 	rounded = True,
# 	filled = True)
# plt.show()
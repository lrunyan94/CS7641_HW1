import matplotlib.pyplot as plt
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

##########========== MODEL ==========##########
clf = DecisionTreeClassifier(criterion='entropy', ccp_alpha=.00068)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________DECISION TREE __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

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


# ##########========== MODEL ==========##########
clf = DecisionTreeClassifier(criterion='entropy',ccp_alpha=.00125)

##########========== TEST SCORE ==========##########
start_time = time.time()
clf.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________DECISION TREE __________")
print("Training Time", training_time)
print("Train Accuracy: ",clf.score(x_train, y_train))
print("Test Accuracy: ",clf.score(x_test,y_test))

########################################################################################


# ##########========== LEARNING CURVE ==========##########
# train_sizes, train_score, test_score = learning_curve(clf , x_train,y_train, cv=10, 
#  														scoring = 'accuracy', n_jobs=-1, 
#  														train_sizes = np.linspace(0.01,.5, 100),
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
# plt.title('Decision Tree Learning Curve')
# plt.legend()
# plt.show()


##########========== PRUNING ==========##########

# path = clf.cost_complexity_pruning_path(x_train,y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
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
# ax[0].set_xlabel("Alpha")
# ax[0].set_ylabel("Number of Nodes")
# ax[0].set_title("Number of nodes vs CCP Alpha")
# ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
# ax[1].set_xlabel("Alpha")
# ax[1].set_ylabel("Depth of Tree")
# ax[1].set_title("Depth vs CCP alpha")
# fig.tight_layout()
# plt.show()

# scores = []
# stds = []
# for ccp_alpha in ccp_alphas:
# 	clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes = None, random_state=0, ccp_alpha=ccp_alpha)
# 	score = cross_val_score(clf, x_train, y_train, cv=5)
# 	stds.append(score.std())
# 	scores.append(score.mean())

# plt.plot(ccp_alphas,scores, marker="o", drawstyle="steps-post")
# plt.title('Decision Tree Accuracy v. CCP Alpha Values')
# plt.xlabel('Alpha')
# plt.ylabel('Model Accuracy')
# plt.show()

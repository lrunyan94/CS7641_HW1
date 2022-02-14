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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.compose import make_column_transformer
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from tensorflow import keras


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

#########========== PRE-PROCESSING DS1==========##########
# #CSV NAME
# csv = 'aw_fb_data.csv'
# target = 'activity'
# x_drop = [target]
# df = pd.read_csv(csv)
# #pre processing
# df = OHE_Helper(df, 'device')
# df = encode_Helper(df, 'activity')

# # #parse and split
# Y = df[target]
# X = df.drop(x_drop, axis=1)

# # #Normalize Input Data
# min_max = MinMaxScaler()
# X_Norm = min_max.fit_transform(X)
# X = pd.DataFrame(X_Norm)
# x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.1, random_state = 42)

# ##########========== MODEL ==========##########
# clf = MLPClassifier(hidden_layer_sizes=(18,10,), activation='relu', solver='adam', 
# 						alpha=0.0001, batch_size=100,
# 						learning_rate_init=0.001, max_iter=1000, shuffle=True, 
# 						random_state=None, tol=0.0001, verbose=True, warm_start=False, 
# 						early_stopping=False, 
# 						validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
# 						n_iter_no_change=10)

##########========== TEST SCORE ==========##########
# start_time = time.time()
# dec_tree.fit(x_train,y_train)
# training_time = time.time()-start_time
# print("__________DECISION TREE __________")
# print("Training Time", training_time)
# print("Train Accuracy: ",dec_tree.score(x_train, y_train))
# print("Test Accuracy: ",dec_tree.score(x_test,y_test))



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


##########========== MODEL ==========##########
clf = MLPClassifier(hidden_layer_sizes=(18,30,), activation='relu', solver='adam', 
						alpha=0.0001, batch_size=100,
						learning_rate_init=0.001, max_iter=1000, shuffle=True, 
						random_state=None, tol=0.0001, verbose=True, warm_start=False, 
						early_stopping=False, 
						validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
						n_iter_no_change=10)
##########========== TEST SCORE ==========##########
# start_time = time.time()
# clf.fit(x_train,y_train)
# training_time = time.time()-start_time
# print("__________DECISION TREE __________")
# print("Training Time", training_time)
# print("Train Accuracy: ",clf.score(x_train, y_train))
# print("Test Accuracy: ",clf.score(x_test,y_test))


##########========== LEARNING CURVE ==========##########
# #Build NN Model & Compile
# model = keras.Sequential([ 
# 	keras.layers.Dense(34, input_shape=(17,), activation='relu'),
# 	keras.layers.Dense(680, activation='relu'),
# 	keras.layers.Dense(6, activation='sigmoid')])
# model.compile(optimizer='adam',
# 			loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# 			metrics=['accuracy'])

# #Train Model
# start = time.time()
# #model.fit(x_train, y_train, batch_size = 100, epochs=500)
# model.fit(x_train,y_train)
# #Calculate execution time
# end = time.time()
# print("Execution time: ", end-start, "Seconds")

# print("MODEL EVALUATION")
# model.evaluate(x_test,y_test)


##########========== LAYER OPTIMIZATION ==========##########
# scores = []
# train_times = []
# layer_sizes = [100,125,150, 175, 200]
# for size in layer_sizes:
# 	model = MLPClassifier(hidden_layer_sizes=(18,size,), activation='relu', solver='adam', 
# 						alpha=0.0001, batch_size=100,
# 						learning_rate_init=0.001, max_iter=1000, shuffle=True, 
# 						random_state=None, tol=0.0001, verbose=True, warm_start=False, 
# 						early_stopping=False, 
# 						validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
# 						n_iter_no_change=10)
# 	start = time.time()
# 	model.fit(x_train,y_train)
# 	end = time.time()
# 	train_times.append(end-start)
# 	scores.append(model.score(x_train,y_train))

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(layer_sizes, scores)
# ax[0].set_xlabel("Nodes in Hidden Layer")
# ax[0].set_ylabel("Accuracy")
# ax[0].set_title("Neural Network Accuracy v. Nodes in Hidden Layer")
# ax[1].plot(layer_sizes, train_times)
# ax[1].set_xlabel("Nodes in Hidden Layer")
# ax[1].set_ylabel("Time to Train Model")
# ax[1].set_title("Training Time v. Nodes in Hidden Layer")
# fig.tight_layout()
# plt.show()


##########========== LEARNING RATE OPTIMIZATION ==========##########
# # scores = []
# train_times = []
# learning_rates = linspace(.1,.1,.1)
# for lr in learning_rates:
# 	model = MLPClassifier(hidden_layer_sizes=(17,size,), activation='relu', solver='adam', 
# 						alpha=0.0001, batch_size=50,
# 						learning_rate_init=lr, max_iter=1000, shuffle=True, 
# 						random_state=None, tol=0.0001, verbose=True, warm_start=False, 
# 						early_stopping=False, 
# 						validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
# 						n_iter_no_change=10)
# 	start = time.time()
# 	model.fit(x_train,y_train)
# 	end = time.time()
# 	train_times.append(end-start)
# 	scores.append(model.score(x_train,y_train))

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(layer_sizes, scores)
# ax[0].set_xlabel("Nodes in Hidden Layer")
# ax[0].set_ylabel("Accuracy")
# ax[0].set_title("Neural Network Accuracy v. Nodes in Hidden Layer")
# ax[1].plot(layer_sizes, train_times)
# ax[1].set_xlabel("Nodes in Hidden Layer")
# ax[1].set_ylabel("Time to Train Model")
# ax[1].set_title("Training Time v. Nodes in Hidden Layer")
# fig.tight_layout()
# plt.show()

#Train Model
start = time.time()
#model.fit(x_train, y_train, batch_size = 100, epochs=500)
clf.fit(x_train,y_train)
#Calculate execution time
end = time.time()
# print()
# print()
# print("======================RESULTS=====================")
# print()
# print("Execution time: ", end-start, "Seconds")
# print("Training Accuracy: :", model.score(x_train,y_train))
# print("Validation Accuracy: :", model.score(x_test,y_test))
# print()
# print("--------------------------------------------------")
# print()
# print()

plt.plot(clf.loss_curve_)
plt.ylabel('Model Loss')
plt.xlabel('Training Iterations')
plt.title('Neural Network Model Loss v. Training Iterations')
plt.show()

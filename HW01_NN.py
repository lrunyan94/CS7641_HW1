#Tensor Flow Test

#import
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np

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
#x_drop =['Lying', 'Running 3 METs', 'Running 5 METs', 'Running 7 METs', 'Self Pace walk', 'Sitting'] 
x_drop = [target]

# Read Data and parse into test and train data
df = pd.read_csv(csv)

#pre processing
df = OHE_Helper(df, 'device')
#df = OHE_Helper(df, 'activity')
#df = encode_Helper(df, 'activity')
activity_dict = {'Lying':0, 'Sitting':1, 'Self Pace walk':2, 'Running 3 METs':3, 'Running 5 METs':3, 'Running 7 METs':3 }
df['activity'] = df.activity.apply(lambda x: activity_dict[x])

#Normalize Data

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
Y = df['activity']
X = df.drop(['activity', 'fitbit'], axis=1)

var_cols = X.columns
min_max = MinMaxScaler()
for c in var_cols:
	print(c)
	df[c] = min_max.fit_transform(df[c])
print(df.head())

# x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 42)

# # #Build NN Model & Compile
# model = keras.Sequential([ 
# 	keras.layers.Dense(34, input_shape=(17,), activation='relu'),
# 	keras.layers.Dense(680, activation='relu'),
# 	keras.layers.Dense(4, activation='sigmoid')])
# model.compile(optimizer='adam',
# 			loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# 			metrics=['accuracy'])


# # #Train Model
# start = time.time()
# model.fit(x_train, y_train, batch_size = 1, epochs=5)

# #Calculate execution time
# end = time.time()
# print("Execution time: ", end-start, "Seconds")

# print("MODEL EVALUATION")
# model.evaluate(x_test,y_test)


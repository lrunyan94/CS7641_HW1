#

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

#Normalize Input Data
min_max = MinMaxScaler()
X_Norm = min_max.fit_transform(X)
X = pd.DataFrame(X_Norm)

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3, random_state = 42)


#create decision tree
ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=20, learning_rate = 1)

start_time = time.time()
ada.fit(x_train,y_train)
training_time = time.time()-start_time
print("__________RESULTS__________")
print('Training Time: ', training_time)
print("Train Accuracy ",ada.score(x_train,y_train))
print("Test Accuracy: ",ada.score(x_test,y_test))



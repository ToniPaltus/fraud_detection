# import libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# data preparation
train_features = pd.read_csv('../datasets/train_features.csv')
test_features = pd.read_csv('../datasets/test_features.csv')

train_features.drop(['Unnamed: 0'], axis=1, inplace=True)
test_features.drop(['Unnamed: 0'], axis=1, inplace=True)

#print(train_features.head())


# Model preparation
categories = ['category', 'merchant', 'state', 'job']

# Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

for category in categories:
    train_features[category] = label_encoder.fit_transform(train_features[category])
    test_features[category] = label_encoder.fit_transform(test_features[category])

train_features.head()

#Split
y_train = train_features['is_fraud'].values
X_train = train_features.drop(['is_fraud'], axis='columns').values

y_test = test_features['is_fraud'].values
X_test = test_features.drop(['is_fraud'], axis='columns').values

print('y_train:   ', len(y_train))
print('X_train:   ', len(X_train))
print()
print('y_test:   ', len(y_test))
print('X_test:   ', len(X_test))

# SMOTE
from imblearn.over_sampling import SMOTE

method = SMOTE()
X_train_resampled, y_train_resampled = method.fit_resample(X_train, y_train)
X_test_resampled, y_test_resampled = method.fit_resample(X_test, y_test)

print(len(y_train_resampled[y_train_resampled == 0]))
print(len(y_train_resampled[y_train_resampled == 1]))
print(len(X_train_resampled))
print('X_resampled:\t', len(X_train_resampled))
print('y_resampled:\t', len(y_train_resampled))

# Modeling
from sklearn.svm import SVC

# svc_model = SVC()
# svc_model.fit(X_train_resampled, y_train_resampled)
#
# predict = svc_model.predict(X_test)
#
# print('>>> Confusion matrix:\n', confusion_matrix(y_test, predict), end='\n\n')
# print('>>> Classification report:\n', classification_report(y_test, predict), end='\n\n')
# print('>>> ROC-AUC:\t', roc_auc_score(y_test, predict))


print("Num GPUs available: ", len(tf.config.list_physical_devices('GPU')))

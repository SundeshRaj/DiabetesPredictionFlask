import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("diabetes.csv")

# 1 = Glucose
# 4 = Insulin
# 5 = BMI
# 7 = Age
X = dataset.iloc[:,[1,4,5,7]]
y = dataset.iloc[:,[8]]

model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state = 0)
model.fit(X,y.values.ravel())

pickle.dump(model, open('randForest.pkl', 'wb'))

testModel = pickle.load(open('randForest.pkl','rb'))
print(testModel.predict([[85,0,23.3,27]]))




import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



def create_model():
    model = Sequential()
    #16 feature inputs (votes) going into an 32-unit layer 
    model.add(Dense(32, input_dim=5, kernel_initializer='normal', activation='relu'))
    # Another hidden layer of 16 units
    model.add(Dropout(0.2))
    #addig dropout of 0.2 so only 80% of neurons used
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
     #addig dropout of 0.2 so only 80% of neurons used
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
     #addig dropout of 0.2 so only 80% of neurons used
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
     #addig dropout of 0.2 so only 80% of neurons used
    model.add(Dropout(0.2))
    # Output layer with a binary classification (Democrat or Republican political party)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

breast_cancer_data = pd.read_csv('Breast_cancer_data.csv')
breast_cancer_data.dropna(inplace=True)

all_features = breast_cancer_data[['mean_radius','mean_texture', 'mean_perimeter', 
                    'mean_area', 'mean_smoothness','diagnosis']].drop('diagnosis', axis=1).values
all_classes = breast_cancer_data['diagnosis'].values

# Wrap our Keras model in an estimator compatible with scikit_learn
x = [17.99],[10.38],[122.8],[1001.0],[0.1184]
x = np.array([[9.504,12.44,60.34,273.9,0.1024]])
estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
estimator.fit(all_features,all_classes)
print(estimator.predict(x))
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
print(cv_scores.mean())







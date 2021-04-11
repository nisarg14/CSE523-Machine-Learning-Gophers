# Code

CTGAN will extend the dataset by finding out patters in it and following the similar patterns. This extended our dataset from 15000 samples to 1 million samples. For that, we have used to following code snipper:

```python
from ctgan import CTGANSynthesizer
discrete_columns = range(100, 120)

ctgan = CTGANSynthesizer(epochs=10)
ctgan.fit(df, discrete_columns)

samples = ctgan.sample(1000000)
```

1. Random Forest Classifier (Conventional)

In this, we have used Sklearn's random forest classifier algorithm for default values on our dataset
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_jobs=-1, verbose=True)

%time model.fit(x_train.drop(columns='3'), y_train)

x_test.drop(columns='2', inplace=True)

%time model.score(x_test, y_test)
```

2. Random Forest Classifier using Optuna

In this approach, we have used optuna to optimise the hyperparameters of RFC. The n_estimators range from 2 to 20 and the max_depth ranges from 1 to 32

```python
def objective(trial):
    
    n_estimators = trial.suggest_int('n_estimators', 2, 20)
    max_depth = int(trial.suggest_float('max_depth', 1, 32, log=True))
    
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
    
    clf.fit(x_train.drop(columns='3'), y_train)
    
    return clf.score(x_test, y_test)  

```

3. XGBoose Classifier using Optuna

In this approach, we have used optuna to optimise the hyperparameters of XGBoost classsifier. The n_estimators range from 0 to 1000 and the learning rate ranges from 0.005 to 0.5 loguniformly'

**_NOTE:_** We have used a GPU based approach to increase the speed of the algorithms. So please refer the notebook before using this code snippet

```python
def objective(trial):   
    
#changing parameters at every trial
    param = {
                "n_estimators" : trial.suggest_int('n_estimators', 0, 1000),  #suggest_int is for the range of of the parameter
                'max_depth':trial.suggest_int('max_depth', 2, 25),
                'reg_alpha':trial.suggest_int('reg_alpha', 0, 5),
                'reg_lambda':trial.suggest_int('reg_lambda', 0, 5),
                'min_child_weight':trial.suggest_int('min_child_weight', 0, 5),
                'gamma':trial.suggest_int('gamma', 0, 5),
                'learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
                'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
                'nthread' : -1,
                'verbosity' : 3,
                'njobs' : -1, #n_jobs – The number of parallel jobs. If this argument is set to -1, the number is set to CPU count.
                'tree_method': 'gpu_hist'
            }
    model = XGBClassifier(**param)  #picking model as XGBoost classifier

    model.fit(x_train.drop(columns='3'), y_train)   #fitting the model

    return model.score(x_test, y_test)
```

4. K-Nearest Neighbours using Optuna

In this approach, we have used optuna to optimise the hyperparameters of RFC. The n_neighbors range from 2 to 20 and the leaf_size ranges from 10 to 400

```python

def objective(trial):  #optuna knn function
    
    #its parameters
    n_neighbors = trial.suggest_int('n_neighbors', 2, 20)
    leaf_size = int(trial.suggest_float('leaf_size', 10, 400))
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, n_jobs=-1)
    
    clf.fit(x_train.drop(columns='3'), y_train)
    
    return clf.score(x_test, y_test)
```

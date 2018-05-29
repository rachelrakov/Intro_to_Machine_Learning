[<<< Previous](parameters.md) | [Next >>>](resources.md)

## Changing Paramaters

Every classification algorithm has paramaters, which we can see above where we created an instance of a classifier.

~~~
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
~~~

In our linear SVC example, the paramater is C, with a default of 1.0.  It is important to know that the default paramater is not always the best paramater for the data, and that it is common to try several different values of C in order to optimize the algorithm for your data.  In the code below, we show an example of searching through several different values of C to find the best value for paramater C for our particular data.  We search across our training data only, to ensure that we are not generalizing too closely to our testing data (which would be an example of *overfitting* our data to the classification algorithm).


```python
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

from sklearn.grid_search import GridSearchCV
grid_search = GridSearchCV(LinearSVC(), param_grid, cv=10)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
```

    {'C': 1}
    0.7082474226804124
    

With our new optimal C paramater (which in our case happens to be the default paramater, though that is not always the case), we can reclassify, setting C manually with the result of C that we get from grid searching through different paramaters of C. 


```python
classifier = LinearSVC(C=1)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
```


```python
classifier.score(X_test, y_test)
```




    0.70892226148409898



## Add visualization or animation of the paramater searching (Hannah)

## At the end, add some questions about ethics of machine learning (Hannah)


```python

```

[<<< Previous](parameters.md) | [Next >>>](resources.md)

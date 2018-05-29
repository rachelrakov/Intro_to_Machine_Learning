# Supervised Classification with sklearn

One of the best things about sklearn is the simplicity of its syntax.

To do machine learning with sklearn, follow these three steps (the function names remain the same, regardless of the classifier you use!):

## Step 1:  Import your desired classifier


```python
from sklearn.svm import LinearSVC
```

## Step 2: Create an instance of your machine learning algorithm


```python
classifier = LinearSVC()
```

## Step 3:  Fit your data to your classifier (train), predict labels for unseen data (test), and score!


```python
classifier.fit(X_train, y_train)

```




    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)




```python
y_predict = classifier.predict(X_test)
```


```python
classifier.score(X_test, y_test)
```




    0.70803886925795056



Right now, our classifier can correctly predict previously unseen news and data about 71% of the time.  We can get more information about how we doing by creating a confusion matrix. This confusion matrix shows how many times we are predicting categories correctly.


```python
from sklearn.metrics import confusion_matrix
```


```python
confusion_matrix(y_test, y_predict)
confusion_matrix
```


|      |actual news | actual romance |
|:--: | :--:| :--:|
|predicted news | 759 | 397 |
|predicted romance|282 | 826|



## Visualization of the decision boundry


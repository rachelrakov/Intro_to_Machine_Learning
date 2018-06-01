[<<< Previous](unsupervised.md) | [Next >>>](lda.md)

## Feature Extraction Using Bag of Words

We're almost ready to do some machine learning!  First, we need to turn our sentences into the type of *feature vectors* the algorithm we plan to work with expects. Jumping ahead a bit, the `Sklearn` implementation of the algorithm we will use for unsupervised learning requires that the text be in *bag of words* form, which is the unique words in the text and the count of occurances of that word. 

### Read data in from a spreadsheet
Lets take the data we just saved out and load it back into a dataframe so that we can do some analysis with it!

```python
import pandas as pd
df = pd.read_csv("df_news_romance.csv")
df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>sentence</th>
      <th>NN</th>
      <th>JJ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>news</td>
      <td>['The', 'Fulton', 'County', 'Grand', 'Jury', '...</td>
      <td>11</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>news</td>
      <td>['The', 'jury', 'further', 'said', 'in', 'term...</td>
      <td>13</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>news</td>
      <td>['The', 'September-October', 'term', 'jury', '...</td>
      <td>16</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>news</td>
      <td>['``', 'Only', 'a', 'relative', 'handful', 'of...</td>
      <td>9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>news</td>
      <td>['The', 'jury', 'said', 'it', 'did', 'find', '...</td>
      <td>5</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>

Then we print the first 5 rows of the *sentence* column in the spreadsheet:

```python
df['sentence'].head()
```

    0    ['The', 'Fulton', 'County', 'Grand', 'Jury', '...
    1    ['The', 'jury', 'further', 'said', 'in', 'term...
    2    ['The', 'September-October', 'term', 'jury', '...
    3    ['``', 'Only', 'a', 'relative', 'handful', 'of...
    4    ['The', 'jury', 'said', 'it', 'did', 'find', '...
    Name: sentence, dtype: object



### Bag of Words
We preprocess our data using sklearn's text feature extraction tools. In particular, we use the `CountVectorizer` which computes the frequency of each token in the document. We can strip out stop words using the `stop_words` keyword argument.


```python
from sklearn.feature_extraction.text import CountVectorizer

tf_vectorizer = CountVectorizer(stop_words='english')
tf = tf_vectorizer.fit_transform(df['sentence'])
```

`CountVectorizer` processes the text such that `tf` is a sparse matrix containing the count of words in each document. One document in the Brown corpus is the following sentence: 
>Mrs. Robert O. Spurdle is chairman of the committee , which includes Mrs. James A. Moody , Mrs. Frank C. Wilkinson , Mrs. Ethel Coles , Mrs. Harold G. Lacy , Mrs. Albert W. Terry , Mrs. Henry M. Chance , 2d , Mrs. Robert O. Spurdle , Jr. , Mrs. Harcourt N. Trimble , Jr. , Mrs. John A. Moller , Mrs. Robert Zeising , Mrs. William G. Kilhour , Mrs. Hughes Cauffman , Mrs. John L. Baringer and Mrs. Clyde Newman .

Via the `CountVectorizer` the stop words, punctuation, and very low frequency words have been removed. This yeilds the words and their counts listed below and visualized in the word cloud. 

```python
{'2d': 1, 'albert': 1, 'baringer': 1, 'cauffman': 1, 'chairman': 1, 'chance': 1, 'clyde': 1, 'coles': 1, 
 'committee': 1, 'ethel': 1, 'frank': 1, 'harcourt': 1, 'harold': 1, 'henry': 1, 'hughes': 1, 'includes': 1, 
 'james': 1, 'john': 2, 'jr': 2, 'kilhour': 1, 'lacy': 1, 'moller': 1, 'moody': 1, 'mrs': 15, 'newman': 1, 
 'robert': 3, 'spurdle': 2, 'terry': 1, 'trimble': 1, 'wilkinson': 1, 'william': 1, 'zeising': 1}

```

![Word cloud visualization, where the size of the word is relative to its frequency in a sentence, of "Mrs. Robert O. Spurdle is chairman of the committee , which includes Mrs. James A. Moody , Mrs. Frank C. Wilkinson , Mrs. Ethel Coles , Mrs. Harold G. Lacy , Mrs. Albert W. Terry , Mrs. Henry M. Chance , 2d , Mrs. Robert O. Spurdle , Jr. , Mrs. Harcourt N. Trimble , Jr. , Mrs. John A. Moller , Mrs. Robert Zeising , Mrs. William G. Kilhour , Mrs. Hughes Cauffman , Mrs. John L. Baringer and Mrs. Clyde Newman ."](../images/countvect_wordcloud.png?)


[<<< Previous](unsupervised.md) | [Next >>>](lda.md)
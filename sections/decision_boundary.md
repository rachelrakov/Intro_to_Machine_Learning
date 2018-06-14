Appendix: Visualize the decision boundary
================================

This appendix walks through the decision boundary visualizations found in the discussion of [supervised_classification](supervised_classification.md)

First we go through the steps of feeding the data into the algorithm because we will need the attributes of the model for the visualization. 

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

df = pd.read_csv("df_news_romance.csv")
fv = df[["NN", "JJ"]]

X_train, X_test, y_train, y_test = train_test_split(fv, df['label'],
                                                stratify=df['label'], 
                                                test_size=0.25,
                                                random_state = 42)


classifier = LinearSVC(random_state=42)
classifier.fit(X_train, y_train)
```



To visualize the decision boundary, we plot a set of contours. Any points within contours belonging to a shared color family, for example shades of blue, are assigned to the class denoted by that color. A lighter color indicates less certainity that the point belongs in that class. In this example, news is blue and romance is orange. This example only has 2 features; to visualize datasets with more often requires using a dimension reduction algorithim to find a 2D representation of the data. 

We make `plot_boundary` a function because we will reuse it on the test data. `ListedColormap` and accessing colors using colormap indexing (`mcm.tab20c(1)`) is used to match the background colors to the colors we have been using for the feature vector. We then use the `Normalize` function to center the colors around the decision boundary. You can also use an out of the box colormap, listed at https://matplotlib.org/examples/color/colormaps_reference.html


```python
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import numpy as np
np.random.seed(42)

tabcm = mcolors.ListedColormap([mcm.tab20c(0), mcm.tab20c(1), mcm.tab20c(2), mcm.tab20c(3),
                                mcm.tab20c(7), mcm.tab20c(6), mcm.tab20c(5), mcm.tab20c(4)])
# orange - positive, blue - negative
norm = mcolors.BoundaryNorm([-7, -5, -3, -1, 0,  1, 3, 5, 7], ncolors=tabcm.N) 

def jitter(arr):
    scale = .01*(arr.min() - arr.max())
    return arr + np.random.randn(arr.shape[0]) * scale

def plot_boundary(ax, clf, Xt, Xs, ys, title):
    # using all the data, create a meshgrid for the decision boundary
    h = .02
    x_min, x_max = Xt['NN'].values.min() - 4, Xt['NN'].values.max() + 1 
    y_min, y_max = Xt['JJ'].values.min() - 1, Xt['JJ'].values.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, cmap=tabcm, levels = [-7, -5, -3, -1, 0, 1, 3], 
                     norm=norm, alpha=.5)
    
    #plot the decision boundary
    ax.contour(xx, yy, Z, colors='k', linewidths=3, levels=[-1, 0, 1],
               linestyles=['--', '-', '--'], alpha=.5, zorder=20)
    
    # plot and label the data
    ax.scatter(jitter(Xs['NN'][ys=="news"]), jitter(Xs['JJ'][ys=="news"]), 
                color="tab:blue", label="news", edgecolor='k', alpha=.5)
    ax.scatter(jitter(Xs['NN'][ys=="romance"]), jitter(Xs['JJ'][ys=="romance"]), 
               color="tab:orange", label="romance", edgecolor='k', alpha=.5)
    
   

    ax.set_title(title)
    ax.set_xlabel("nouns")
    ax.set_ylabel("adjectives")
    # todo: extend xx.min to past countour
    # set the axes boundaries against the whole dataset
    ax.set_xlim(xx.min(), xx.max())
    #ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect("equal")
```

Now we call our function to make each decision boundary:

```python
%matplotlib inline
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(figsize=(15,7), nrows=2)
plot_boundary(ax1, classifier, fv, X_train, y_train, "Linear SVC: Training Data")
plot_boundary(ax2, classifier, fv, X_test, y_test, "Linear SVC: Testing Data")
fig.savefig("images/both.png", bbox_inches = 'tight', pad_inches = 0)
```

![The dark gray line in the figure is the decision boundary that the *LinearSVC* classifier found for this set of training data. All the data (dots) to the left of the gray line in the area with the orange background are classified as romance, while all the data to the right in the blue area are classified as news. The leftward skew of the classification space is due to the data being very dense and highly overlapping. Visualization of the decision boundary of the scatter plot found via the fit method. Here we have two plots: the decision boundary generated from the training data, and the testing data plotted against the decision boundary](../images/both.png)

## Generating a wordcloud from frequencies
This appendix walks through the word cloud visualization found in the discussion of [Bag of Words](bag_of_words.md) feature extraction.

`CountVectorizer` computes the frequency of each word in each document. In the Brown corpus, each sentence is fairly short and so it is fairly common for all the words to appear only once. For a word cloud, we want to find a sentence with a variety of frequencies. We convert `tf` to an array because tf is natively a sparse matix, which is not navigable in the same way as other data structures.

```python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("df_news_romance.csv")
tf_vectorizer = CountVectorizer(stop_words='english')
tf = tf_vectorizer.fit_transform(df['sentence'])
```

 We then search for documents (rows) which have words with more than 10 occurances, with a statement that returns `True` when any column (axis=1) has a value greater than 10. `get_feature_names` yields the words being counted, listed in the same order as the counts in the array.


```python
import numpy as np
tf_mat = tf.toarray()
docs = tf_mat[(tf_mat>10).any(axis=1)]
words = np.array(tf_vectorizer.get_feature_names())
```

`CountVectorizer` assigns every word in the corpus to a column in the array. We want to only look at the words that occur in the document, so we ask for all values in the array `doc` that are greater than 0. `doc` is arbitrarily chosen from our set of highly variable documents


```python
doc = docs[1] 
idx = (doc>0)
doc_words = words[idx]
doc_counts = doc[doc>0]
```

We uze the `zip` method to couple the words to their counts, and then convert the collection of pairs to a dictionary using the `dict` function.


```python
frequencies = dict(zip(doc_words, doc_counts))
frequencies
```




    {'2d': 1,
     'albert': 1,
     'baringer': 1,
     'cauffman': 1,
     'chairman': 1,
     'chance': 1,
     'clyde': 1,
     'coles': 1,
     'committee': 1,
     'ethel': 1,
     'frank': 1,
     'harcourt': 1,
     'harold': 1,
     'henry': 1,
     'hughes': 1,
     'includes': 1,
     'james': 1,
     'john': 2,
     'jr': 2,
     'kilhour': 1,
     'lacy': 1,
     'moller': 1,
     'moody': 1,
     'mrs': 15,
     'newman': 1,
     'robert': 3,
     'spurdle': 2,
     'terry': 1,
     'trimble': 1,
     'wilkinson': 1,
     'william': 1,
     'zeising': 1}



We can also find out what row these frequencies come from so that we can compare to the orginal document. `(tf_mat>10).any(axis=1)` is `True` whenever any column in a row has a value greater than 10, and `nonzero` returns the position of `True` values. We then do a little unpacking and grab the element at position `1` because we took the document at position `1` from the docs matrix. We then select the row in our dataframe at the same position to get the original sentence.


```python
doc_id = (tf_mat>10).any(axis=1).nonzero()[0][1]
df['sentence'][doc_id]
```




    "['Mrs.', 'Robert', 'O.', 'Spurdle', 'is', 'chairman', 'of', 'the', 'committee', ',', 'which', 'includes', 'Mrs.', 'James', 'A.', 'Moody', ',', 'Mrs.', 'Frank', 'C.', 'Wilkinson', ',', 'Mrs.', 'Ethel', 'Coles', ',', 'Mrs.', 'Harold', 'G.', 'Lacy', ',', 'Mrs.', 'Albert', 'W.', 'Terry', ',', 'Mrs.', 'Henry', 'M.', 'Chance', ',', '2d', ',', 'Mrs.', 'Robert', 'O.', 'Spurdle', ',', 'Jr.', ',', 'Mrs.', 'Harcourt', 'N.', 'Trimble', ',', 'Jr.', ',', 'Mrs.', 'John', 'A.', 'Moller', ',', 'Mrs.', 'Robert', 'Zeising', ',', 'Mrs.', 'William', 'G.', 'Kilhour', ',', 'Mrs.', 'Hughes', 'Cauffman', ',', 'Mrs.', 'John', 'L.', 'Baringer', 'and', 'Mrs.', 'Clyde', 'Newman', '.']"



To make the wordcloud, we are going to use a special purpose library called [WordCloud](https://github.com/amueller/word_cloud/tree/c6a58531efacda4b1b40d613bf45f494b2077ed4) to visualize the frequency of the vectorized words. Here, we generate our wordcloud directly from the frequencies we computed above. 


```python
from wordcloud import WordCloud

wordcloud = WordCloud(background_color='white').fit_words(frequencies)
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
fig, ax = plt.subplots(figsize=(15,15))
_ = ax.imshow(wordcloud, interpolation='bilinear')
_ = ax.axis("off")
fig.savefig("images/countvect_wordcloud.png", bbox_inches = 'tight', pad_inches = 0)
```

![Word cloud visualization, where the size of the word is relative to its frequency in a sentence, of "Mrs. Robert O. Spurdle is chairman of the committee , which includes Mrs. James A. Moody , Mrs. Frank C. Wilkinson , Mrs. Ethel Coles , Mrs. Harold G. Lacy , Mrs. Albert W. Terry , Mrs. Henry M. Chance , 2d , Mrs. Robert O. Spurdle , Jr. , Mrs. Harcourt N. Trimble , Jr. , Mrs. John A. Moller , Mrs. Robert Zeising , Mrs. William G. Kilhour , Mrs. Hughes Cauffman , Mrs. John L. Baringer and Mrs. Clyde Newman ."](../images/countvect_wordcloud.png?)

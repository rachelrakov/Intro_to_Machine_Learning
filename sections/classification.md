# What Is Classification? 

Let's show an example of classification using fruit!

## Example: Fruit

How would you describe apples to a computer?  How would they differ from oranges?

Remember, computers can only really understand numbers, true false values, and strings within a predefined set

![Table of fruit frit features showing height, width, color, mass, and roundness](images/fruit3.png)

Source: Andrew Rosenberg

Our fruit test shows us everything we need to do a classification machine learning test. For each item with a *label* (apple, orange, lemon), we use a series of values to try to capture machine-understandable information about the item.  These values are a *feature representation* of the item in question.  The features themselves, as we can see above, can be numeric, true/false values, or a string in a set of predefined strings.

## What if we had a new, unknown fruit?

![fruit2](images/fruit2.png)
Source: Andrew Rosenberg

Our fruit test is an example of a *classification* task.  Classification allows you to predict a *categorical* value.  This is a type of *supervised* machine learning, meaning we know the labels ahead of time and can give them to the machine learning algorithm so that it can be trained to knows what the categories of our data are.  This way, when it comes time to give the previously algorithm previously unseen data, it knows which categories it's looking for.


## Getting Our Data

Let's get to coding!

We are going to *classify* two different sets of sentences from very different source material in the Brown corpus: one set of sentences from a corpus of news text, and the other set of sentences from a corpus of romance novel text. 

```python
from nltk.corpus import brown
```

For a list of categories in the Brown corpus, use the following code

```python
for cat in brown.categories():
    print (cat)
```

    adventure
    belles_lettres
    editorial
    fiction
    government
    hobbies
    humor
    learned
    lore
    mystery
    news
    religion
    reviews
    romance
    science_fiction
    

## Get the sentences from each corpus


```python
news_sent = brown.sents(categories=["news"])
romance_sent = brown.sents(categories=["romance"])
```

## Take a look at the first 5 sentences in each corpus 


```python
print (news_sent[:5])
print ()
print (romance_sent[:5])

```

    [['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', 'Friday', 'an', 'investigation', 'of', "Atlanta's", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.'], ['The', 'jury', 'further', 'said', 'in', 'term-end', 'presentments', 'that', 'the', 'City', 'Executive', 'Committee', ',', 'which', 'had', 'over-all', 'charge', 'of', 'the', 'election', ',', '``', 'deserves', 'the', 'praise', 'and', 'thanks', 'of', 'the', 'City', 'of', 'Atlanta', "''", 'for', 'the', 'manner', 'in', 'which', 'the', 'election', 'was', 'conducted', '.'], ['The', 'September-October', 'term', 'jury', 'had', 'been', 'charged', 'by', 'Fulton', 'Superior', 'Court', 'Judge', 'Durwood', 'Pye', 'to', 'investigate', 'reports', 'of', 'possible', '``', 'irregularities', "''", 'in', 'the', 'hard-fought', 'primary', 'which', 'was', 'won', 'by', 'Mayor-nominate', 'Ivan', 'Allen', 'Jr.', '.'], ['``', 'Only', 'a', 'relative', 'handful', 'of', 'such', 'reports', 'was', 'received', "''", ',', 'the', 'jury', 'said', ',', '``', 'considering', 'the', 'widespread', 'interest', 'in', 'the', 'election', ',', 'the', 'number', 'of', 'voters', 'and', 'the', 'size', 'of', 'this', 'city', "''", '.'], ['The', 'jury', 'said', 'it', 'did', 'find', 'that', 'many', 'of', "Georgia's", 'registration', 'and', 'election', 'laws', '``', 'are', 'outmoded', 'or', 'inadequate', 'and', 'often', 'ambiguous', "''", '.']]
    
    [['They', 'neither', 'liked', 'nor', 'disliked', 'the', 'Old', 'Man', '.'], ['To', 'them', 'he', 'could', 'have', 'been', 'the', 'broken', 'bell', 'in', 'the', 'church', 'tower', 'which', 'rang', 'before', 'and', 'after', 'Mass', ',', 'and', 'at', 'noon', ',', 'and', 'at', 'six', 'each', 'evening', '--', 'its', 'tone', ',', 'repetitive', ',', 'monotonous', ',', 'never', 'breaking', 'the', 'boredom', 'of', 'the', 'streets', '.'], ['The', 'Old', 'Man', 'was', 'unimportant', '.'], ['Yet', 'if', 'he', 'were', 'not', 'there', ',', 'they', 'would', 'have', 'missed', 'him', ',', 'as', 'they', 'would', 'have', 'missed', 'the', 'sounds', 'of', 'bees', 'buzzing', 'against', 'the', 'screen', 'door', 'in', 'early', 'June', ';', ';'], ['or', 'the', 'smell', 'of', 'thick', 'tomato', 'paste', '--', 'the', 'ripe', 'smell', 'that', 'was', 'both', 'sweet', 'and', 'sour', '--', 'rising', 'up', 'from', 'aluminum', 'trays', 'wrapped', 'in', 'fly-dotted', 'cheesecloth', '.']]
    

## What do you notice about the format of the data above?
Each sentence is already *tokenized* - split into a series of word and punctuation stringes, with whitespace removed. This saves us the time of having to do all of this work ourselves!

To start to organize our data, let's put these sentences into a pandas *DataFrame*, an object which has a format very similar to an Excel spreadsheet.  We will first make two spread sheets (one for news, and one for romance), and then combine them into one.  We will also add the category each sentences came from, which will be our *labels* for each sentence and its associated feature representation (which we will build ourselves).


```python
ndf = pd.DataFrame({'sentence': news_sent,
                    'label':'news'})
rdf = pd.DataFrame({'sentence':romance_sent, 
                    'label':'romance'})
```


```python
# combining two spreadsheets into 1
df = pd.concat([ndf, rdf])
```

Let's see what this DataFrame looks like!


```python
df 
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>sentence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>news</td>
      <td>[The, Fulton, County, Grand, Jury, said, Frida...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>news</td>
      <td>[The, jury, further, said, in, term-end, prese...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>news</td>
      <td>[The, September-October, term, jury, had, been...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>news</td>
      <td>[``, Only, a, relative, handful, of, such, rep...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>news</td>
      <td>[The, jury, said, it, did, find, that, many, o...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>news</td>
      <td>[It, recommended, that, Fulton, legislators, a...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>news</td>
      <td>[The, grand, jury, commented, on, a, number, o...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>news</td>
      <td>[Merger, proposed]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>news</td>
      <td>[However, ,, the, jury, said, it, believes, ``...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>news</td>
      <td>[The, City, Purchasing, Department, ,, the, ju...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>news</td>
      <td>[It, urged, that, the, city, ``, take, steps, ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>news</td>
      <td>[Implementation, of, Georgia's, automobile, ti...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>news</td>
      <td>[It, urged, that, the, next, Legislature, ``, ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>news</td>
      <td>[The, grand, jury, took, a, swipe, at, the, St...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>news</td>
      <td>[``, This, is, one, of, the, major, items, in,...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>news</td>
      <td>[The, jurors, said, they, realize, ``, a, prop...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>news</td>
      <td>[Nevertheless, ,, ``, we, feel, that, in, the,...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>news</td>
      <td>[``, Failure, to, do, this, will, continue, to...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>news</td>
      <td>[The, jury, also, commented, on, the, Fulton, ...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>news</td>
      <td>[Wards, protected]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>news</td>
      <td>[The, jury, said, it, found, the, court, ``, h...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>news</td>
      <td>[``, These, actions, should, serve, to, protec...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>news</td>
      <td>[Regarding, Atlanta's, new, multi-million-doll...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>news</td>
      <td>[The, jury, did, not, elaborate, ,, but, it, a...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>news</td>
      <td>[Ask, jail, deputies]</td>
    </tr>
    <tr>
      <th>25</th>
      <td>news</td>
      <td>[On, other, matters, ,, the, jury, recommended...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>news</td>
      <td>[Four, additional, deputies, be, employed, at,...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>news</td>
      <td>[(, 2, )]</td>
    </tr>
    <tr>
      <th>28</th>
      <td>news</td>
      <td>[Fulton, legislators, ``, work, with, city, of...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>news</td>
      <td>[The, jury, praised, the, administration, and,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4401</th>
      <td>romance</td>
      <td>[Let's, make, it, moonlight, and, the, call, o...</td>
    </tr>
    <tr>
      <th>4402</th>
      <td>romance</td>
      <td>[Ticonderoga, folded, a, few, minutes, too, so...</td>
    </tr>
    <tr>
      <th>4403</th>
      <td>romance</td>
      <td>[We've, got, rid, of, the, steam, yachts, and,...</td>
    </tr>
    <tr>
      <th>4404</th>
      <td>romance</td>
      <td>[Why, not, come, down, smartly, in, the, world...</td>
    </tr>
    <tr>
      <th>4405</th>
      <td>romance</td>
      <td>[He, swayed, them, somewhat, ,, but, the, deba...</td>
    </tr>
    <tr>
      <th>4406</th>
      <td>romance</td>
      <td>[Financing, emerged, as, the, main, obstacle, .]</td>
    </tr>
    <tr>
      <th>4407</th>
      <td>romance</td>
      <td>[Mr., Willis, made, it, evident, that, he, had...</td>
    </tr>
    <tr>
      <th>4408</th>
      <td>romance</td>
      <td>[``, Nobody, will, underwrite, it, ,, I'm, tel...</td>
    </tr>
    <tr>
      <th>4409</th>
      <td>romance</td>
      <td>[``, I, know, what, I'm, talking, about, in, t...</td>
    </tr>
    <tr>
      <th>4410</th>
      <td>romance</td>
      <td>[``, There's, plenty, of, risk, money, '', ,, ...</td>
    </tr>
    <tr>
      <th>4411</th>
      <td>romance</td>
      <td>[``, All, right, '', ,, William, said, .]</td>
    </tr>
    <tr>
      <th>4412</th>
      <td>romance</td>
      <td>[``, We'll, try, to, swing, the, deal, on, tha...</td>
    </tr>
    <tr>
      <th>4413</th>
      <td>romance</td>
      <td>[If, we, can't, raise, the, capital, ,, we're,...</td>
    </tr>
    <tr>
      <th>4414</th>
      <td>romance</td>
      <td>[Nothing, has, been, lost, .]</td>
    </tr>
    <tr>
      <th>4415</th>
      <td>romance</td>
      <td>[You're, up, against, it, anyhow, .]</td>
    </tr>
    <tr>
      <th>4416</th>
      <td>romance</td>
      <td>[Why, won't, you, give, me, a, chance, '', ?, ?]</td>
    </tr>
    <tr>
      <th>4417</th>
      <td>romance</td>
      <td>[A, silence, fell, .]</td>
    </tr>
    <tr>
      <th>4418</th>
      <td>romance</td>
      <td>[Heads, instinctively, turned, in, Willis', di...</td>
    </tr>
    <tr>
      <th>4419</th>
      <td>romance</td>
      <td>[He, smiled, at, William, and, slowly, rubbed,...</td>
    </tr>
    <tr>
      <th>4420</th>
      <td>romance</td>
      <td>[``, I, feel, I, must, answer, the, question, ...</td>
    </tr>
    <tr>
      <th>4421</th>
      <td>romance</td>
      <td>[I'm, not, giving, you, a, chance, ,, Bill, ,,...</td>
    </tr>
    <tr>
      <th>4422</th>
      <td>romance</td>
      <td>[Good, luck, to, you, '', .]</td>
    </tr>
    <tr>
      <th>4423</th>
      <td>romance</td>
      <td>[``, All, the, in-laws, have, got, to, have, t...</td>
    </tr>
    <tr>
      <th>4424</th>
      <td>romance</td>
      <td>[Sweat, started, out, on, William's, forehead,...</td>
    </tr>
    <tr>
      <th>4425</th>
      <td>romance</td>
      <td>[Across, the, table, ,, Hamrick, saluted, him,...</td>
    </tr>
    <tr>
      <th>4426</th>
      <td>romance</td>
      <td>[Nobody, else, showed, pleasure, .]</td>
    </tr>
    <tr>
      <th>4427</th>
      <td>romance</td>
      <td>[Spike-haired, ,, burly, ,, red-faced, ,, deck...</td>
    </tr>
    <tr>
      <th>4428</th>
      <td>romance</td>
      <td>[``, Hello, ,, boss, '', ,, he, said, ,, and, ...</td>
    </tr>
    <tr>
      <th>4429</th>
      <td>romance</td>
      <td>[``, I, suppose, I, can, never, expect, to, ca...</td>
    </tr>
    <tr>
      <th>4430</th>
      <td>romance</td>
      <td>[``, I'm, afraid, not, '', .]</td>
    </tr>
  </tbody>
</table>
<p>9054 rows × 2 columns</p>
</div>




```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>sentence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>news</td>
      <td>[The, Fulton, County, Grand, Jury, said, Frida...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>news</td>
      <td>[The, jury, further, said, in, term-end, prese...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>news</td>
      <td>[The, September-October, term, jury, had, been...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>news</td>
      <td>[``, Only, a, relative, handful, of, such, rep...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>news</td>
      <td>[The, jury, said, it, did, find, that, many, o...</td>
    </tr>
  </tbody>
</table>
</div>



### So how many labels do we have?


```python
df['label'].value_counts()
```




    news       4623
    romance    4431
    Name: label, dtype: int64



### What if we want to visualize that information?


```python
fig, ax = plt.subplots()
_ = df['label'].value_counts().plot.bar(ax=ax)

```


![png](output_28_0.png)


We have slightly more news data than romance data, which we should keep in mind as we go ahead with classification.

## What should we use as features for our data set?  What did we use as features for our fruit example before?
![fruit3](images/fruit3.png)

## Now that we are using sentences, how can we best repersent each sentence as a series of values?

One idea is to use how many particular *parts of speech* the sentence contains.

- Nouns: Most basically described as a person, place, or thing.  Counting nouns can help determine how many topics are being discussed in a sentence.
- Adjectives: Descriptors of nouns (eg. "yellow", "angry", "charming").  Counting adjectives can help determine how often descriptive words are being added to nouns, which can demonstrate writing style.
- Adverbs: Descriptors of verbs (eg. "quickly", "hungrily", "annoyingly").  Counting adverbs can help determine how often the manner of the verb is modified, which can also demonstrate writing style.

#### Why might we want to use these parts of speech to distinguish between news sentences and romance sentences?


We will now compute all of the parts of speech on each sentence (row) in our dataframe.


```python
# compute parts of speech on each sentence (row)
pos_all = pos_tag_sents(df['sentence'])
```


```python
print (pos_all[:5])
```

    [[('The', 'DT'), ('Fulton', 'NNP'), ('County', 'NNP'), ('Grand', 'NNP'), ('Jury', 'NNP'), ('said', 'VBD'), ('Friday', 'NNP'), ('an', 'DT'), ('investigation', 'NN'), ('of', 'IN'), ("Atlanta's", 'NNP'), ('recent', 'JJ'), ('primary', 'JJ'), ('election', 'NN'), ('produced', 'VBD'), ('``', '``'), ('no', 'DT'), ('evidence', 'NN'), ("''", "''"), ('that', 'IN'), ('any', 'DT'), ('irregularities', 'NNS'), ('took', 'VBD'), ('place', 'NN'), ('.', '.')], [('The', 'DT'), ('jury', 'NN'), ('further', 'RB'), ('said', 'VBD'), ('in', 'IN'), ('term-end', 'JJ'), ('presentments', 'NNS'), ('that', 'IN'), ('the', 'DT'), ('City', 'NNP'), ('Executive', 'NNP'), ('Committee', 'NNP'), (',', ','), ('which', 'WDT'), ('had', 'VBD'), ('over-all', 'JJ'), ('charge', 'NN'), ('of', 'IN'), ('the', 'DT'), ('election', 'NN'), (',', ','), ('``', '``'), ('deserves', 'VBZ'), ('the', 'DT'), ('praise', 'NN'), ('and', 'CC'), ('thanks', 'NNS'), ('of', 'IN'), ('the', 'DT'), ('City', 'NNP'), ('of', 'IN'), ('Atlanta', 'NNP'), ("''", "''"), ('for', 'IN'), ('the', 'DT'), ('manner', 'NN'), ('in', 'IN'), ('which', 'WDT'), ('the', 'DT'), ('election', 'NN'), ('was', 'VBD'), ('conducted', 'VBN'), ('.', '.')], [('The', 'DT'), ('September-October', 'NNP'), ('term', 'NN'), ('jury', 'NN'), ('had', 'VBD'), ('been', 'VBN'), ('charged', 'VBN'), ('by', 'IN'), ('Fulton', 'NNP'), ('Superior', 'NNP'), ('Court', 'NNP'), ('Judge', 'NNP'), ('Durwood', 'NNP'), ('Pye', 'NNP'), ('to', 'TO'), ('investigate', 'VB'), ('reports', 'NNS'), ('of', 'IN'), ('possible', 'JJ'), ('``', '``'), ('irregularities', 'NNS'), ("''", "''"), ('in', 'IN'), ('the', 'DT'), ('hard-fought', 'JJ'), ('primary', 'NN'), ('which', 'WDT'), ('was', 'VBD'), ('won', 'VBN'), ('by', 'IN'), ('Mayor-nominate', 'NNP'), ('Ivan', 'NNP'), ('Allen', 'NNP'), ('Jr.', 'NNP'), ('.', '.')], [('``', '``'), ('Only', 'RB'), ('a', 'DT'), ('relative', 'JJ'), ('handful', 'NN'), ('of', 'IN'), ('such', 'JJ'), ('reports', 'NNS'), ('was', 'VBD'), ('received', 'VBN'), ("''", "''"), (',', ','), ('the', 'DT'), ('jury', 'NN'), ('said', 'VBD'), (',', ','), ('``', '``'), ('considering', 'VBG'), ('the', 'DT'), ('widespread', 'JJ'), ('interest', 'NN'), ('in', 'IN'), ('the', 'DT'), ('election', 'NN'), (',', ','), ('the', 'DT'), ('number', 'NN'), ('of', 'IN'), ('voters', 'NNS'), ('and', 'CC'), ('the', 'DT'), ('size', 'NN'), ('of', 'IN'), ('this', 'DT'), ('city', 'NN'), ("''", "''"), ('.', '.')], [('The', 'DT'), ('jury', 'NN'), ('said', 'VBD'), ('it', 'PRP'), ('did', 'VBD'), ('find', 'VB'), ('that', 'IN'), ('many', 'JJ'), ('of', 'IN'), ("Georgia's", 'NNP'), ('registration', 'NN'), ('and', 'CC'), ('election', 'NN'), ('laws', 'NNS'), ('``', '``'), ('are', 'VBP'), ('outmoded', 'VBN'), ('or', 'CC'), ('inadequate', 'JJ'), ('and', 'CC'), ('often', 'RB'), ('ambiguous', 'JJ'), ("''", "''"), ('.', '.')]]
    

## What's with those part of speech labels?  They aren't helpful at all!
The Penn Tagset, which NLTK uses for it's part of speech tagger, is not particularly intuitive.  Fortunately, they provide code that allows you to check what different tags stand for.


```python
nltk.help.upenn_tagset("NN")
nltk.help.upenn_tagset("JJ")
nltk.help.upenn_tagset("RB")
```

    NN: noun, common, singular or mass
        common-carrier cabbage knuckle-duster Casino afghan shed thermostat
        investment slide humour falloff slick wind hyena override subhumanity
        machinist ...
    JJ: adjective or numeral, ordinal
        third ill-mannered pre-war regrettable oiled calamitous first separable
        ectoplasmic battery-powered participatory fourth still-to-be-named
        multilingual multi-disciplinary ...
    RB: adverb
        occasionally unabatingly maddeningly adventurously professedly
        stirringly prominently technologically magisterially predominately
        swiftly fiscally pitilessly ...
    



Now we know the tags for the different parts of speech we want to count in each sentence.  Let's now write a function that will count the parts of speech to us, when given a part of speech tagged sentence (such as we have already in our DataFrame) and the part of speech we want to count (for example, "NN" to count the number of nouns in the sentence).


```python
def countPOS(pos_tag_sent, POS):
    pos_count = 0
    all_pos_counts = []
    for sentence in pos_tag_sent:
        for word in sentence:
            tag = word[1]
            if tag [:2] == POS:  
                pos_count = pos_count+1
        all_pos_counts.append(pos_count)
        pos_count = 0
    return all_pos_counts
```

We will now call this function three different times, one for each of the parts of speech we are counting.  As we finish counting them, we put the results into the DataFrame, saving us the trouble of having to do so later.


```python
df['NN'] = countPOS(pos_all, 'NN')
df['JJ'] = countPOS(pos_all, "JJ")
df['RB'] = countPOS(pos_all, "RB")
```


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>sentence</th>
      <th>NN</th>
      <th>JJ</th>
      <th>RB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>news</td>
      <td>[The, Fulton, County, Grand, Jury, said, Frida...</td>
      <td>11</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>news</td>
      <td>[The, jury, further, said, in, term-end, prese...</td>
      <td>13</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>news</td>
      <td>[The, September-October, term, jury, had, been...</td>
      <td>16</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>news</td>
      <td>[``, Only, a, relative, handful, of, such, rep...</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>news</td>
      <td>[The, jury, said, it, did, find, that, many, o...</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>sentence</th>
      <th>NN</th>
      <th>JJ</th>
      <th>RB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4426</th>
      <td>romance</td>
      <td>[Nobody, else, showed, pleasure, .]</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4427</th>
      <td>romance</td>
      <td>[Spike-haired, ,, burly, ,, red-faced, ,, deck...</td>
      <td>9</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4428</th>
      <td>romance</td>
      <td>[``, Hello, ,, boss, '', ,, he, said, ,, and, ...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4429</th>
      <td>romance</td>
      <td>[``, I, suppose, I, can, never, expect, to, ca...</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4430</th>
      <td>romance</td>
      <td>[``, I'm, afraid, not, '', .]</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### So how many features do we have?


```python
df.groupby('label').sum()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NN</th>
      <th>JJ</th>
      <th>RB</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>news</th>
      <td>31593</td>
      <td>6678</td>
      <td>2935</td>
    </tr>
    <tr>
      <th>romance</th>
      <td>13821</td>
      <td>4022</td>
      <td>3570</td>
    </tr>
  </tbody>
</table>
</div>



## Let's visualize this data!
What do you notice about the data when we look visualize it?  Do you think our features will be good at predicting news and romance sentences?  Which features do you think will be the most useful?


```python
fig, (ax1,ax2) = plt.subplots(ncols=2,  figsize=(10,5))
_ = df.groupby('label').sum().plot.bar(ax=ax1)
_ = df.groupby('label').sum().T.plot.bar(ax=ax2, color=['gray','hotpink'])

```


![png](output_46_0.png)


### Save the dataframe to your computer as a csv file (comma separated value)



```python
df.to_csv("path/filename.csv")
```


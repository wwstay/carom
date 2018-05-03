==========================================
The theory behind the model of Match-Maker
==========================================

**Match-Maker** is an entity-resolution/deduplication algorithm. Given two tables of hotel data including name,address,city, lat,longs this algorithm matches table 1 records to table 2 records. This algorithm has incorporated a probability based method provided by this paper http://wwwconference.org/proceedings/www2014/proceedings/p409.pdf

The programme consists of the following steps:

* Cleaning and preparation of the data
    * Removing punctuation and stopwords
    * Stemming words
    * Decoding unicode and lowercasing
    * Tokenization

* Generating Probabilities
    * EM Algorithm
    * p(w)-- Probability that the word (w) is a core

* Comparison methods
    * Probability based
    * Edit distance based
    * Abbreviation based
    * Cossine similarity based
    * Lat-Long based

If you have questions/comments feel free to reach me at: sgoswam3@illinois.edu

Cleaning and preparation of the data
------------------------------------

The data present in MongoDB has to be cleaned and formatted effectively before we can do anything with it. The following are the methods employed in this programme for setting up the dataset. 
The file responsible for this is ``prepare_4_EM.py``


Removing punctuation and stopwords
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stopwords are words that we believe are not useful for our match making. These words include, "that,is,of, etc." filler words provided by the `nltk package <https://www.nltk.org/api/nltk.html>`_. We use stop words from French, German and English. Along with stopwords punctuations are removed.

.. literalinclude:: src/prepare_4_EM.py 
    :language: python
    :lines: 153-168

Stemming words
^^^^^^^^^^^^^^

There are many words written in different ways. Such as *road* has other forms such as *rd*, *roads*, *rds* and we need to tell the computer that these words are the same. Using `United States Postal Service's (USPS) street suffix abbreviations <http://pe.usps.com/text/pub28/28apc_002.htm>`_ and `secondary unit designations <http://pe.usps.com/text/pub28/28apc_003.htm>`_ we were able to find the necessary abbreviations or stemmed formats for plenty of words. This helped reduce the lexicon of our programme and make matching better.

A dictionary was created using 

.. literalinclude:: src/genstem.py
	:language: python


Decoding Unicode and lowercasing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A lot of hotel names are addresses in foreign countries have characters that are unicode. Using unidecoder, we decode them into ASCII. Also everything is converted to lowercase to remove case sensitivity of any table. 

.. literalinclude:: src/prepare_4_EM.py
    :language: python
    :lines: 213-233


Tokenization
^^^^^^^^^^^^

After all the cleaning, each word was seperated from sentences of hotel names/address. Each word becomes a token for comparison. This makes the original dataframe of names and addresses into a data frame where these columns are lists of lists of cleaned data.
The implementation of tokenizing and stemming is given by

.. literalinclude:: src/prepare_4_EM.py
    :language: python
    :lines: 70-90


Generating Probabilities
------------------------

The core idea of the match making method is that each hotel name or address, would have core words (like *Best* and *Western* in `Best Western Hotel Sangria Hill` or *Radison* in `Hotel Radison Delhi`) and background words (like *Hotel* or *inn* that is common in many hotel names). If two names are to match, their core words have to match and those that don't match should be background words.

We assume that core words and background words follow a certain probability distribution and these words we see in our table are samples from them. Thus, we say that,
The distribution of all core words is **P(C)**.
The distribution of all background words is **P(B)**.

Our goal is to find these distributions from the data given to us i.e. for each table.


EM Algorithm
^^^^^^^^^^^^

EM stands for Estimation Maximization. This algorithm is useful for unsupervised clustering of data. EM will help us find the two distributions. 

Naively, this algorithm works like this:

We start with a uniform distribution for both P(C) and P(B). From the guess of uniform we label the data belonging to cluster P(C) or P(B). We update the parameters of P(C) and P(B) until convergence, i.e. until the distribution of the data is accurately captured by P(C) and P(B) (they have been sufficiently clustered). 

More info can be found online.

Once we get these distributions we move on to generating whether a word is core or not.

The implementation of the EM algorithm is done in CoreBGgeneration.py

.. literalinclude:: src/CoreBGgeneration.py
    :language: python
    :lines: 88-153

p(w)-- Probability that the word (w) is a core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After getting the distributions P(C) and P(B) (hereon referred to as C and B respectively), for each word we formulate a probability that whether that word is a core word or not, as equation


.. math::
	:label: coreprob
	
	p(w) = \frac{\alpha \times C(w)}{\alpha\times C(w) + (1-\alpha)\times B(w)}

:math: `\alpha` gives a weightage to how much favour should we give core words over background words. If :math: `\alpha=1`, then we make the assertion that all words are core. Similarly, if  :math: `\alpha=0`, then we make the assertion that all words are part of background.

This p(w) will be crucial while making judgements whether two properties are same or not

The implementation for this is at ``CoreBGgeneration.py``

.. literalinclude:: src/CoreBGgeneration.py
    :language: python
    :lines: 377-397




Comparison methods
------------------

We use more than one comparison model. Each adding on to the other, in hopes to increase accuracy.


Probability based
^^^^^^^^^^^^^^^^^

For a match to happen, two things should occur. 

1. The words that are in common should either both be core words or both be background words. The way that is written in terms of probability is

.. math:: p_1(w)*p_2(w) + (1-p_1(w))(1-p_2(w))
	:label: assert_same_words

Here p1 and p2 are p(w) from the two different tables we are comparing.

2. The words that are not in common should be asserted to be part of background. So each of these words will get probability written in the form

.. math:: 1-p(w)
	:label: assert_diff_words

Here, the p(w) would belong to the table from which the words that were uncommon belong to. So if in between hotel Chandrika from table 1 and Chandrika from table 2, p(w) would be p_1(w) and vice versa for the other case

The implementation is in ``CoreBGgeneration.py``

.. literalinclude:: src/CoreBGgeneration.py
    :language: python
    :lines: 288-344


Edit distance based
^^^^^^^^^^^^^^^^^^^

Two entities could refer to the same thing if they have slight spelling errors in their name. (hotel tennessey and hotel tenesse). So the edit distance allows us to put a metric on how many characters do we need to change to get the other sentence.

If it is not too many (<20-25% of the characters) then we could assume that a spelling mistake occurred and assert that those two words are same. And what follows is the same probabilistic model added treating them as words in common.

Along with word by word comparison, we have included a whole concatenated string edit distance comparison. So names such as northwest and north west, would be considered the same. The % of characters kept is added to the probability. 

.. note:: a direct sum of % of characters kept is added. There might be a better way; such as incorporating each word as common and using the probabilistic method mentioned above. We just stuck to this for now. 

This implementation is in ``CoreBGgeneration.py``

.. literalinclude:: src/CoreBGgeneration.py
    :language: python
    :lines: 184-223


Abbreviation based
^^^^^^^^^^^^^^^^^^

This method is optional. This makes first letter abbreviations from words such as Cafe Coffee Day to CCD and then makes comparisons. So if the two entries are 'oyo' and 'on your own' are two entries; they will be considered the same and all words will be asserted as common words in the same vein as the probabilistic model.

This implementation is in ``CoreBGgeneration.py``

.. literalinclude:: src/CoreBGgeneration.py
    :language: python
    :lines: 257-283

Cossine Similarity based
^^^^^^^^^^^^^^^^^^^^^^^^
We convert hotel names and addresses into one hot vectors each, concatenate them to create a large vector and use that to calculate the cossine similarity. One hot implies a vector that is as long as the vocabulary of the document. So, if I have these three coffee places:
1. Starbucks Coffee
2. Kalmane Coffee
3. Cafe Coffee Day

We will have a vector of length 5 (each index corresponds to a word [Starbucks,Kalmane,Cafe,Coffee,Day]) So for the first entry (Starbucks Coffee) our vector will be '1' on all indexes corresponding to the words in the name. So,
Starbucks Coffee = [1,0,0,1,0]. 

Similarly we do this one hot encoding to addresses. Then concatenate the two vectors and then compute the cosine similarity.

The cosine similarity is just given by the cos of the angle made by these vectors in their n-dimensional space. 

.. math::
    \cos(\theta) = \frac{\vec{A}\cdot\vec{B}}{|A||B|}

A and B are two vectors in this formula. This is included in the below function

This implementation is in ``Deduplication_algo.py``

.. literalinclude:: src/Deduplication_algo.py
    :language: python
    :lines: 372-393 

Lat-Long based
^^^^^^^^^^^^^^

Last but not the least, a comparison of lat-long distances. The `haversine formula <https://en.wikipedia.org/wiki/Haversine_formula>`_ is used to produce a number between 0 and 1 telling me whether how close or far it is. I subtracted it from 1 to invert the distance basically making 1 right on top of each other and a 0 very far away from it. This 0-1 number is added simply to the final probability.

.. note:: A better method might exist for adding to final probability

This implementation is in ``Deduplication_algo.py``

.. literalinclude:: src/Deduplication_algo.py
   :language: python
   :lines: 417-444 
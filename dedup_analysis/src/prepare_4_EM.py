"""
.. module:: prepare_4_EM 
   :synopsis: Data cleaning for use in match making \
   \
   
.. moduleauthor:: Shubhang Goswami <shubhang@wwstay.com>\
"""


import pandas as pd 
import numpy as np 
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import os
import cPickle as pk
import math
from unidecode import unidecode
from icu import CharsetDetector
import re


def findidx(StemWords,word):
    """
    Gets the word to replace. So if word is "road", it will replace it with "rd"

    Args:
        StemWords: A dictionary where keys are stemmed words (like "rd") and values are lists that are long/other forms of the key (like road)
        word: The word in question, whether to replace or not.

    Returns:
        str: A word, either replaced(stemmed) or the same word.
    """
    A = []
    for key,value in StemWords.items():
        if word in value:
            return key
        else:
            continue
    return word




def flatten2one(xS):
    """
    Reduces a series of lists (list of lists) down to a single list. Used as a apply function on Series data.
    
    Args:
        xS: A pandas series of lists.
    
    Returns:
        reduced single list containing all entities.
    """
    ret =[]
    for sub_list in xS.tolist():
        if isinstance(sub_list,float):
            ret.append(np.nan)
        else:
            for item in sub_list:
                ret.append(item)
    return ret
    #return [np.nan if isinstance(sub_list,float) else item for sub_list in xS.tolist() for item in sub_list]



#
def stemmer(x,stemdic):
    """
    Strip sentence into individual words, removing spaces and punctuation. 
    Then replaces words with their stemmed counterparts given by the StemWords (a dictionary of words and their stem)
    
    Runs function findidx on series row data. This is used as an apply function on Series.

    Args:
        x:  The sentence provided by a pandas series (usually hotel name or address)
        stemdic: Dictionary containing a list of words as values and their stemmed counterparts as keys

    Returns:
        list: list of tokenized words. (So a sentence becomes a list of stripped and stemmed words)

    """
    if isinstance(x,float):
        return np.nan
    tokens = x.replace('[^\w\s]','').strip().split()
    for idx,word in enumerate(tokens):
        tokens[idx] = findidx(stemdic.loc[stemdic.index.str.startswith(word[0])],word)
    return tokens    




def dumpstopwords():
    """
    Dumps stop words that are to be removed (words such as is, and, the, etc.) into a pickle file.
    
    Args:
        None

    Returns:
        None
    """
    eng_stop_words = set(stopwords.words("english")+['test','1234','various','addresses'])
    fr_stop_words = set(stopwords.words("french"))
    ger_stop_words = set(stopwords.words("german"))
    stop_words = eng_stop_words.union(fr_stop_words).union(ger_stop_words)
    pk.dump(stop_words,open("stop_words.pk","wb"))


def get_relevcols(df, colname):
    """
    Get 1 or more columns that contains a particular name. Certain tables have more than one address column in dataframe.
    This function allows you to get all columns that contain a particular name (such as address)

    Args:
        df: The dataframe that has all details of hotel properties
        colname: The name of the column you want to pick out

    Returns:
        list: A list of column names that correspond to colname.
	"""
    return [col for col in df.columns.values if colname in col.lower()]



#
def loadfilters():
    """
    Generates stop words and the stemming dictionary required to clean the data.

    Args:
        None

    Returns:
        pair: (set of stop words, stemming words dictionary)
    """
    try:
        stop_words = pk.load(open("stop_words.pk","rb"))
    except:
        dumpstopwords()
        stop_words = pk.load(open("stop_words.pk","rb"))

    stemingdic = pd.read_csv("stemming_dict.csv",header=None,index_col=False)
    stemingdic.index = stemingdic[0]
    stemingdic = stemingdic.iloc[:,1]
    return (stop_words,stemingdic)




def gramclean(x,stop_words):
    """
    Removes punctuation, stop words and cleans the sentence and gives back the whole sentence
    Used in a apply function.
    
    Args:
        x: A sentence from pandas series. (Hotel name or hotel address)
        stop_words: Set containing not useful words (is, that, the etc.)

    Returns:
        str: A string that is the cleaned sentence
    """

    if pd.isnull(x):
        return np.nan
    return ' '.join([word for word in re.sub(r"[^\w\s]","",x).split() if word not in stop_words])




#
def preclean(dat):
    """
    Removes words that have an extra 's' after like inns,hotels is stemmed to inn,hotel

    Args:
        dat: A series column containing tokenized words

    Returns:
        Series: A series with cleaned sentences.
    """
    cln_d = set(flatten2one(dat))
    def snip(x,cln):
        """
        Core part of the preclean function. Used as an apply function
        
        Args:
            x: The word/string in series row
            cln: set of all words series dat

        Returns:
            None. (Changes things inplace)
        """
        if isinstance(x,float):
            return
        if len(x)<1:
            return
        for idx,w in enumerate(x):
            if len(w)<=2 or w.isdigit():
                continue
            if w[:-1] in cln:
                x[idx]=w[:-1]

    dat.apply(snip,args=(cln_d,))
    return dat




#
def translittunicode(x):
    """
    Translitterate unicode to ASCII. Used as apply function

    Args:
        x: A unicode string

    Returns:
        str: A decoded ASCII string

    """
    if isinstance(x,float):
        return x
    else:
        try:
            x = x.decode('utf8')
        except Exception, e:
            encoding = CharsetDetector(x).detect().getName()
            x = x.decode(encoding)

    return unidecode(x)




def prepareem(prop_subdf,name):
    """
    Prepares features for the EM algorithm. The core component of this file. 
    The function cleans data, tokenizes it, changes transliterates unicode to ascii, removes stop/common words and
    builds the feature set needed for running EM.

    Args:
        prop_subdf: Pandas dataframe containing table from mongodb dump 
        name: Name of the table or supplier from which this table originated (string)

    Returns:
        dataframe: Ready to use dataframe for EM algorithm
    """
    #Load the stop words and stemming dictionary to use
    stop_words,stemingdic = loadfilters()
    
    #Clean data and tokenize
    namedat = prop_subdf[get_relevcols(prop_subdf,'name')].iloc[:,0].str.lower()
    addrdat = prop_subdf[get_relevcols(prop_subdf,'address')].iloc[:,0].str.lower()
    namedat = namedat.apply(gramclean, args=(stop_words,))
    addrdat = addrdat.apply(gramclean, args=(stop_words,))
    
    #Transliterate non ASCII characters
    namedat = namedat.apply(translittunicode)
    addrdat = addrdat.apply(translittunicode)

    #Stem certain common words
    namedat = namedat.apply(stemmer, args=(stemingdic,))
    addrdat = addrdat.apply(stemmer, args=(stemingdic,))

    idname = "mapped_to_ext_id"
    if idname not in prop_subdf:
        idname = "eanhotelid"

    #Create a feature set that will be used for classification
    emfeatures = pd.DataFrame({"name": namedat,
                 "address": addrdat,
                "city": prop_subdf[get_relevcols(prop_subdf,'city')].iloc[:,0].str.lower(),
                "latitude":prop_subdf[get_relevcols(prop_subdf,'latitude')].iloc[:,0],
                "longitude":prop_subdf[get_relevcols(prop_subdf,'longitude')].iloc[:,0],
                 idname: prop_subdf[get_relevcols(prop_subdf,idname)].iloc[:,0]
                              })
    
    #Clean and transliterate city names as well
    emfeatures.city = emfeatures.city.apply(translittunicode)
    preclean(emfeatures.name)
    preclean(emfeatures.address)

    pk.dump(emfeatures,open("emfeatures_"+name+".pk","wb"))

    return emfeatures




if __name__=='__main__':
#     client = MongoClient()
#     db = client['supplier_static_database']
#     #prepareem(db.taap_properties, "taap")
    
#     #Get dataframe from mongo
#     prop_subdf= pd.DataFrame(list(db.ean_properties.find()))
#     prop_subdf.columns = map(unicode.lower, prop_subdf.columns)
#     prepareem(prop_subdf,"ean")
    
    client = MongoClient()
    prop_lon_taap = pd.DataFrame(list(client['supplier_static_database'].taap_properties.find({"City": "London", "Country": "GBR"})))
    prop_lon_hotbed = pd.DataFrame(list(client['supplier_static_database'].hotelbeds_properties.find({"city_code": "LON"})))

    prop_lon_taap.columns = map(unicode.lower,prop_lon_taap.columns)
    prop_lon_hotbed.columns = map(unicode.lower, prop_lon_hotbed.columns)

    prop_lon_taap[~pd.isnull(prop_lon_taap.mapped_to_ext_id)].to_csv("taap_LON.scsv",sep=";",index=None,encoding="utf-8")
    prop_lon_hotbed[~pd.isnull(prop_lon_hotbed.mapped_to_ext_id)].to_csv("hotelbeds_LON.scsv",sep=";",index=None,encoding="utf-8")
















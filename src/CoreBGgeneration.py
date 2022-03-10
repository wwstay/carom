"""
.. module:: CoreBGgeneration
   :synopsis: The generator of probability distributions

.. moduleauthor:: Shubhang Goswami <shubhang@wwstay.com>\


"""
from __future__ import division
import pandas as pd
import numpy as np
import cPickle as pk
import collections
import editdistance



def flatten2one(xS):
    """
    Reduces a series of lists (list of lists) down to a single list. Used as a apply function on Series data.
    
    Args:
        xS: A list of lists.

    Returns:
        list: reduced single list containing all entities.

    """
    return [item for sub_list in xS.tolist() for item in sub_list]





def initial_probs(lis_names):
    """
    Generate initial uniform distribution. The distribution is simply the frequency
    of the occurance of the word divided by the total number of words. (Term freq.)
    
    Args:
        lis_names: List of lists of all records tokenized from table

    Returns:
        dict: A dictionary of the word as key and term-frequency prob as value. (string:float)
    """
    vocab_ctr = collections.Counter((flatten2one(lis_names)))
    prob = collections.OrderedDict()
    networds = len(flatten2one(lis_names))
    if networds==0:
        return prob
    for k,v in vocab_ctr.items():
        if isinstance(k,float):
            continue
        prob[k]=v/networds
    return prob





def word_estimation(name,C,B):
    """
    Calculate z(word). This is the probability of word belonging to core.
    For more information, read this paper: http://wwwconference.org/proceedings/www2014/proceedings/p409.pdf 
    
    Args:
        name (list): The hotel name or address tokenized into words
        C    (dict): The probability distribution of core words 
        B    (dict): The probability distribution of background words

    Returns:
        dict: A dictionary of (word,probability) pairs. keys are words, values are floats  
    """
    z=dict()
    denom=0.0
    for w in name:
        denom+=C[w]/(B[w]+1e-10)
    
    for word in name:
         z[word] = C[word]/((B[word]+1e-10)*(denom+1e-10))
    
    return z





def update_weights(Z,C,B,networds):
    """
    After getting new z(w), update the parameters/distributions C and B to better represent the data.
    This is the update step in the EM algorithm. For details: http://wwwconference.org/proceedings/www2014/proceedings/p409.pdf 

    Args:
        Z (list): the distribution containing all z.
        C (dict): the probability distribution of core words
        B (dict): the probability distribution of background words
        networds (int): Total number of words (in the whole table)

    Returns:
        pair: (c,b) containing the updated probabilitiy distribution (core, background)
    """
    c=dict()
    b=dict()
    for word in C:
        #print word
        new_c = 0.0
        new_b = 0.0
        for z in Z:
            #print z,
            if word in z:
                #print z[word]
                new_c+=z[word]
                new_b+=(1-z[word])
    
        new_c=new_c/len(Z)
        new_b=new_b/(networds-len(Z) + 1e-07)
        #print "new C({}): ".format(word), new_c
        c[word] = new_c
        b[word] = new_b
        #print "next"
    
    #print c
    return (c,b)
    




#
def core_algo1(names):
    """
    Run the EM algorithm that updates the distribution until some degree of convergence
    The convergence criteria chosen is the number of field entries.
    Starts with term frequency distribution and then updates until convergence
    
    Args:
        names: The list of list/series of lists of tokenized records that are to be deduplicated.

    Returns:
        pair: (prob_C,prob_B); converged probability distributions C (core) and B (background)
    """

    prob_C = prob_B = initial_probs(names)
    networds = len(flatten2one(names))
    for i in range(len(prob_C)):
        all_z = names.apply(word_estimation,args=(prob_C,prob_B))
        prob_C,prob_B = update_weights(all_z,prob_C,prob_B,networds)

    #Removes any nan values if it exists
    prob_C.pop(np.nan,None)
    prob_B.pop(np.nan,None)

    return prob_C,prob_B





def assrtcore(word,pofcore):
    """
    Args:
        word (str): A word
        pofcore (dict): Core probability distribution
    Returns:
        float: probabilities of being core (float)
    """
    return pofcore[word]

def assrtbg(word,pofcore):
    """
    Args:
        word (str): A word
        pofcore (dict): Core probability distribution

    Returns:
        float: probabilities of being background
    """
    return (1+1e-11-pofcore[word])




#
def distmetric(s1,s2,pr1,pr2):
    """
    Calculate edit distance, if (1-edit distance) or how many characters do we retain>75% of the length,
    then we probably have a misspelling, and we can assert that both are the same.

    Being the same implies, both are either core or both are background.

    Args:
        s1  (list): sentence from first table
        s2  (list): sentence from second table 
        pr1 (dict): core probability distribution of the words of first table
        pr2 (dict): core probability distribution of the words of second table

    Returns:
        float: the probability of being a match 
    """
    workons1 = set(s1) - set(s2)
    workons2 = set(s2) - set(s1)
    ret = 0.0

    reduceds1 = ''.join([w1 for w1 in sorted(s1)])
    reduceds2 = ''.join([w2 for w2 in sorted(s2)])
    charsimi = 1-(editdistance.eval(reduceds1,reduceds2)/max(len(reduceds1),len(reduceds2)))
    
    if charsimi>=0.8:
        ret+=charsimi

    #Misspellings would be part of the set outside the union
    for w1 in workons1:
        #filter out those words from sentence 2 that are 75% close to this particular word w1 from sentence 1 
        dat = filter(lambda x: (1-((editdistance.eval(w1,x))/max(len(x),len(w1))))>0.75, workons2)
        
        if len(dat)<1:
            continue
        
        #print "Inside distance metric: ", dat
        #The misspelled words are the same, thus both are part of core or part of background
        ret+= reduce((lambda x,y: x*y),[pr2[a] for a in dat])*pr1[w1] + reduce((lambda x,y: (1-x)*(1-y)),[pr2[a] for a in dat])*(1-pr1[w1])
    
    return ret




#
def edit_func(s1,s2,pr1,pr2):
    """
    Added edit functions that can factor into the probability. The two called in here are 
    misspelling and abbreviation. Abbreviation is currently switched off.

    Args:
        s1  (list): sentence from first table
        s2  (list): sentence from second table 
        pr1 (dict): core probability distribution of the words of first table
        pr2 (dict): core probability distribution of the words of second table

    Returns:
        float: representing probabilities of two sentences to be the same after edits.
    """

    fn=0.0

    #abbreviation probability 2 word abbrvs and 3 word abbrvs
    #dat =( abbrv_func(s1,s2,pr1,pr2,2)+abbrv_func(s1,s2,pr1,pr2,3) )
    #fn+=dat
    
    #Probability from misspelling
    fn = fn+distmetric(s1,s2,pr1,pr2)
    return fn




def abbrv_func(s1,s2,pr1,pr2,n):
    """
    If I abbreviate Cafe Coffee Day to CCD, they both are still the same, this creates abbreviations and checks if same.

    If they are the same words then each of their probabilities are multiplied (of being core or background)

    Args:
        s1  (list): tokenized sentence from first table
        s2  (list): tokenized sentence from second table
        pr1 (dict): core probability distribution of the words of first table
        pr2 (dict): core probability distribution of the words of second table

    Returns:
        float: probability of match
    """

    #n corresponds to number of words abbreviations
    ret = 0.0
    for i in range(len(s1)-n+1):
        w1=s1[i:i+n]
        abbs = ''.join([w[:1] for w in w1])
        #If abbreviation exists then those are same, and thus we can say that these words and 
        #their abbreviations are all part of the core or all part of background
        if(abbs in s2):
            ret+= reduce( (lambda x,y: x*y),[pr1[a] for a in w1])*pr2[abbs] + reduce((lambda x,y: (1-x)*(1-y)),[pr1[a] for a in w1])*(1-pr2[abbs])

    return ret




def dedup(sent1,sent2,pofcore1,pofcore2):
    """
    The core deduplication algorithm. It takes in two sentences (sent1 and sent2) and the probability distribution of core words.

    The algorithm starts by separating words that are in common. Those words in common should be core words in both or background words in both.
    Those not common should be background words in their respective tables.

    We incorporate edit functions (abbreviation and distance metric) and generate probabilities from them.

    Args:
        sent1  (list): tokenized sentence from first table
        sent2  (list): tokenized sentence from second table
        pofcore1 (dict): core probability distribution of the table where sent1 came from
        pofcore2 (dict): core probability distribution of the table where sent2 came from.

    Returns:
        float: Probability of sent1 being a match with sent2
    """
    #To be a same match, all core words should be the same and all other words should be background
    fprob = 0.0
    if isinstance(sent1,float) or isinstance(sent2,float):
        return 0.0
        
    if len(sent1)==0 or len(sent2)==0:
        return 0.0

    common_w = set(sent1).intersection(sent2)
    s1_uncommon = set(sent1) - set(sent2)
    s2_uncommon = set(sent2) - set(sent1)
    commonprob = 1.0
    #If a word is common, it should be part of the core in both sentences or part of background in both sentences
    for w_ in common_w:
        commonprob= commonprob* ( assrtcore(w_,pofcore1)*assrtcore(w_,pofcore2) 
                                    + assrtbg(w_,pofcore1)*assrtbg(w_,pofcore2))
    
    #print "Commons: ", common_w, commonprob
    #If a word is in sentence 1 but not in sentence 2, then it should be a background word
    prob1 = 1.0
    for w1_ in s1_uncommon:
        prob1*=assrtbg(w1_,pofcore1)
    
    #print "uncommon1: ", s1_uncommon, prob1
    
    #If a word is in sentence 2 but not in sentence 1, then it should be a background
    prob2 = 1.0
    for w2_ in s2_uncommon:
        prob2*=assrtbg(w2_,pofcore2)
    
    #print "uncommon2: ", s2_uncommon, prob2
    
    fprob = fprob + prob1*prob2*commonprob
    #print "fprob", fprob
    
    #Add probabilities gotten from edit functions
    fprob= fprob+edit_func(sent1,sent2,pofcore1,pofcore2)
    
    return fprob





def test_sample(alpha,pC,pB):
    """
    A test function to test my dedup algorithm. Not important. Ignore
    Args:
        alpha (float): The weightage of word being core
        pC (dict): The probability distribution of Core words 
        pB (dict): The probability distribution of background words

    Returns:
        None 
    """
    #sample_set = pd.Series([["best", "western", "lamplighter", "inn"],["best","western"],["Hotel","Amsterdam"]])
    #pC,pB = core_algo1(sample_set)
    
    pofcore = dict()
    for k,v in pC.items():
        print k,v
        pofcore[k] = alpha*pC[k]/(alpha*pC[k] + (1-alpha)*pB[k])
    p1 =["best", "western","hotel"]
    p2= ["best","western"]
    #print pofcore
    print dedup(p1,p2,pofcore,pofcore)
    




def coreprob(pC,pB, alpha):
    """
    This function takes the probability distributions generated from the EM algorithm and runs the formula:
        
        .. math:: 
            P(w) = \\frac{\\alpha\\cdot C(w)}{\\alpha\\cdot C(w) + (1-\\alpha )\\cdot B(w)}

    where w is the word.

    Alpha indicates the weight. If alpha is 0 all words are background, if alpha is 1 all words are core.
    
    Args:
        pC (dict): The probability distribution of Core words 
        pB (dict): The probability distribution of background words
    
    Returns:
        dict: the final probability a word is a core or not.
    """
    pofcore = dict()
    for k,v in pC.items():
        pofcore[k] = alpha*pC[k]/(alpha*pC[k] + (1-alpha)*pB[k])
    return pofcore




#
def loaddistributions(filename,cityname):
    """
    Load the saved tokenized features from prepare_4_EM file, run EM algorithm and return the probabilities of address and names.

    Args:
        filename (str): Name of file where the tokenized features are stored
        cityname (str): Name of the city subset where deduplication is to occur 

    Returns:
        pair: (pcore_name,pcore_addr) i.e the core probability distribution of the hotel names and hotel addresses for all the properties in that record 
    """
    emfeatures = pk.load(open(filename+".pk","rb"))
    alpha = 0.8
    #Working with only london properties as of now
    lon_names = emfeatures[emfeatures.city == cityname].name
    lon_addr = emfeatures[emfeatures.city == cityname].address

    #Bring out the Core and Background distributions from the EM algorithm of name
    pC_distr_name,pB_distr_name = core_algo1(lon_names)
    pC_distr_addr, pB_distr_addr = core_algo1(lon_addr)
    #print "rd prob P(C), P(B) ", pC_distr_addr["rd"], pB_distr_addr["rd"]
    #Generate probability of being a core
    pcore_name = coreprob(pC_distr_name,pB_distr_name,alpha)
    pcore_addr = coreprob(pC_distr_addr,pB_distr_addr,alpha)
    
    #pk.dump((pcore_name,pcore_addr),open(filename+"distributions.pk","wb"))
    #print "File: ", filename+"distributions.pk created"
    return (pcore_name,pcore_addr)







#EXTRA:

# def preclean(dat):
#     cln_d = set(flatten2one(dat))
    
#     def snip(x,cln):
#         if len(x)<1:
#             return
#         for idx,w in enumerate(x):
#             if len(w)<=2 or w.isdigit():
#                 continue
#             if w[:-1] in cln:
#                 x[idx]=w[:-1]

#     dat.apply(snip,args=(cln_d,))
#     return dat

#Extra function not used, part of old experiments
# def corecmp(w1,w2, pofcore):
#     if(len(w1)==0 or len(w2)==0):
#         return 0
#     if w1[0]==w2[0]:
#         return assrtcore(w1[0], pofcore)*assrtcore(w2[0],pofcore)
#     else:
#         return 0

# def abbrv(w1,w2, pofcore1,pofcore2):
#     if(len(w1)==0 or len(w2)==0):
#         return 0
#     abbs = ''.join([w[:1] for w in w1])
#     print "inside abbrv: ", abbs
#     if abbs==w2[0]:
#         return assrtcore(w1,pofcore1)*assrtcore(w2,pofcore2) + assrtbg(w1,pofcore1)*assrtbg(w2,pofcore2)
#     else:
#         return 0

# def dedup(p1,p2,i,j,pofcore):
#     if(i==len(p1) and j==len(p2)):
#         return 1
#     ret = 0

#     #PI = {corecmp: assrtcore, abbrv: assrtcore, bginsert: assrtbg, bgdelete: assrtbg}
#     E = {corecmp: (1,1), abbrv: (2,1), bginsert: (1,1), bgdelete: (1,2)}
#     for e in E.keys():
#         l,r = E[e] 
#         W1 = p1[i:i+l]
#         W2 = p2[j:j+r]
#         pi = e(W1,W2,pofcore)
#         if pi==0:
#             continue
#         ret+=pi*dedup(p1,p2,i+l,j+r)
    
#     return ret



#lon_names = preclean(lon_names)
#lon_addr = preclean(lon_addr)
#set(flatten2one(lon_names))

# #s1 = ["cafe","coffee","day"]
# #s2= ["cafe","coffee","day"]
# #s2 = ["cafe","coffee","day"]
# s1=["best","western","hotel"]
# s2=["bw","hotel"]
# tempp = {"cafe": 0.5,"coffee": 0.45,"day": 1.0, "ccd": 1.0, "best":0.3,"western": 0.5,"bw":0.70,"hotel":0.24,"coffefe":0.45,"gay":0.8}
# print dedup(s1,s2,tempp,tempp)

#exmpl = pd.Series([["starbucks","coffee"],["kalmane","coffee"],["starbucks"]])


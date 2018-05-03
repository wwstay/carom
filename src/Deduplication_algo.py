"""
.. module:: Deduplication_algo 
   :synopsis: Core match making algorithm

.. moduleauthor:: Shubhang Goswami <shubhang@wwstay.com>


"""
import CoreBGgeneration
from pymongo import MongoClient
import prepare_4_EM
import pandas as pd
import numpy as np
import cPickle as pk
from scipy import sparse, spatial
from collections import OrderedDict
import math
from monary import Monary
from bson.code import Code
import multiprocessing as mp
import bson
import time


def gen_probvec(name,pCdict,i):
    """
    Generates a vector of length number of words (each words corresponds to an index). For each hotel name/address the vector this function generates is:
    The core probability of word in each idx corresponding to words of the hotel name/address.
    (So if our total table was- "Hotel A" and "B" the vector would be of dim 3 and for "Hotel A" the vector would be [p(hotel),p(A),0] )
    
    Args:
        name (list): A list of tokenized words
        pCdict (dict): A dictionary containing the probability of word being core as a list for both tables
        i (int): index number indicating whether it comes from table 1 or 2

    Return:
        list: A sparse vector of num_words dimension with probabilities. 
    """
    vec = np.zeros(len(pCdict))
    for word in name:
        idx = pCdict.keys().index(word)
        if len(pCdict[word])==1:
            vec[idx] = pCdict[word][0]
        else:
            vec[idx] = pCdict[word][i]
    
    return vec
    
def genreg_vec(name,pCdict):
    """
    Generates a vector of length number of words (each words corresponds to an index). For each hotel name/address the vector this function generates is:
    For each word in a particular name/address, the vector has a 1 and for the rest 0.
    So if our total table was - "hotel A","b" then the vector would be 3 dimensional and for "hotel A" this function would spit out would be [1,1,0]

    Args:
        name (list): A list of tokenized words
        pCdict (dict): A dictionary containing the probability of a word being core (key=word, val=probability)
    Returns:
        list: A one hot vector of num_of_words dimension 
    """
    vec=np.zeros(len(pCdict))
    for word in name:
        idx = pCdict.keys().index(word)
        vec[idx]=1.0
        
    return vec

def add_vals_to_vectdic(odict,addict):
    """
    This function creates a dictionary that contains the probabilistic values of words of both the tables.
    So a key would be the word and the value would be a list (of max 2 elements when comparing 2 and if one word doesn't exist in both tables then only 1)

    Args:
        odict (dict): An ordered dictionary passed in with vals as list and keys as words
        addict (dict): The probabilities of each words you want to append to the ordered dictionary

    Returns:
        dict: A ordered dictionary where the keys are words and values are lists of probabilities from different tables

    """
    for key,val in addict.iteritems():
        if key in odict:
            odict[key].append(val)
        else:
            odict[key] = []
            odict[key].append(val)
            
    return odict

def makecumulativedict(p_corename1,p_corename2):
    """
    This function appends different word:prob dictionary for each table to a single dict with all words.
    (so the new dict has word:[prob1,prob2] for words common in both tables). This function is useful to produce vectors of probabilities (running gen_probvec) 
    
    Args:
        p_corename1 (dict): Dictionary of word:probability from table 1
        p_corename2 (dict): Dictionary of word:probability from table 2

    Returns:
        OrderedDict: A combined dictionary of word:list(probs) of both tables
    """
    vector_dict = OrderedDict([(k,[]) for k in sorted(p_corename1)])
    vector_dict = add_vals_to_vectdic(vector_dict,p_corename1)
    vector_dict = add_vals_to_vectdic(vector_dict,p_corename2)
    return vector_dict



def check_result(inp1,inp2,idcolumn):
    """
    A function used to check accuracy, and extract values that were predicted wrong.

    Args:
        inp1 (DataFrame): Matched dataframe according to model
        inp2 (DataFrame): The original dataframe to which matching occured with human matched values
        idcolumn (str): The column name where the algorithm outputted the matches
    Returns:
        triplet: Three DataFrames that include the merged dataframe after prediction, the ones that were wrong and the third dataframe with the wrong outputs merged with table 2.

    """
    result = pd.merge(inp1,inp2,left_on=[idcolumn],right_on=["mapped_to_ext_id"])
    falses = result[result[idcolumn]!=result["mapped_to_ext_id_x"]]
    print "Acc: ", 1.0*len(result[result[idcolumn]==result["mapped_to_ext_id_x"]])/len(result)
    print "num falses: ", len(falses)
    print "num total: ", len(result)
    wrongones = pd.merge(falses[["name_x","address_x","mapped_to_ext_id_x",idcolumn]],inp2[["name","address","mapped_to_ext_id"]],left_on=["mapped_to_ext_id_x"],right_on=["mapped_to_ext_id"])
    return result,falses,wrongones



def getColsfromdb(supplier_name):
    """
    Monary requires column names and their respective types as input for its query. Thif function obtains the column names and a list of their respective types are generated.

    .. note:: There is a better way to get types along with names by mongo programming. If interested contact me, and I will send you an email from the guy who wrote monary.

    Args:
        supplier_name (str): Name of the collection in supplier_static_database
    Returns:
        pair: A pair of lists. (the column names,their respective types) 
    """
    client = MongoClient()
    db = client['supplier_static_database']
    map = Code("function(){   for (var key in this) {emit(key,null);} }") 
    reduce = Code("function(key, stuff) {return null;}")
    pT = getattr(db, supplier_name)

    mR = pT.map_reduce(map,reduce,supplier_name + "_keys")
    types_ = [type(v).__name__ for k,v in sorted(pT.find().limit(-1).skip(100).next().items())]
    cols4db = []
    for doc in mR.find():
        cols4db.append(doc["_id"])
    cols4db = sorted(cols4db)
    
    for i,t in enumerate(types_):
        if "unicode" in t:
            types_[i] = "string:50"
        if "ObjectId" in t:
            types_[i] = "id"
        if "list" in t or "NoneType" in t:
            types_[i] = "string:50"
        if "int" in t:
            types_[i]="int64"
        if "float" in t:
            types_[i]="float64"
            
    
    try:
        assert len(types_)==len(cols4db)
    except:
        tmpdiff = len(types_)-len(cols4db)
        if tmpdiff<0:
            for i in range(abs(tmpdiff)):
                types_.append("string:10")
        else:
            del cols4db[-tmpdiff:]
            
    return cols4db,types_

    

def getMonaryDF(supplier_name,colnames,types):
    """
    This function generates the dataframe extracted from mongodb

    Args:
        supplier_name (str): The name of the collection in supplier_static_database
        colnames (list): List of column names of the collection 
        types (list): List of column types of the collection 
    Returns:
        DataFrame: The extracted dataframe from mongodb
    """
    client = Monary()
    data = client.query("supplier_static_database",supplier_name,{},colnames,types)
    dat = pd.DataFrame(np.matrix(data).transpose(), columns=colnames)
    dat.replace("",np.nan,inplace=True)
    dat[["Latitude","Longitude"]] = dat[["Latitude","Longitude"]].replace(0.0,np.nan)
    dat = dat[pd.notnull(dat['mapped_to_ext_id'])]
    #dat.columns = map(unicode,dat.columns)
    return dat

def create_distribution(prop1,prop2,supplier_name1,supplier_name2):
    """
    This function is the heart. Running this runs your whold deduplication between supplier_name1 and supplier_name2.

    .. note:: This function contains parallel programming. The problem of PP hasn't been fixed and I urge you to take a look and solve that.

    Args:
        prop1 (DataFrame): DataFrame object for property 1 we are using to compare
        prop2 (DataFrame): DataFrame object for property 2 we are using to compare
        supplier_name1 (str): Name of table where we got prop1 properties from 
        supplier_name2 (str): Name of table where we got prop2 properties from

    Returns:
        DataFrame: The overall matched resolved dataframe.

    """
    ccolsaddr1 = prepare_4_EM.get_relevcols(prop1,"address")
    ccolsname1 = prepare_4_EM.get_relevcols(prop1,"name")
    
    ccolsaddr2 = prepare_4_EM.get_relevcols(prop2,"address")
    ccolsname2 = prepare_4_EM.get_relevcols(prop2,"name")
    
    output = mp.Queue()
    
    def appendercols(ccols):
        val = dict()
        for c in ccols:
            val[c] = ""
        prop.fillna(value=val)
        prop["address"] = ""
        for c in ccols:
            prop["address"]+=prop[c]
            del prop[c]
            
    if len(ccolsaddr1)>1:
        appendercols(ccolsaddr1)
    if len(ccolsname1)>1:
        appendercols(ccolsname1)
        
    if len(ccolsaddr2)>1:
        appendercols(ccolsaddr2)
    if len(ccolsname2)>1:
        appendercols(ccolsname2)
    
    try:
        params1 = pk.load(open("emfeatures_"+supplier_name1+".pk","rb"))
    except:
        params1 = prepare_4_EM.prepareem(prop1,supplier_name1)
        
    try:
        params2 = pk.load(open("emfeatures_"+supplier_name2+".pk","rb"))
    except:
        params2 = prepare_4_EM.prepareem(prop2,supplier_name2)
    
    citynames1 = params1.city.unique().tolist()
    
    
    citynames2 = params2.city.unique().tolist()
    cities = set(citynames1).intersection(set(citynames2))
    
    alpha = 0.8
    
    processes = [0]*len(cities)
    for i,c in enumerate(cities):
        p1 = params1[params1.city==c]
        p2 = params2[params2.city==c]
        processes[i] = mp.Process(target=city_wide_dedup,args=(p1,p2,c))
        try:
            processes[i].start()
        except Exception, e:
            print e
            print c
            processes[i].terminate()
            time.sleep(0.1)
            print "Slacking worker terminated"
    

    for p in processes:
        p.join()
        
    result =[]
    for p in processes:
        result.append(output.get())
    
    return result


def city_wide_dedup(cityparam1,cityparam2,city_name):
    """
    The algorithm that is being run in parallel and matching properties city by city

    Args:
        cityparam1 (DataFrame): The subset of properties from table 1 of a particular city
        cityparam2 (DataFrame): The subset of properties from table 2 of a particular city
        city_name (str): Name of the city we are subsetting
    Returns:
        DataFrame: A dataframe same as cityparam1 with predicted matches appended in a new column called "final_mapped"

    """
    pname1,paddr1 = city_wide_distr(cityparam1,0.8,city_name)
    pname2,paddr2 = city_wide_distr(cityparam2,0.8,city_name)
    
    vector_dict_Pcorename = makecumulativedict(pname1,pname2)
    vector_dict_Pcoreaddr = makecumulativedict(paddr1,paddr2)
    
    cityparam1["final_mapped"]= cityparam1.apply(matchrecord,args=(pname1,paddr1,
                                              cityparam2,
                                              pname2,paddr2,vector_dict_Pcorename,
                                              vector_dict_Pcoreaddr),
                                              axis=1)
    return cityparam1



def DB_dedup(supplier_name1,supplier_name2):
    """
    The wrapper that just takes two supplier names and calls other functions to complete the match making

    Args:
        supplier_name1 (str): The name of the collection 1 in supplier_static_database
        supplier_name2 (str): The name of the collection 2 in supplier_static_database
    Returns:
        DataFrame: The final dataframe with mapped properties 
    """
    colnames1,types1 = getColsfromdb(supplier_name1)
    colnames2,types2 = getColsfromdb(supplier_name2)
    
    prop1 = getMonaryDF(supplier_name1,colnames1,types1)
    prop2 = getMonaryDF(supplier_name2,colnames2,types2)
    
    netres = create_distribution(prop1,prop2,supplier_name1,supplier_name2)
    
    return pd.concat(netres,ignore_index=True)
        




def city_wide_distr(params,alpha,filename,save=False):    
    """
    Creates required probability distribution of names and addresses

    Args:
        params (DataFrame): The dataframe of cleaned and tokenized properties with names and addresses
        alpha (float): The weight of core words. If alpha is 1, all words are regarded as core words and if alpha is 0 all words are regarded as background
        filename (str): Name to save distributions to pickle file. Would work if save is True.
        save (bool): If save is True, it will save the distribution to pickle file.

    Returns:
        pair: (core probability of words in hotel names, core probability of words in hotel addresses) 
    """
    #Working with only london properties as of now
    names = params.name
    addr = params.address
   
    #Bring out the Core and Background distributions from the EM algorithm of name
    pC_distr_name,pB_distr_name = CoreBGgeneration.core_algo1(names)
    pC_distr_addr, pB_distr_addr = CoreBGgeneration.core_algo1(addr)

    
    #Generate probability of being a core
    pcore_name = CoreBGgeneration.coreprob(pC_distr_name,pB_distr_name,alpha)
    pcore_addr = CoreBGgeneration.coreprob(pC_distr_addr,pB_distr_addr,alpha)
    
    if save:
        pk.dump((pcore_name,pcore_addr),open(filename+"distributions.pk","wb"))
        print "File: ", filename+"distributions.pk created"
    return pcore_name,pcore_addr


def get_cossim(row2,row1,vect_name,vect_addr):
    """
    Gets the cossine similarity between two vectors. It concatenates vectors of name and address into a larger dimensional vector and computes the cossine similarity
    This function is used as an apply function.

    Args:
        row2 (Series): A row of a dataframe2
        row1 (Series): A row of a dataframe1
        vect_name (dict): The cumulative vector dictionary made from makecumulativedict for names
        vect_addr (dict): The cumulative vector dictionary made from makecumulativedict for addresses
    
    Returns:
        float: The similarity measure

    """
    genvec2 = np.concatenate([genreg_vec(row2.loc["name"],vect_name), genreg_vec(row2.loc["address"],vect_addr)])
    genvec1 = np.concatenate([genreg_vec(row1.loc["name"],vect_name), genreg_vec(row1.loc["address"],vect_addr)])
    sim_measure_gen = 1-spatial.distance.cosine(genvec1,genvec2)
    if math.isnan(sim_measure_gen):
                sim_measure_gen = 0.0
            
    return sim_measure_gen
    
    



def vect_matchmaker(row, vect_name,vect_addr, data2):
    """
    A wrapper to apply cossine similarity measure for each row in dataframe2. This function is used as an apply function.

    Args:
        row (Series): A row of a dataframe1
        vect_name (dict): An ordered dictionary of words of names from both DataFrames
        vect_addr (dict):  An ordered dictionary of words of addresses from both DataFrames
        data2 (DataFrame): The whole second dataframe to match with
    Returns:
        Series: A series of cossine similarities of row with every single entry in data2 

    """
    measure = data2.apply(get_cossim,args=(row,vect_name,vect_addr),axis=1)
    return measure
    


def haverdist(row2,row1):
    """
    A function that computes haversine distance between two pairs of latitudes and longitudes
    Used as an apply function
    
    Args:
        row2 (Series): A row of a dataframe2
        row1 (Series): A row of a dataframe1

    Returns:
        float: 1-haversine_distance between entry in row1 and entry in row2. So 1 indicates on top of each other, while 0 indicates very far.
    
    """
    if pd.isnull(row1.latitude) | pd.isnull(row2.latitude) | pd.isnull(row1.longitude) | pd.isnull(row2.longitude):
        return 0.0
        
    lat1 = math.radians(row1.latitude)
    lon1 = math.radians(row1.longitude)
    lat2 = math.radians(row2.latitude)
    lon2 = math.radians(row2.longitude)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return (1-c)




def distprobs(row,data2):
    """
    A wrapper to apply haversine distance metric to entries between a row in dataframe1 to every entry in dataframe2

    Args:
        row (Series): A row of a dataframe1
        data2 (DataFrame): The dataframe to match dataframe1 with.

    Returns:
        Series: A series of haveresine closeness for entry in row 1 with every entry in data2
    """
    dists = data2.apply(haverdist,args=(row,),axis=1)
    return dists



def matching(row2,row1,pCname_self,pCaddr_self, pCsent_name, pCsent_addr):
    """
    This function runs the dedup algorithm from CoreBGgeneration for hotel names and addresses.

    Args:
        row2 (Series): A row of a dataframe2
        row1 (Series): A row of a dataframe1
        pCname_self: The word core probabilities for hotel names in dataframe2
        pCaddr_self: The word core probabilities for hotel addresses in dataframe2
        pCsent_name: The word core probabilities for hotel names in dataframe1
        pCsent_addr: The word core probabilities for hotel addresses in dataframe1

    Returns:
        float: The probability of being a match.
    """
    name_prob = CoreBGgeneration.dedup(row1.loc["name"],row2.loc["name"],pCsent_name,pCname_self)
    addr_prob = CoreBGgeneration.dedup(row1.loc["address"],row2.loc["address"],pCsent_addr,pCaddr_self)
    #print "name:address probability, ", name_prob, address_prob
    return name_prob*(addr_prob+0.2*name_prob)
    

def matchrecord(row,pCname_self,pCaddr_self,data2,pCname2,pCaddr2,vect_name,vect_addr):
    """
    This function combines all different models and methods into one and provides the best match as result.
    .. note:: This needs to improve in terms of how much weight/how much should we listen to which model. This is a common ensemble method problem, and should be solved with the training set provided.
    
    Args:
        row (Series): A row of a dataframe1
        pCname_self (dict): The word core probabilities for hotel names in dataframe1
        pCaddr_self (dict): The word core probabilities for hotel addresses in dataframe1
        data2 (DataFrame): The dataframe to match with
        pCname2 (dict): The word core probabilities for hotel names in dataframe2
        pCaddr2 (dict): The word core probabilities for hotel addresses in dataframe2
        vect_name (dict): An ordered dictionary of words of names from both DataFrames
        vect_addr (dict): An ordered dictionary of words of addresses from both DataFrames

    Returns:
        str: The index with the maximum probability of being a match, or "none" if none found

    """

    probs=data2.apply(matching,args=(row,pCname2,pCaddr2,pCname_self,pCaddr_self),axis=1)
    
    
    vect_prob = vect_matchmaker(row,vect_name,vect_addr,data2)

    latlongprob = distprobs(row,data2)
    
    fnalprobs = 0.14*probs + 0.86*vect_prob + latlongprob
    
    if (probs + vect_prob).max()<=1e-06:
        return "none"
    
    return data2.loc[probs.idxmax(),"mapped_to_ext_id"]


def main():
    """
    main function to start with. This will make a comparison between two properties belonging to mongo. Right now it is tuned to hotelbeds_properties and taap_properties. Update those for different suppliers.
    Also calculates the time taken for the whole process to run from start to finish.
    Args:
        None
    Returns:
        None
    """

    pd.options.mode.chained_assignment = None
    start_time = time.time()
    print "Starting at: ", start_time
    try:
        fnal_df = DB_dedup("hotelbeds_properties","taap_properties")
    except:
        print("--- %s seconds ---" % (time.time() - start_time))

    print "End of process"
    pk.dump(fnal_df,open("hotelbeds_taap_dedup_results.pk","wb"))
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()

#get_ipython().run_cell_magic(u'time', u'', u'%%capture capt\nreload(CoreBGgeneration)\nreload(prepare_4_EM)\nfnal_df = DB_dedup("hotelbeds_properties","taap_properties")')

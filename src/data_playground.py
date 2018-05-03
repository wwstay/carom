import pandas as pd 
import numpy as np 
from pymongo import MongoClient
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import os
import cPickle as pk
import pdb
import math

def get_df(df):
	#Extract column names
	sve_col = [col for col in df.columns.values if 'name' in col.lower() or 'address' in col.lower() or 'latitude' in col.lower() or 'longitude' in col.lower() or "city" in col.lower()]
	sve_col.append('location')
	#Extract columns
	ndf = pd.DataFrame([df[ccol].str.lower() if 'name' in ccol.lower() or 'address' in ccol.lower() or 'city' in ccol.lower() else df[ccol] for ccol in sve_col]).T
	ndf.columns = map(unicode.lower, ndf.columns)

	return ndf		


def get_relevcols(df, colname):
	return [col for col in df.columns.values if colname in col.lower()]

def dump4bigram(*args):
	for idx,subdf in enumerate(args):
		subdf[get_relevcols(subdf, 'name')].to_csv(repr(idx)+"_names.txt",sep='\n',encoding='utf-8',index=False,header=False)
		subdf[get_relevcols(subdf, 'address')].to_csv(repr(idx)+"_address.txt",sep='\n',encoding='utf-8',index=False,header=False)

def tokener(textS):
	#Remove punctuation
	def splitter(txt):
		tx1 = txt.str.replace('[^\w\s]','')
		return tx1.str.split()
	
	tdat = textS.apply(splitter,axis=1)
	tdatS = pd.Series(tdat.squeeze(),index=tdat.index)
	#Flatten Series
	flattdat = [item for sublist in tdatS.tolist() for item in sublist]
	#Get english+french+german stopwords (such as and, or, is, the, a etc.)
	eng_stop_words = set(stopwords.words("english")+['test','1234','various','addresses'])
	fr_stop_words = set(stopwords.words("french"))
	ger_stop_words = set(stopwords.words("german"))
	stop_wrds = eng_stop_words.union(fr_stop_words).union(ger_stop_words)
	
	#Remove all such stopwords
	flattdat = [w for w in flattdat if w.lower() not in stop_wrds]

	return flattdat

def exploretext(textS,nme):
	#Tokenize data
	flattdat = tokener(textS)
	nltext = nltk.Text(flattdat)
	
	#Print most common bigrams
	print nltext.collocations()
	
	#Plot cumulative frequency distribution of words
	fdistr = nltk.FreqDist(nltext)
	fdistr.plot(50,cumulative=True)
	plt.savefig("cuml_freq_indvwords_"+nme+".png")
	plt.close()

	#Plot freq distribution of number of letters in a token present
	lettfdistr = nltk.FreqDist([len(w) for w in nltext])
	lettfdistr.plot(50)
	plt.savefig("letr_freq_"+nme+".png")
	plt.close()

	#Plot bigrams 
	nlbigrams = list(nltk.bigrams(flattdat))
	print nlbigrams[:50]
	fd = nltk.FreqDist(nlbigrams)
	fd.plot(10, cumulative=True)
	plt.savefig("bigram_freq_"+nme+".png")
	plt.close()

#Produce n gram
def ngrammer(inplis, n):
	if len(inplis)<1:
		return np.nan
	return list(zip(*[inplis[i:] for i in range(n)]))

def stopwordgen():
	eng_stop_words = set(stopwords.words("english")+['test','1234','various','addresses'])
	fr_stop_words = set(stopwords.words("french"))
	ger_stop_words = set(stopwords.words("german"))
	return eng_stop_words.union(fr_stop_words).union(ger_stop_words)

def stemmer(x,stemdic):
	tokens = x.str.replace('[^\w\s]','').str.split().tolist()[0]
	
	for word in tokens:
		if word in stemdic:
			word = stemdic[word]


def gramclean(x,stopwords):
	if x.isnull().all():
		return np.nan
	return ' '.join([word for word in x.str.replace('[^\w\s]','').str.split().tolist()[0] if word not in stopwords])

#Clean data frame and generate ngrams
def gramsplitter(df,n):

	stop_wrds = stopwordgen()

	df = df.apply(gramclean,axis=1,args=(stop_wrds,))
	return df.map(lambda x: x if (isinstance(x,float)) else ngrammer(x.split(" "),n))

def analyze(dat):
	#Get name and address data
	namedat = dat[get_relevcols(dat,'name')]
	addr_dat = dat[get_relevcols(dat,'address')]

	#exploretext(namedat,"name")
	
	#exploretext(addr_dat,"addr")
	
	#Create dataframe
	prop_data = pd.DataFrame({"name_bigrams": gramsplitter(namedat,2),
							  "address_trigrams": gramsplitter(addr_dat,3)})
	return prop_data

def createparams(db_props,propname):
	prop_dat= pd.DataFrame(list(db_props.find()))
	#hot_beds = pd.DataFrame(list(db.hotelbeds_properties.find()))
	colrmv = ["verification","verified","mapped_to","mapped_to_ext_id"]
	prop_dat.drop(colrmv,axis=1,inplace=True)

	prop_subdf = get_df(prop_dat)
	print prop_subdf.columns
	
	prop_params = analyze(prop_subdf)
	prop_params['latitude'] = prop_subdf[get_relevcols(prop_subdf,'latitude')]
	prop_params['longitude'] = prop_subdf[get_relevcols(prop_subdf,'longitude')]
	prop_params['city'] = prop_subdf['city']
	print prop_params
	prop_params.to_csv(propname+"_parameters.scsv",encoding='utf-8',index=False,sep=';')
	pk.dump(prop_params,open(propname+"_params.pk","wb"))

def benchmark_approach(db):
	createparams(db.hotelbeds_properties,"hotelbeds")
	createparams(db.taap_properties,"taap")

def emapproach(db):
	prop_dat= pd.DataFrame(list(db.find()))
	prop_subdf = get_df(prop_dat)
	names = prop_subdf[get_relevcols(prop_subdf,'name')]
	addr_dat = prop_subdf[get_relevcols(prop_subdf, 'address')]

	clean_names = names.apply(gramclean, args=(stopwordgen(),), axis=1)
	clean_addr = addr_dat.apply(gramclean,args=(stopwordgen(),),axis=1)


def main():
	client = MongoClient()
	db = client['supplier_static_database']
	emapproach(db.hotelbeds_properties)

	
	

if __name__ == '__main__':
	sns.set_style("darkgrid")
	sns.set_context("notebook")
	sns.set(color_codes=True)
	main()



 ####### EXTRA CODE- EXPERIMENTS #######
	# if not os.path.isfile("taap_subdf.scsv"):
	# 	createcsv(db.taap_properties,"taap_subdf")
	# taap_subdf = pd.read_csv("taap_subdf.scsv",sep=";",encoding='utf-8')
	#hotbed_subdf = pd.read_csv("hotbed_subdf.scsv",sep=";")
	#print taap_subdf


#In main()	#prop_subdf.to_csv(propname+".scsv",encoding='utf-8',index=False,sep=';')

#In analyze(): 	
		#bagof2name = map(tuple,set(map(frozenset,list(nltk.bigrams(tokener(namedat))))))
		#bagof3addr = map(tuple,set(map(frozenset,list(nltk.trigrams(tokener(addr_dat))))))
		#dump4bigram(taap_subdf,hotbed_subdf)

#In analyze(dat):	
	# def splitting(txt,gram=2):
	# 	if(pd.isnull(txt.values)):
	# 		return np.nan
	# 	tx1 = txt.str.replace('[^\w\s]','').str.split().tolist()[0]
	# 	if(len(tx1)==0):
	# 		return np.nan
	# 	txlis = [w for w in tx1 if w.lower() not in stop_wrds]
	# 	print 1, txlis
	# 	print 2, nltk.bigrams(txlis)
	# 	print 3, list(nltk.bigrams(txlis))
	# 	print 4, map(frozenset,list(nltk.bigrams(txlis)))
	# 	print 5, set(map(frozenset,list(nltk.bigrams(txlis))))
	# 	print 6, map(tuple,set(map(frozenset,list(nltk.bigrams(txlis)))))
	# 	print len(map(tuple,set(map(frozenset,list(nltk.bigrams(txlis)))))))
	# 	if gram==2:
	# 		if(len(txlis)<2):
	# 			return tuple(txlis)
	# 		return map(tuple,set(map(frozenset,list(nltk.bigrams(txlis)))))
	# 	else:
	# 		if(len(txlis)<3):
	# 			return tuple(txlis)
	# 		return map(tuple,set(map(frozenset,list(nltk.trigrams(txlis)))))

# 	
	#pdb.set_trace()
	# print splitting(namedat.iloc[1],2)
	# prop_nameser = []
	# prop_addrser = []
	# for idx,row in namedat.iterrows():
	# 	prop_nameser.append(splitting(row))
	# 	prop_addrser.append(splitting(addr_dat.iloc[idx],3))

'''
	prop_data = pd.DataFrame({'name_bigrams': prop_nameser, 'address_trigrams': prop_addrser})
	
	#print namedat.apply(splitting)
	print prop_data

	assert 1==0
	prop_data = pd.DataFrame(namedat.apply(splitting,args=(2,), axis=1))
	print prop_data
	print type(prop_data)

	prop_data['address_trigrams'] = addr_dat.apply(splitting,args=(3,),axis=1)

	#print prop_data

	return prop_data
	'''
#In get_df():
	# for row,idx in ndf.iteritems:
	# 	namecol = [col if 'name' in col for col in ndf.columns]
	# 	for word in row[namecol].split():
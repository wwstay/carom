import pandas as pd 
import numpy as np
import cPickle as pk 
import nltk
from collections import Counter, OrderedDict
from shapely.geometry import MultiPoint
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse, spatial
import math

class OrderedCounter(Counter, OrderedDict):
    pass

def learn_vocab(txS):
	#flat = ['null' if len(sublist)<1 else item for sublist in txS.values.tolist() for item in sublist]
	flat = []
	for sublist in txS.values.tolist():
		if(isinstance(sublist, float)):
			flat.append('null')
			continue
		for item in sublist:
			flat.append(item)

	dat = map(frozenset,flat)
	vocab = OrderedCounter(dat)
	#tuped = [(k,v) for k,v in vocab.items()]
	#return OrderedDict(tuped)
	return vocab
	#print vocab

def gen_vec(gram_txt, vocab):
	vec = np.zeros(len(vocab)+1)
	gram_txt = map(frozenset,gram_txt)
	#print gram_txt
	if(isinstance(gram_txt,float)):
		vec[-1] = 1
		return vec
	for gram in gram_txt:
		# print gram,
		# print vocab.keys().index(gram)
		try:
			vec[vocab.keys().index(gram)]=1
		except:
			continue
	#for k,v in vocab.items():
	return vec

def getcossim(data1, data2):
	result = []
	for idx1,vec1 in enumerate(data1):
		corr={}
		for idx2,vec2 in enumerate(data2):
			sim = 1-spatial.distance.cosine(vec1,vec2)
			if math.isnan(sim):
				corr[idx2] = 0.0
			else:
				corr[idx2] = sim
		most_sim = max(corr, key=corr.get)
		result.append({most_sim: corr[most_sim]})
		print corr

	return result

def main():
	taap = pk.load(open("taap_params.pk","rb"))
	hotbeds = pk.load(open("hotelbeds_params.pk","rb"))
	name_voc = learn_vocab(taap['name_bigrams'])
	address_voc = learn_vocab(taap['address_trigrams'])

	#taap_sub = pd.read_csv("taap_subdf.scsv",sep=';')
	#print hotbeds
	taap_sub = taap[taap['city']=='london']
	hotbeds_sub = hotbeds[hotbeds['city']=='london']

	#print hotbeds_sub['name_bigrams'].iloc[3]
	hotb_namevec = hotbeds_sub['name_bigrams'].apply(gen_vec,args=(name_voc,))
	hotb_addrvec = hotbeds_sub['address_trigrams'].apply(gen_vec,args=(address_voc,))
	print "Hotelbeds Done!!"
	taapb_namevec = taap_sub['name_bigrams'].apply(gen_vec,args=(name_voc,))
	taapb_addrvec = taap_sub['address_trigrams'].apply(gen_vec,args=(address_voc,))
	print "TAAP Done!"
	
	# pk.dump(pd.DataFrame({'name_vec': taapb_namevec,
	# 			  'address_vec': taapb_addrvec}), open("taap_vecs.pk","wb"))
	# pk.dump(pd.DataFrame({'name_vec': hotb_namevec,
	# 			  'address_vec': hotb_addrvec}), open("hotelb_vecs.pk","wb"))
	
	cossim_name = getcossim(taapb_namevec[:50], hotb_namevec)
	print cossim_name

	# most_conn = np.argmax(cossim,axis=1)
	# print most_conn[1],taap_sub['name_bigrams'].iloc[1], hotbeds_sub['name_bigrams'].iloc[most_conn[1]]



	#print namevec
	#print addrvec

if __name__ == '__main__':
	main()
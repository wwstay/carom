
# coding: utf-8

import pandas as pd
import urllib2
from bs4 import BeautifulSoup
from cookielib import CookieJar

#Visit link 1 to see the structure of the table involved. This algo assigns the abbreviations to a dictionary
def Get_abbrvsC1(tabl):
    Abbrv = {}
    state = "0"
    for row in tabl.findAll("tr"):
        cells = row.findAll("td")
        if not (state in Abbrv):
            Abbrv[state] = []
        if len(cells)<2:
            #print "\t", cells[0].find(text=True)
            Abbrv[state].append(cells[0].find(text=True).strip().lower())
        elif len(cells)>3:
            continue
        else:
            state = cells[2].find(text=True).strip().lower()
            if not (state in Abbrv):
                Abbrv[state] = []
            Abbrv[state].append(cells[1].find(text=True).strip().lower())
            #print cells[0].find(text=True),"\t", cells[1].find(text=True),"\t",cells[2].find(text=True)
    del Abbrv[""]
    del Abbrv["0"]
    return pd.Series(Abbrv)



#Link 2 is simpler, column length is same in each column for table. Assigns abbreviations from link 2
def Get_abbrvsC2(tabl):
    Abbrv={}
    for row in tabl.findAll("tr"):
        cells = row.findAll("td")
        if len(cells)==2:
            Abbrv[cells[1].find(text=True).strip().strip("*").lower()]=[cells[0].find(text=True).strip().lower()]
    del Abbrv['']
    del Abbrv['approved abbreviation']
    return pd.Series(Abbrv)       


#Returns the word if not in the dictionary, else assigns the abbreviated/key part of it (so avenue -> ave)
def findidx(StemWords,word):
    A = []
    for key,value in StemWords.items():
        if word in value:
            return key
        else:
            return word

if __name__ == '__main__':
    #These links contain common USPS abbreviations from which I build my stemming dictionary
    link1 = "http://pe.usps.com/text/pub28/28apc_002.htm"
    link2 = "http://pe.usps.com/text/pub28/28apc_003.htm"
    #To prevent an error, we needed to enable cookies and send request
    cj = CookieJar()
    agent_ = 'Mozilla/5.0 (Windows NT 6.1; rv:54.0) Gecko/20100101 Firefox/54.0'

    #Open the links
    opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
    opener.addheaders = [('user-agent',agent_)]
    pageC1response = opener.open(link1)
    pageC2response = opener.open(link2)

    soupC1 = BeautifulSoup(pageC1response)
    soupC2 = BeautifulSoup(pageC2response)

    tablC1 = soupC1.table
    tablC2 = soupC2.table

    abbrvSC2 = Get_abbrvsC2(tablC2)
    abbrvSC1 = Get_abbrvsC1(tablC1)


    for key_ in abbrvSC1.keys().values:
        if key_[-1]=='s':
            if key_[:-1] in abbrvSC1:
                abbrvSC1[key_[:-1]] = abbrvSC1[key_[:-1]] + abbrvSC1[key_]
                del abbrvSC1[key_]


    StemWords = pd.concat([abbrvSC1,abbrvSC2])

    #Added some extra words to the dictionary, such that if it finds any of these words we can represent it as such
    StemWords["frt"] = ["fort","frt","forts","frts","ft"]
    StemWords["n"] = ["north","nrth","n"]
    StemWords["s"] = ["south","s"]
    StemWords["e"] = ["east","e"]
    StemWords["w"] = ["west","w"]
    StemWords["hwy"] = ["highway","hgwy","hway","hw","hwy","highways","hgwys","hways","hws","hwys"]
    StemWords["hotel"] = ["hotel","hotels","hotl"]
    StemWords["hostel"] = ["hostel","hostels","hostls","hostl"]

    #Store the dictionary to be used later
    StemWords = StemWords.sort_index()
    StemWords.to_csv("stemming_dict.csv")




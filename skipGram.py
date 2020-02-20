from __future__ import division
import argparse
import pandas as pd



# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize

import spacy
from sklearn.feature_extraction.text import CountVectorizer


__authors__ = ['Raphael Attali','Ariel Modai','Niels Nicolas','Michael Allouche']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']
  
def text2sentences(path):
	# feel free to make a better tokenization/pre-processing
    #spacy_nlp = spacy.load("en_core_web_sm")
    sentences = []
    with open(path) as f:
        for l in f:
            #string= l.lower().split()
            #spacy_tokens=spacy_nlp(string)
            #string_tokens = [token.orth_ for token in spacy_tokens]  
            #spacy_tokens_join = " ".join(string_tokens)
            #sentences.append(spacy_tokens_join)   
            #vectorizer = CountVectorizer()
            #X_sample = vectorizer.fit_transform(sentences)
            #vectorizer.get_feature_names()
            sentences.append( l.lower().split() )
    return sentences

def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs
    
class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.trainset =set(tuple(row) for row in sentences) # set of sentences
        self.vocab = []   # list of valid words
        self.w2id={}  # word to ID mapping
        self.word_occurences={}
        
        cont=0
        for l_sentence in sentences:
            for sentence in l_sentence:
                for word in sentence.split():
                    cont=cont+1
                    if word not in self.vocab:
                        self.vocab.append(word) 
                        self.word_occurences[word]=1
                    else:
                        self.word_occurences[word]+=1
        for i in range(len(self.vocab)):
            self.w2id[self.vocab[i]]=i
        #sort the dictionariesS
        #self.trainset=sorted(list(self.trainset.keys()))
        #self.w2id=sorted(list(self.w2id.keys())) 
        self.nEmbed=nEmbed

        self.negativeRate=negativeRate
        self.winSize=winSize
        self.minCount=minCount
        self.w=np.random.uniform(-0.8,0.8,(len(self.vocab),self.nEmbed))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.nEmbed, len(self.vocab))) 
        self.alpha = 0.001
        #raise NotImplementedError('implement it!')
        
    def sample(self, omit): 
        probability_occur=self.word_occurences.copy()
        probability_occur.update((x, y**(3/4)) for x, y in probability_occur.items())
        denominator_sample=sum(probability_occur.values())
        probability_occur.update((x, y/denominator_sample) for x, y in probability_occur.items())
        sample_positive=1
        number_positive=0
        negativeIds=[]
        while sample_positive==1:
            sampled_list=np.random.choice(list(probability_occur.keys()), size=self.negativeRate, p=list(probability_occur.values()))
            for negative_word in sampled_list:
                if self.w2id[negative_word] in omit:
                    number_positive+=1
            if number_positive==0:
                sample_positive=0
        for negative_word in sampled_list:      
            negativeIds.append(self.w2id[negative_word])
        return negativeIds
        
        #"""samples negative words, ommitting those in set omit"""
        #raise NotImplementedError('this is easy, might want to do some preprocessing to speed up')
    
    def softmax(self,x):
        return np.exp(x)/sum(np.exp(x))
       
    def forward_propagate(self,x_train): 
        self.hidden_layer = np.dot(self.w.T,x_train).reshape(self.nEmbed,1) 
        self.u = np.dot(self.w2.T,self.hidden_layer) 
        self.y_pred = self.softmax(self.u)   

    def back_propagate(self,y_train,x_train,sample_id):
        error=self.y_pred-y_train
        
        dL_dw2=np.dot(self.hidden_layer,error.T)
        x_reshape= np.array(x_train).reshape(len(self.x_train),1) 
        dL_dw=np.dot(x_train[[i for i in sample_id]],np.dot(self.w2,error).T)
        
        self.w=self.w-self.alpha*dL_dw
        self.w2=self.w2-self.alpha*dL_dw2
        
        acjdi
   
    def train(self):
        for counter, sentence in enumerate(self.trainset):
            sentence = filter(lambda word: word in self.vocab, sentence)

            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                winsize = np.random.randint(self.winSize) + 1
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))
                self.trainWords=0
                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx: continue
                    negativeIds = self.sample({wIdx, ctxtId})
                    sample_id=negativeIds.append(wIdx)
                    self.trainWord(wIdx, ctxtId, sample_id)
                    self.trainWords += 1    #What is the meaning of this variable? I think it is the size of the  context
                    
                    
            if counter % 1000 == 0:
                print (' > training %d of %d' % (counter, len(self.trainset)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0 
                self.accLoss = 0.
    
    
    def trainWord(self, wordId, contextId, sample_id):
        size_input=len(self.vocab)
        x_train = [0 for x in range(size_input)] 
        y_train = [0 for x in range(size_input)] 
        x_train[wordId]=1
        y_train[contextId]=1
        self.forward_propagate(x_train)
        self.back_propagate(y_train,x_train,sample_id)
        #raise NotImplementedError('here is all the fun!')
        
    def save(self,path):
        #print("not implemented")
	    raise NotImplementedError('implement it!')
        
    def similarity(self,word1,word2):
	    #"""
		#computes similiarity between the two words. unknown words are mapped to one common vector
        #:param word1:
        #:param word2:
        #:return: a float \in [0,1] indicating the similarity (the higher the more similar)
        #"""
        raise NotImplementedError('implement it!')
    @staticmethod
    def load(path):
        raise NotImplementedError('implement it!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')
    opts = parser.parse_args()
    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train()
        sg.save(opts.model)
    else:
        pairs = loadPairs(opts.text)
        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
             # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a,b))

##############
#In order to test without command lines
#f=open("train.txt","r")            
#text=f.read()
#sentences = text2sentences(text)

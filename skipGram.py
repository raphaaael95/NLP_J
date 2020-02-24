from __future__ import division
import argparse
import pandas as pd
#import time
import json

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize
import scipy
import logging
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

__authors__ = ['Raphael Attali','Ariel Modai','Niels Nicolas','Michael Allouche']
__emails__  = ['raphael.attali@student-cs.fr','niels.nicolas@student-cs.fr','ariel.modai@student-cs.fr','michael.allouche@student-cs.fr']
  
def text2sentences(path):
	# feel free to make a better tokenization/pre-processing
    spacy_nlp = spacy.load("en_core_web_sm")
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append([token.lemma_.replace(' ','_').lower() for token in spacy_nlp(l) if (len(token.lemma_)>2)
            and (token.lemma_ in spacy_nlp.vocab) and (',' not in token.lemma_)
            and('.' not in token.lemma_) and ('\n' not in token.text)
            and (not token.is_stop) and (not token.is_punct)
            and (not token.is_digit) and (not token.is_space)])
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
        self.w2 = np.random.uniform(-0.8, 0.8, (len(self.vocab),self.nEmbed)) 
        self.alpha = 0.001
        self.trainWords=0
        self.accLoss=0
        self.loss=[]
        
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
            number_positive=0
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
       

           
    def train(self):
        logging.basicConfig(filename='example.log',level=logging.DEBUG)
        for counter, sentence in enumerate(self.trainset):
            sentence = filter(lambda word: word in self.vocab, sentence)

            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                winsize = np.random.randint(self.winSize) + 1
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))
                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx: continue
                    negativeIds = self.sample({wIdx, ctxtId})
                    negativeIds.append(wIdx)
                    self.trainWord(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1    #number of iterations
            print("Niels est un petit juif")
            print(counter)  
            #time.sleep(4)
            if counter % 1000 == 0:
                print (' > training %d of %d' % (counter, len(self.trainset)))
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0 
                self.accLoss = 0.
            logging.debug(self.loss)
            if counter==1003:
                print("counter 1003")
                break
            

            
    
    def trainWord(self, wordId, contextId, sample_id):
        size_input=len(self.vocab)
        x_train = [0 for x in range(size_input)] 
        y_train = [0 for x in range(size_input)] 
        context=[0 for x in range(size_input)]
        context[contextId]=1
        x_train[wordId]=1
        y_train[contextId]=1        
        #convert list in numpy
        x_train=np.asarray(x_train)
        y_train=np.asarray(y_train)
        context=np.asarray(context)
        self.hidden_layer = np.dot(self.w.T,x_train).reshape(self.nEmbed,1)
        #print(self.hidden_layer.shape)
        #self.u = np.dot(self.w2,self.hidden_layer) 
        #self.y_pred = self.softmax(self.u) 
        self.accLoss=-np.log(self.softmax(self.w2[contextId,:].T * self.hidden_layer.T))
        for id_sample in sample_id:
            self.accLoss=self.accLoss-np.log(self.softmax(-self.w2[id_sample,:].T * self.hidden_layer.T))
            
        grad_V_output_pos = (self.softmax(self.w2[contextId,:].T * self.hidden_layer.T) - 1) *self.hidden_layer.T  # h or w
        grad_V_input = (self.softmax(self.w2[contextId,:].T * self.hidden_layer.T) - 1) * self.w2[contextId,:].T
        grad_V_output_neg_list = []
        
        for id_sample in sample_id:
            grad_V_output_neg_list.append(self.softmax(self.w2[id_sample,:].T * self.hidden_layer.T) * self.hidden_layer.T)
            grad_V_input += self.softmax(self.w2[id_sample,:].T * self.hidden_layer.T) * self.w2[id_sample,:].T
        # use SGD to update w, c_pos, and c_neg_1, ... , c_neg_K

        self.w2[contextId,:] = self.w2[contextId,:] - self.alpha * grad_V_output_pos
        self.w[contextId,:] = self.w[contextId,:] - self.alpha * grad_V_input
        
    def save(self,path):
        parameters={
            "input_weights":self.w.tolist(),
            "output_weights":self.w2.tolist(),
            "vocab":self.vocab,
            "w2id":self.w2id
        }
        json.dump(parameters,open(path,'wt'))
        
    def similarity(self,word1,word2):
        self.w=np.asarray(self.w)
        self.w2=np.asarray(self.w2)
        self.w=self.w.reshape((len(self.vocab),self.nEmbed))
        self.w2=self.w2.reshape((len(self.vocab),self.nEmbed))
        #cosine=cosine_similarity(self.w[self.w2id[a],:], self.w[self.w2id[b],:], dense_output=True)
        if a not in self.vocab or b not in self.vocab:
            return 0
        else:
            cosine=scipy.spatial.distance.cosine(self.w[self.w2id[a],:], self.w[self.w2id[b],:])
            logging.debug(cosine)
            return cosine
    #print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%')
        #"""
		#computes similiarity between the two words. unknown words are mapped to one common vector
        #:param word1:
        #:param word2:
        #:return: a float \in [0,1] indicating the similarity (the higher the more similar)
        #"""
        #raise NotImplementedError('implement it!')
    @staticmethod
    def load(path):
        with open(path) as json_file:
            data_j = json.load(json_file)
            sg=SkipGram(" ")
            sg.w=data_j["input_weights"]
            sg.w2=data_j["output_weights"]
            sg.vocab=data_j["vocab"]
            sg.w2id=data_j["w2id"]
            return sg
        #raise NotImplementedError('implement it!')

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
            
            #print(a,b)
            print(sg.similarity(a,b))

##############



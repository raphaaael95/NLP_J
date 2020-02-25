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
from collections import OrderedDict

__authors__ = ['Raphael Attali','Ariel Modai','Niels Nicolas','Michael Allouche']
__emails__  = ['raphael.attali@student-cs.fr','niels.nicolas@student-cs.fr','ariel.modai@student-cs.fr','michael.allouche@student-cs.fr']
  
def text2sentences(path):
	# feel free to make a better tokenization/pre-processing
    spacy_nlp = spacy.load("en_core_web_sm")
    sentences = []
    with open(path) as f:
        for l in f:
            string=l.lower()
            #string=string.rstrip("\n")
            #string=string.rstrip(" ")
            #string=string.rstrip("-")
            #string=string.rstrip(".")
            #string=''.join([i for i in string if not i.isdigit()])
            spacy_tokens=spacy_nlp(string)
            spacy_tokens=[i for i in spacy_tokens if len(i)>2]
            lemmas = [token.lemma_ for token in spacy_tokens]
            stopwords = spacy.lang.en.stop_words.STOP_WORDS
            a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() or lemma == '-PRON-']
            a_lemmas = [lemma for lemma in a_lemmas if lemma!='-PRON-']
            a_lemmas = [lemma for lemma in a_lemmas if lemma not in stopwords]
            #a_lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in stopwords]
            #string_tokens = [token.orth_ for token in lemmas if not token.is_punct if not token.is_stop]
            #sentences.append([token.lemma_.replace(" ","_").lower() for token in spacy_nlp(l) if (len(token.lemma_)>2)
            #    and (token.lemma_ in spacy_nlp.vocab) and ("," not in token.lemma_)
            #    and('.' not in token.lemma_) and ("\n" not in token.text)
            #    and (not token.is_stop) and (not token.is_punct)
            #    and (not token.is_digit) and (not token.is_space)])
            if(len(a_lemmas)>=1):
                sentences.append(a_lemmas)
    return sentences

def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    #text2sentences(data)
    pairs = zip(data['word1'],data['word2'],data['similarity'])     
    return pairs
    
class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.trainset =set(tuple(row) for row in sentences) # set of sentences
        self.trainset=sorted(self.trainset)
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
        self.alpha = 0.05
        self.trainWords=0
        self.accLoss=0
        self.loss=[]
        self.w2id=OrderedDict(sorted(self.w2id.items(),key=lambda t:t[0]))
        #print(self.w2id)
        
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
            sampled_list=np.random.choice(list(probability_occur.keys()), size=self.negativeRate-1, p=list(probability_occur.values()))
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
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
       

           
    def train(self):
        for i in range(10):             # number of epochs
            for counter, sentence in enumerate(self.trainset):
                sentence = filter(lambda word: word in self.vocab, sentence)
                sentence = list(sentence)
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
                #if counter==5:
                #    print("counter 1003")
                #    break
            

            
    
    def trainWord(self, wordId, contextId, sample_id):
        size_input=len(self.vocab)
        x_train = [0 for x in range(size_input)] 
        context=[0 for x in range(size_input)]
        x_train[wordId]=1
        context[contextId]=1


        #convert list in numpy
        x_train=np.asarray(x_train)
        context=np.asarray(context)
        self.hidden_layer=self.w[wordId,:]
        #self.pred=self.sigmoid(np.dot(self.hidden_layer,self.w2.T))
        self.pred_pos=self.sigmoid(np.dot(self.w2[contextId,:].T , self.hidden_layer))
        self.accLoss=self.accLoss-np.log(self.pred_pos)
        self.pred_neg_l=[]
        for id_sample in sample_id:
            self.pred_neg=np.dot(self.w2[id_sample,:].T , self.hidden_layer)
            self.accLoss=self.accLoss-np.log(self.sigmoid(-self.pred_neg))
            self.pred_neg_l.append(self.pred_neg)

                
        grad_V_output_pos = np.dot(self.pred_pos - 1, self.hidden_layer)  # h or w
        grad_V_input = np.dot(self.pred_pos - 1, self.w2[contextId,:])
        grad_V_output_neg_list = []
        
        
        for neg_prod in self.pred_neg_l:
            grad_V_output_neg_list.append(np.dot(self.sigmoid(neg_prod) , self.hidden_layer))
            grad_V_input += np.dot(self.sigmoid(neg_prod) , self.w2[id_sample,:])
        # use SGD to update w, c_pos, and c_neg_1, ... , c_neg_K

        self.w2[contextId,:] = self.w2[contextId,:] - self.alpha * grad_V_output_pos
        self.w[wordId,:] = self.w[wordId,:] - self.alpha * grad_V_input
        cont=0
        for grad_V_output_neg in grad_V_output_neg_list:
            self.w2[sample_id[cont],:] = self.w2[sample_id[cont],:] - self.alpha * grad_V_output_neg
            cont=cont+1
        
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
        if a not in self.vocab and b not in self.vocab:
            return scipy.spatial.distance.cosine(np.mean(self.w, axis=0), np.mean(self.w, axis=0))
        elif a not in self.vocab:
            return scipy.spatial.distance.cosine(np.mean(self.w, axis=0), self.w[self.w2id[b],:])
        elif b not in self.vocab:
            return scipy.spatial.distance.cosine(self.w[self.w2id[a],:], np.mean(self.w, axis=0))
        else:
            cosine=scipy.spatial.distance.cosine(self.w[self.w2id[a],:], self.w[self.w2id[b],:])
            #logging.debug(cosine)
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
        similarities=[]
        similarities_pred=[]
        for a,b,similarity in pairs:
            similarities.append(similarity)
             # make sure this does not raise any exception, even if a or b are not in sg.vocab
            
            #print(a,b)
            similarity_pred=sg.similarity(a,b)
            similarities_pred.append(similarity_pred)
        logging.basicConfig(filename='example.log',level=logging.DEBUG)
        #logging.debug(similarities_pred)
        #similarities.astype(float32)
        #similarities_pred.astype(float32)
        logging.debug(np.corrcoef(similarities,similarities_pred))
        
##############



import sys
from read_input import load_test_data
from naive_bayes import Naive_Bayes
import numpy as np
from pickle import load

if __name__=='__main__':
    # argument = sys.argv
    # test_directory =  argument[1]

    test_directory = 'C:\\CSCI544-NaturalLanguageProcessing\\HW1\\test\\'
    # get reviews and paths of each file from the given directory
    review_test_list, paths = load_test_data(test_directory)

    # read the trained naive bayes model
    with open('nbmodel.txt','r') as file:
        model = file.read().splitlines()

    # number of classes to predict
    classes = [None for i in range(int(model[0]))]

    naive_bayes = Naive_Bayes(classes)
    processed_test_reviews = naive_bayes.test_data_preprocessing(review_test_list)

    # initialize logprior[class]: stores the prior probability for all the class
    logprior = [None] * len(classes)

    # initialize loglikelihood[class][word] stores the likelihood probability for a word given all the classes
    loglikelihood = [None] * len(classes)

    # read the logprior values from the trained model
    i = 0
    for i in range(0,4):
        logprior[i] = float(model[i+1])

    # read the different class name and the loglikehood values from the trained model
    i+=2
    classes[0] = model[i]
    i+=1
    cl = 1
    dic = {}
    while i<len(model):
        words = model[i].split(':')
        if len(words)==1:
            classes[cl] = model[i]
            loglikelihood[cl-1] = dic
            dic = {}
            cl+=1
        else:
            dic[words[0]] = float(words[1])
        i+=1
    loglikelihood[cl-1] = dic

    # read the vocabulary
    with open('vocabulary.pkl','rb') as f:
        vocabulary = load(f)
    
    # initialize the P(class|word): stores sum of logprior and loglikelihood
    final_probabilty = [None] * len(classes)
    # predicted class will be store in this list
    classification = []
    
    for review in processed_test_reviews:
        sumloglikelihoods = [0] * len(classes)
        words = review.split()
        for word in words:
            if word in vocabulary:
                for ci in range(len(classes)):
                    # sum represents log(P(word|c)) = log(P(word1|c)) + ... + log(P(wordn|c))
                    sumloglikelihoods[ci] += loglikelihood[ci][word]
                    #Computes P(ci|review)
                    final_probabilty[ci] = logprior[ci] + sumloglikelihoods[ci]    
        # append the class having maximum probabilty        
        classification.append(final_probabilty.index(max(final_probabilty)))

    ''' write the output in a file '''
    with open('nboutput.txt','w') as f:
        p=0
        for c in classification:
            f.write("%s %s %s\n" % (classes[c].split('.')[1],classes[c].split('.')[0],paths[p]))
            p+=1

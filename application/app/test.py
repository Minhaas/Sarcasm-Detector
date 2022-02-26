import numpy as np
import pickle
import os
import topic
import heapq
fileObject1 = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vecdict.p'), 'r')
fileObject2= open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'classif.p'), 'r')
topic_mod = topic.topic(model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'topics.tp'),\
                        dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'topics_dict.tp'))
vec = pickle.load(fileObject1)
classifier = pickle.load(fileObject2)
'''topics = topic_mod.transform(sentence)
print(topic_mod.get_topic(0))
print ('grams:')
'''
coeff = vec.inverse_transform(classifier.coef_)[0]
#print(coeff)
#coeff['Topic :6']
'''largest = heapq.nlargest(int(100/2.0), coeff, key=coeff.get)
smallest = heapq.nsmallest(int(100/2.0), coeff, key=coeff.get)
for j in range(int(100/2.0)):
    print largest[j], coeff[largest[j]]
for j in range(int(100/2.0)):
    print smallest[j], coeff[smallest[j]]

print 'sentiment:'

print 'Positive sentiment', coeff['Positive sentiment']
print 'Positive sentiment 1/2', coeff['Positive sentiment 1/2']
print 'Positive sentiment 2/2', coeff['Positive sentiment 2/2']
print 'Positive sentiment 1/3', coeff['Positive sentiment 1/3']
print 'Positive sentiment 2/3', coeff['Positive sentiment 2/3']
print 'Positive sentiment 3/3', coeff['Positive sentiment 3/3']
print 'Negative sentiment', coeff['Negative sentiment']
print 'Negative sentiment 1/2', coeff['Negative sentiment 1/2']
print 'Negative sentiment 2/2', coeff['Negative sentiment 2/2']
print 'Negative sentiment 1/3', coeff['Negative sentiment 1/3']
print 'Negative sentiment 2/3', coeff['Negative sentiment 2/3']
print 'Negative sentiment 3/3', coeff['Negative sentiment 3/3']

print 'Blob sentiment', coeff['Blob sentiment']
print 'Blob subjectivity', coeff['Blob subjectivity']
print 'Blob sentiment 1/2', coeff['Blob sentiment 1/2']
print 'Blob sentiment 2/2', coeff['Blob sentiment 2/2']
print 'Blob subjectivity 1/2', coeff['Blob subjectivity 1/2']
print 'Blob subjectivity 2/2', coeff['Blob subjectivity 2/2']
print 'Blob sentiment 1/3', coeff['Blob sentiment 1/3']
print 'Blob sentiment 2/3', coeff['Blob sentiment 2/3']
print 'Blob sentiment 3/3', coeff['Blob sentiment 3/3']
print 'Blob subjectivity 1/3', coeff['Blob subjectivity 1/3']
print 'Blob subjectivity 2/3', coeff['Blob subjectivity 2/3']
print 'Blob subjectivity 3/3', coeff['Blob subjectivity 3/3']

print 'topics:'

topics_tag=[]
topics_coeff=[]
topics_num=[]
for j in range(200):
    topics_tag.append('Topic :' +str(j))
    topics_coeff.append(coeff[topics_tag[j]])
    if 'Topic: 0' in coeff:
        print("abc")
    topics_num.append(j)
topics_tag=np.array(topics_tag)
topics_num=np.array(topics_num)
topics_coeff=np.array(topics_coeff)

topics_num=topics_num[topics_coeff.argsort()]
topics_tag=topics_tag[topics_coeff.argsort()]
topics_coeff=topics_coeff[topics_coeff.argsort()]
for j in range(10):
    print topics_coeff[j], topic_mod.get_topic(topics_num[j])
for j in range(190,200):
    print topics_coeff[j], topic_mod.get_topic(topics_num[j])
'''
print 'Validating'

output = classifier.predict(testvec)
print classification_report(test_targets, output, target_names=cls_set)

#BASIC TEST
basic_test=["This is just a long sentence, to make sure that it's not how long the sentence is that matters the most",\
            'I just love when you make me feel like shit','Life is odd','Just got back to the US !', \
            "Isn'it great when your girlfriend dumps you ?", "I love my job !", 'I love my son !']
feature_basictest=[]
for tweet in basic_test: 
    feature_basictest.append(feature_extract.dialogue_act_features(tweet,topic_mod))
feature_basictest=np.array(feature_basictest) 
feature_basictestvec = vec.transform(feature_basictest)

print basic_test
print classifier.predict(feature_basictestvec)
print classifier.decision_function(feature_basictestvec)



fileObject2.close()
fileObject1.close()

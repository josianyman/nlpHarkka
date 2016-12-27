from opinion_core import vectorizer, clf

towns=["Kotka", "Rauma", "Rovaniemi", "Lappeenranta"]

def toSparseVec(filename):
    return vectorizer.transform(open(filename))     #Text to TFIDF-vector

def toPredict(sparseVec):
    return clf.predict(sparseVec)                   #TFIDF to tag

def pos_percent(tag):                               #Calculate and positive tags percentage
    pos=neg=0
    for t in tag:
        if t=="pos":
            pos += 1
        else:
            neg +=1
    return pos/(pos+neg+0.0)

def printResultItem(town, pos_percent):
    print(town + ": " + str(pos_percent))           #Print result in fine format
    
result = map(lambda x:
             [x, pos_percent(toPredict(toSparseVec(x+".txt")))],    #Map towns to positive percentage
             towns)

for item in result:
    printResultItem(item[0], item[1])               #Print result

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def get_train_data():
    return open("train_no_tag.txt")     #Train set without tags

def get_test_data():
    return open("evaluate_no_tag.txt")

def get_tags(flag):
    if flag=="train":
        data = open("train.txt")        #Train set with tags
    else:
        data = open("evaluate.txt")     #Test set with tags
    tags=[]
    for line in data:
        tags.append(line[0:3])          #Parse tags from text to vect
    return tags

vectorizer=TfidfVectorizer(sublinear_tf=True)

X_train=vectorizer.fit_transform(get_train_data())  #Convert text to sparse vector

X_test=vectorizer.transform(get_test_data())

clf = LinearSVC(C=0.42)                 #Optimal C is 0.42
clf.fit(X_train, get_tags("train"))     #Train mashine learning engine

predict_test_tags = clf.predict(X_test) #Get predict tags to test set
test_tags=get_tags("test")              #Get test tags for evaluating

accuracy = accuracy_score(test_tags, predict_test_tags)
print("Accuracy " + str(accuracy))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from flask import *
import nltk
from catboost import CatBoostClassifier
from nltk.corpus import stopwords
nltk.download('stopwords')
app=Flask(__name__)
app.config['SECRET_KEY']='mouli'

def encoding(file):
    enc=LabelEncoder()
    file=enc.fit_transform(file)
    return file



def splitting(file):
    global X_train,X_test,y_train,y_test
    X,y=file
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=52)
    return file


@app.route('/')
def home():
    return render_template('templates/index.html')

@app.route('/index',methods=['GET','POST'])
def index():
    print('aa')
    data=pd.read_csv('Dataset/dataset.csv')
    data_enc = encoding(data['subreddit'])
    data['reddit'] = data_enc
    data.drop(['subreddit'], axis=1, inplace=True)
    ps=PorterStemmer()
    cv=CountVectorizer()
    print('bbbbbbbbbbbbbbb')
    corpus = []
    for i in range(len(data)):
        words = re.sub('/n', ' ', data['text'][i])
        words = words.lower()
        words = nltk.sent_tokenize(data['text'][i])
        words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
        words = ' '.join(words)
        corpus.append(words)
    data['clean_text'] = corpus
    cv = CountVectorizer(stop_words='english')
    # global X,y
    X = cv.fit_transform(data['clean_text'], data['reddit']).toarray()
    X = pd.DataFrame(X)
    y = data['label']
    # return redirect(splitting,X,y)
    # global X_train,X_test,y_train,y_test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)
    # splitting(data)
    # model = CatBoostClassifier()
    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # model.fit(X_train, y_train)
    filename = 'finalized_model.sav'
    # pickle.dump(model, open(filename, 'wb'))
    # pred = model.predict(X_test)
    if request.method=='POST':
        a=request.form['name']
        s=int(request.form['selected'])
        if s==1:
            print('--------------------------------')
            bb = cv.transform([a])
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.predict(bb)
            print(result)
            # done=model.predict(bb)
            if result==0:
                result = "Stress"
                print(result)
                flash("Detected as ","danger")
            else:
                result= "no Stress"
                flash("Detected as ","warning")
            return render_template('index.html',result = result)
        if s==2:
            bb = cv.transform([a])
            # model3 = Sequential()
            # model3.add(Dense(16, activation='sigmoid'))
            # model3.add(Dense(16, activation='sigmoid'))
            # model3.add(Dense(64, activation='relu'))
            # model3.add(Dense(1, activation='softmax'))
            # model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # model3.fit(x=X_train, y=y_train, epochs=50)
            # model3.save('stress.h5')
            model =load_model('stress.h5')
            result=model.predict(bb)
            if result==0:
                result = "Stress"
                print(result)
                flash("Detected as ","danger")
            else:
                result= "no Stress"
                flash("Detected as ","warning")
            return render_template('index.html',result = result)
    return render_template('index.html')
# rgb(255 255 255 / 10%);--  337

if __name__== '__main__':
    app.run(debug=True)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import metrics
import pandas as pd
import numpy as np
import gensim
import re

class Model:
    ''' 
    Class that creates the pipeline for text Classification on the news dataset.
    '''
    def __init__(self, trainfile = "train.csv", testfile = "test.csv", labelsfile = "labels.csv"):
        
        # Loading all the necessary files
        self.train_df = pd.read_csv(trainfile)
        self.test_df = pd.read_csv(testfile)
        self.labels_df = pd.read_csv(labelsfile)

        # Initializing our 4 Classifiers 
        self.log_reg = LogisticRegression()
        self.random_forest = RandomForestClassifier(max_depth=10)
        self.adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
        self.KNeighbors = KNeighborsClassifier(3)
    
    def preprocess(self, row):
        ''' 
        A Utility function that is responsible for text cleaning 
        specifically designed to work with the apply() method for 
        pandas columns.
        --------------------------------------------------------
        row: Accepts a string and passes it though a series of 
             string operations to clean the text.
        '''
        row = str(row) 
        row = row.lower() # Lowercasing the text
        row = re.sub('[^a-zA-Z]', ' ', row) # Removing any punctuations
        row_tokens = word_tokenize(row)
        row = (lemmatizer.lemmatize(word) for word in row_tokens if word not in stopwords) # Removing stopwords
        row_processed = ' '.join(row_tokens)
        row_processed = word_tokenize(row_processed)
        return row_processed
    
    def manipulate_data(self):
        '''
        This function performs data manipulation to get the train file
        in the correct format for further processing.
        '''
        self.train_df['content'] = self.train_df.title + ' ' + self.train_df.author
        self.train_df.content = self.train_df.content.apply(self.preprocess)
    
    def initialize_vectors(self, vecsize, windowsize, epochs):
        ''' 
        Word2Vec: This function trains the Word2Vec (CBOW) model to 
        convert the words to vectors.
        -----------------------------------------------------------
        vecsize: Size of the word embedding
        windowsize: Size of the context words
        epochs: Number of iterations for the Word2Vec model
        '''
        self.w2v = gensim.models.Word2Vec(self.train_df.content,
                                          vector_size=vecsize,
                                          window=windowsize,
                                          min_count=1,
                                          epochs=epochs
                                          )
    
    def get_vectors(self, row): 
        ''' 
        A Utility function that accepts a row of words and replaces them with the 
        vector embeddings.
        '''
        for i,word in enumerate(row):
            if(word in self.w2v.wv.index_to_key): # If embedding for a word exists, replace it. 
                row[i] = np.asarray(self.w2v.wv[word])
            else: 
                row[i] = 0 # If the word is unknown, it will be a zero embedding 
        row = (np.mean(row, axis=0)).tolist() # Averaging the vectors of all words in a document.
        return row
    
    def engineer_features(self):
        ''' 
        This function is responsible for converting the vectors of variable-length 
        sentences of a document to a uniform-length representation by reducing the 
        dimentionality of the corpus without losing much of the information.
        '''
        self.train_df.content = self.train_df.content.apply(self.get_vectors)
        self.df = self.train_df.content.apply(pd.Series)
        self.df['label'] = self.train_df.label
        self.df = self.df.dropna()

    def split(self, testsize):
        ''' 
        Train, test split function. 
        '''
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.loc[:, self.df.columns != 'label'],
                                                                                self.df.label, 
                                                                                test_size=testsize, 
                                                                                random_state=42
                                                                                )
    
    def fit(self, m):
        ''' 
        Fit different models to the dataset as per user input. 
        ------------------------------------------------------
        m: Machine Learning Model
        '''
        if m=='RandomForest':
            self.model = self.random_forest.fit(self.X_train, self.y_train)
        elif m=='LogisticRegression':
            self.model = self.logreg.fit(self.X_train, self.y_train)
        elif m=='AdaBoost':
            self.model = self.adaboost.fit(self.X_train, self.y_train)
        elif m=='KNN':
            self.model = self.KNeighbors.fit(self.X_train, self.y_train)
    
    def predict(self, input_value):
        ''' 
        A simple user input/default prediction function.
        '''
        if input_value == None:
            result = self.model.fit(self.X_test)
        else: 
            result = self.random_forest.fit(np.array([input_values]))
        return result
    
    def test(self):
        ''' 
        A function that will automatically test the selected model by passing
        the test file through the above procedure and using the word embeddings 
        learnt on the training file.
        '''
        self.test_df['labels'] = self.labels_df.label
        self.test_df['content'] = self.test_df.title + ' ' + self.test_df.author
        self.test_df.content = self.test_df.content.apply(self.preprocess)
        self.test_df.content = self.test_df.content.apply(self.get_vectors)
        self.df = self.test_df.content.apply(pd.Series)
        self.df['label'] = self.labels_df.label
        self.df = self.df.dropna()
        self.prediction = self.model.predict(self.df.loc[:, self.df.columns != 'label'])
        score = metrics.f1_score(self.df.label, self.prediction) # F1 metric score
        cm = metrics.confusion_matrix(self.df.label, self.prediction) # Confustion matrix
        return score, cm


if __name__ == '__main__':
    model_instance = Model()
    model_instance.manipulate_data()
    model_instance.initialize_vectors(300)
    model_instance.engineer_features()
    model_instance.split(0.25)
    model_instance.fit('AdaBoost')
    score, cm = model_instance.test()
    print (score, cm)


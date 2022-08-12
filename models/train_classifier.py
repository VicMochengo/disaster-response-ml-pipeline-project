import sys
# data processing libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
#nlp text processing libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')
#ml processing libraries
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

import pickle

import warnings
warnings.simplefilter("ignore")


def load_data(database_filepath):
    """
    load and merge datasets
    input:
         database name
    outputs:
        X - training data i.e raw messages 
        Y - target variables i.e columns that catogorise messages
        column names i.e. category names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disaster_messages.db",  engine)

    # trim df to only have rows with wanted labels/tags i.e related column should only have 0 OR 1
    df = df[(df["related"] == 0)|(df["related"] == 1)]

    #split data to training sets vs target variables 
    X = df["message"]
    y = df.drop(["id", "message", "original", "genre"], axis = 1)

    # listing the columns
    category_names = list(np.array(y.columns))

    return X, y, category_names


def tokenize(text):
    """
    function to normalize and tokenize text i.e disaster messages received
    
    Inputs:
    Pandas series containing disaster messages
    
    
    Outputs:
    List of words that have been processed through provided arguments
    """
    
    # converting all text to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text).lower())
    
    
    # tokenize words
    tokens = word_tokenize(text)
    
    
    #lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    
    # normalize word tokens and remove stop words
    normalizer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    normalized = [normalizer.stem(word) for word in clean_tokens if word not in stop_words]
    
    return normalized


def build_model():
    """ 
    build ml pipeline, and create and return a variable containing an instance of the ml pipeline
    
    """

    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier())) 
    ])
    
    parameters = {
        "vect__min_df": [1, 5],
        "tfidf__use_idf":[True, False],
        "clf__estimator__n_estimators":[10, 25], 
        "clf__estimator__min_samples_split":[2, 5]
        }
        
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 3)

    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Apply predications to test data and evalute metrics of the ML pipeline model
    
    Inputs:
    x_test - prediction data raw messages
    y_test - target variable predictions
    col_names - list of strings containing names for each of the y_pred_array fields i.e category_names
       
    Outputs:
    data_metrics - dataframe containing accuracy, precision, recall and f1 score for a given set of y_train_array and y_pred_array labels.
    """
    y_pred = model.predict(X_test)

    metrics = []
    
    #evaluate metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(y_train_array[:, i], y_pred_array[:, i])
        precision = precision_score(y_train_array[:, i], y_pred_array[:, i])
        recall = recall_score(y_train_array[:, i], y_pred_array[:, i])
        f1 = f1_score(y_train_array[:, i], y_pred_array[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    #sav metrics
    metrics = np.array(metrics)
    data_metrics = pd.DataFrame(data = metrics, index = col_names, columns = ["Accuracy", "Precision", "Recall", "F1"])
      
    return data_metrics


def save_model(model, model_filepath):
    """
    save model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
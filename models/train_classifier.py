import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import string
import numpy as np
import joblib

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()


def load_data(database_filepath):
    """ Load X and Y data from sqlite database
    
    Output: Return X and Y data and category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)
    engine.dispose()

    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """ Process data by Removing punctuation, stop words and then lemmatizing it
    Output: token created from clened messages
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = nltk.word_tokenize(text)

    # lemmatize and remove stop words
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]


def build_model():
    """ Build model to fit X and Y """
    
    ada_clf = AdaBoostClassifier(n_estimators=10)
    
    pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('adaboost', MultiOutputClassifier(ada_clf))
                    ])
    
    parameters = {
    'tfidf__ngram_range': ((1, 1), (1, 2)),
    #'tfidf__max_df': (0.9, 1.0)
    }

    cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the performance of model """
        # Generate predictions
    Y_pred = model.predict(X_test)
    
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    Y_pred_df.head()

    # Print out the full classification report
    for i in range(36):
        print(category_names[i],\
                '\n',\
                classification_report(Y_test.iloc[:,i], Y_pred_df.iloc[:,i]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


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
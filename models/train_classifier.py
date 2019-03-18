import sys
# import libraries
import pandas as pd;
from sqlalchemy import create_engine;
import re;
from nltk import word_tokenize, pos_tag;
from nltk.corpus import stopwords;
from nltk.stem.wordnet import WordNetLemmatizer;
from sklearn.model_selection import GridSearchCV;
from sklearn.pipeline import Pipeline;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.multioutput import MultiOutputClassifier;
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix;
from sklearn.model_selection import train_test_split;
import numpy as np;
import nltk;
import pickle;

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    conn = engine.connect();
    df = pd.read_sql('select * from disaster_data_cleaned', conn)
    X = df['message']
    Y = df.select_dtypes('int64').drop('id', axis=1)
    
    conn.close();
    return X, Y, Y.columns;

def tokenize(text):
    word_net = WordNetLemmatizer();
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower());
    words = word_tokenize(text);
    words = [word for word in words if word not in stopwords.words('english')];
    lemmed = [word_net.lemmatize(word) for word in words];
    return lemmed;

def build_model():
    forest = RandomForestClassifier(n_estimators=10, random_state=1024);
    pipeline = Pipeline([('count', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('model', MultiOutputClassifier(estimator=forest, n_jobs=1))
                        ]);

    parameters = {'model__estimator__max_depth' : [5, 10], 'model__estimator__max_features' : [5, 10], 'model__estimator__criterion' : ['gini', 'entropy']};

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2);

    return cv;


def evaluate_model(model, X_test, Y_test, category_names):
    Y_preds = model.predict(X_test);
    Y_preds = pd.DataFrame(Y_preds);
    Y_preds.columns = Y_test.columns;
    Y_preds.index = Y_test.index;

    for column in Y_test.columns:
        print('Column : ' , column)
        print(classification_report(Y_test[column], Y_preds[column]))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'));


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
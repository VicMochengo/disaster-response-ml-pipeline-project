import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals as
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('disaster_response.db', engine)

# load model
model = joblib.load("../models/disaster_response_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)

    genre_count = df.groupby('genre').count()['message']
    genre_percentage = round((genre_count/genre_count.sum()) * 100, 2)
    genre = list(genre_count.index)
    category_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    category_num = category_num.sort_values(ascending = False)
    categories = list(category_num.index)

    colors = ['yellow', 'green', 'red']
    
    # create visuals
    graphs = [
        {
            "data": [
              {
                "type": "pie",
                "uid": "f4de1f",
                "hole": 0.4,
                "name": "Genre",
                "pull": 0,
                "domain": {
                  "x": genre_percentage,
                  "y": genre
                },
                "marker": {
                  "colors": [
                    "#90ee90",
                    "#dc143c",
                    "#ffff00"
                   ]
                },
                "textinfo": "label+value",
                "hoverinfo": "all",
                "labels": genre,
                "values": genre_count
              }
            ],
            "layout": {
              "title": "Count and Percent of Messages by Genre"
            }
        },
        {
            "data": [
              {
                "type": "bar",
                "x": categories,
                "y": category_num,
                "marker": {
                  "color": 'brown'}
                }
            ],
            "layout": {
              "title": "Count of Messages by Category",
              'yaxis': {
                  'title': "Count"
              },
              'xaxis': {
                  'title': "Genre"
              },
              'barmode': 'group'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
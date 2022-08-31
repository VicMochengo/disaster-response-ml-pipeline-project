
# Disaster Response Pipeline Project
![homepage header](https://github.com/VicMochengo/disaster-response-ml-pipeline-project/blob/master/screenshots/homepage-header.png)




## Project Overview

In this project, we will build a model to classify messages that are sent during disasters. There are 36 pre-determined categories. The aim of the application we are building is to classify messages received and label them in line with the 36 categories by checking and placing a true or false marker against each of the categories. Subsequently, this classification process allows escalation of only relevant messages to the appropriate disaster relief agecny i.e. flood agencies only get water-assistance, weather and flood related alerts. This project entails the building of text ETL and Machine Learning pipelines to facilitate the multi-label classification task i.e. a messages can actually have multiple labels indicating it's positively identified to be belong to more than one category.

The dataset used in training the model is provided by Figure Eight containing real messages that were sent during disaster events.

Finally, this project contains a web app where you can input a message and get classification results.


### Dependencies
 - Python 3.9+
 - Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
 - Natural Language Processing Libraries: NLTK
 - SQLlite Database Libraqries: SQLalchemy
 - Model Loading and Saving Library: Pickle
 - Web App and Data Visualization: Flask, Plotly


### Installation
To clone the repository:
```
git clone https://github.com/VicMochengo/disaster-response-ml-pipeline-project.git
```

## File descriptions
Project structure:
```
        disaster-response-ml-pipeline-project
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_categories.csv
                |-- disaster_messages.csv
                |-- disaster_response.db
                |-- process_data.py
          |-- models
                |-- train_classifier.py
          |-- notebooks
                |-- disaster-pipelines-etl.ipynb
                |-- disaster-pipelines-ml-model.ipynb
          |-- README
```
Executable files description:
 - process_data.py: This python executable code file takes as its input csv files containing message data and message categories (labels), performs ETL processes and then stores the output data in a SQLite database
 - train_classifier.py: This python executable code file fetches data from the database and trains the ML model
 - /data/ folder: This folder contains raw data of messages and categories datasets in csv format
 - /app/ folder: cointains the run.py to iniate the web app as well as go.html master.html which render the webpages.


### Executing program

1. Run the following commands in the project's root directory to set up your database and model.
 * To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
 * To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

2. Run the following command in the app's directory to run your web app. python run.py

3. Go to http://0.0.0.0:3001/


### Additional material

In the notebooks folder you can find two jupyter notebooks that will help you understand how the model works step by step:

ETL Preparation Notebook named "disaste-pipelines-etl.ipynb": provides implementation breakdown for the ETL pipeline i.e. data prepation, data processing and data storage in a sqlite database
ML Pipeline Preparation Notebook named "disaste-pipelines-etl.ipynb": provides implementation breakdown for the Machine Learning Pipeline developed using NLTK and Scikit-Learn
You can use ML Pipeline Preparation Notebook to re-train the model or tune it through a dedicated Grid Search section.

## Authors

- [Victor Mochengo](https://www.github.com/VicMochengo)

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements

 - [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
 - [Figure Eight](https://www.figure-eight.com/) for providing the dataset used to train the model

## Screenshots

1. Training dataset messages distribution by genre
![Training dataset messages distribution by genre](https://github.com/VicMochengo/disaster-response-ml-pipeline-project/blob/master/screenshots/data_summary_genre.png)
2. Training dataset messages distribution by category
![Training dataset messages distribution by category](https://github.com/VicMochengo/disaster-response-ml-pipeline-project/blob/master/screenshots/data_summary_categories.png)
3. Sample classification example:
![Sample classification screenshot 1](https://github.com/VicMochengo/disaster-response-ml-pipeline-project/blob/master/screenshots/example_text_classification_1a.png)
![Sample classification screenshot 2](https://github.com/VicMochengo/disaster-response-ml-pipeline-project/blob/master/screenshots/example_text_classification_1b.png)
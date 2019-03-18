### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

This project uses Python 3, along with Jupyter Notebook. The following libraries are necessary for running the notebook:
* Pandas
* Numpy
* MatplotLib
* Plotly
* Scikit-Learn
* SqlAlchemy
* NLTK
* wordcloud

Packages used by this project can also be installed as a Conda Environment using the provided Requirements.txt file.

To run this project, three steps are required.

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

For this project, I was interested in exploring the AirBnB dataset from Seattle to better understand the following questions:
For this project, I was interested in exploring Disaster Relief data from Figure Eight, by building an end-to-end ETL Pipeline to be able to do the following:
1. Pre-Processing the dataset by organizing the labels into one-hot encodings, and saving into a Database file
2. Tokenizing and cleaning the Natural Language data using NLP processing
3. Building, evaluating, and tuning a Machine Learning model to be able to predict the categories a disaster message would correspond to.

## File Descriptions <a name="files"></a>

The main code for this project is included in 3 files, 'data/process_data.py', 'models/train_classifier.py', 'app/run.py'. Code for processing and modeling is also in the notebooks 'ETL Pipeline Preparation.ipynb' and 'ML Pipeline Preparation.ipynb', which walks through the different steps involved in preparing the data for modeling, and obtaining the final results.

- Starting data is included in the `data` folder, as `disaster_categories.csv', and 'disaster_messages.csv'.
- Processed data is stored as a database in the `data` folder as `DisasterResponse.db`.
- Trained model is stored in the `models` folder as `classifier.pkl`
- The Wordcloud visual is drawn by `wordcloud-plotly.PY`, the code is referenced from [GitHub](https://github.com/PrashantSaikia/Wordcloud-in-Plotly).
- 
## Results<a name="results"></a>

The Average F1 score for all categories was around 0.93. The following chart shows the score by each column:

<img src="https://raw.githubusercontent.com/ravishchawla/ETL-Pipeline-for-Disaster-Data/master/charts/f1_scores.png" data-canonical-src="https://raw.githubusercontent.com/ravishchawla/ETL-Pipeline-for-Disaster-Data/master/charts/f1_scores.png" width="450" height="400" />

Data Visualizations showing further results are available in `charts/` directory. By running the Flask app, you can try your own messages on the model to see results from the trained model.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to FigureEight for providing the data. You can find the Licensing for the data and other descriptive information at theri website [here](https://www.figure-eight.com/dataset/combined-disaster-response-data/). This code is free to use.
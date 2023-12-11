# SentimentAnalysis_with_DistilBERT
 
This project demonstrates the creation of a text classification model using DistilBERT for binary sentiment analysis on the IMDb movie reviews dataset. The model is then served through a Flask application, and benchmarking is performed to evaluate prediction speed.

Project Structure
Dataset: Contains the IMDb movie reviews dataset in CSV format. 

Model: The distilbert_15000_best_f1_2.ipynb notebook contains the code for training the DistilBERT model on the dataset. The best model with the highest F1 score on the test set is saved.

Benchmark: The Benchmark-2.ipynb notebook is used for benchmarking prediction speed. It includes two approaches for speeding up the pipeline: one using DataParallel and the other using mixed precision. Please note that the saved model is not included in the repository due to size constraints.

FlaskApp:

app.py: The main Flask application that takes a text input, predicts the sentiment using the pre-trained model, and displays the output.

Instructions
Part 1 - Model Training
Download the dataset in CSV format and place it in the Dataset folder. Give the appropriate path while training the model

Open and run the distilbert_15000_best_f1_2.ipynb notebook to train the DistilBERT model. The notebook saves the best model based on F1 score.

Part 2 - Flask Application and Benchmarking

Run the Flask application using python FlaskApp/app.py. The application will be accessible at http://localhost:5000.

The index.html file is present in templates folder and the style.css file is present in static folder. Maintain the same folder structure while running the flask application.

Open and run the Benchmark/Benchmark-2.ipynb notebook. Provide the path to the dataset and the saved model to benchmark prediction speed.

Additional Information
The Model folder contains the saved model, which is not included in the repository due to size constraints.

The Flask application allows users to input text and receive sentiment predictions from the pre-trained model.

Benchmarking is performed on prediction speed using the entire dataset. Two approaches for speeding up the pipeline (DataParallel and mixed precision) are explored in the Benchmark folder.

Feel free to explore, modify, and enhance the project for your specific needs.

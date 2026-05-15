# IMDB Sentiment Analyzer
This project implements a binary sentiment classification model for IMDB movie reviews, using TF-IDF vectorization and a Multi-Layer Perceptron (MLP) neural network to predict whether a review is positive or negative.
Project Overview
Task: Binary sentiment analysis (positive / negative)
Dataset: imdb_top_500.csv
Model: MLP Neural Network
Feature Extraction: TF-IDF Vectorizer
Language: Python
Framework: Scikit-learn

## Implementation Steps
Load the imdb_top_500.csv dataset containing review text and sentiment labels.
Convert raw text into numerical features using the TF-IDF Vectorizer.
Split the dataset into training (80%) and test (20%) sets.
Train an MLP neural network classifier on the training data.
Evaluate model performance using accuracy on the held-out test set.
Save the trained model and TF-IDF vectorizer for future inference.

## GitHub CI/CD
Automatically trains and uploads model to Hugging Face Hub on push.

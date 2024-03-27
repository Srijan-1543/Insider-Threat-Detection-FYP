import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import h5py
import pickle

def load_dataset(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)  # Skip the first row
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels
    return X, y

def save_checkpoint(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    dataset_path = r"F:\FYP\Dimensionality Reduction\weekr4.2.csv"  # Update with your dataset path
    output_folder = "featureVector"  # Output folder name
    checkpoint_file = "checkpoint.pkl"  # Checkpoint filename
    saved_pickle_file = "saved_train_data.pkl"  # Previously saved pickle file containing training data split
    ipca_model_file = "ipca_model.pkl"  # File to save IPCA model

    X, y = load_dataset(dataset_path)

    if os.path.exists(saved_pickle_file):
        print("Loading previously saved training data from pickle file...")
        X_train, y_train = load_checkpoint(saved_pickle_file)
        ipca_model = load_checkpoint(ipca_model_file)
    else:
        print("Performing IPCA for dimensionality reduction...")
        ipca_model = IncrementalPCA(n_components=10)
        X_train = ipca_model.fit_transform(X)

        print("Saving IPCA model...")
        save_checkpoint(ipca_model, ipca_model_file)

        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.3, random_state=42)
        
        save_checkpoint((X_train, y_train), saved_pickle_file)

    # Define the XGBoost pipeline
    xgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier())
    ])

    # Define parameter grid for GridSearchCV
    param_grid_gridsearch = {
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.1, 0.01, 0.001],
        'xgb__n_estimators': [100, 200, 300],
        'xgb__subsample': [0.5, 0.75, 1.0],
        'xgb__colsample_bytree': [0.5, 0.75, 1.0]
        # Add more parameters as needed
    }

    # Perform GridSearchCV with KFold cross-validation
    grid_search = GridSearchCV(xgb_pipe, param_grid_gridsearch, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found by GridSearchCV:", grid_search.best_params_)
    print("Best validation accuracy:", grid_search.best_score_)

    # Save the trained model as a checkpoint
    save_checkpoint(grid_search, checkpoint_file)

    # Bayesian Optimization
    param_space_bayes = {
        'xgb__max_depth': (3, 10),
        'xgb__learning_rate': (0.001, 0.1),
        'xgb__n_estimators': (100, 1000),
        'xgb__subsample': (0.5, 1.0),
        'xgb__colsample_bytree': (0.5, 1.0)
        # Add more parameters as needed
    }

    # Perform Bayesian Optimization with KFold cross-validation
    bayes_opt = BayesSearchCV(xgb_pipe, param_space_bayes, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy', n_iter=50, n_jobs=-1, verbose=1)
    bayes_opt.fit(X_train, y_train)

    print("Best parameters found by Bayesian Optimization:", bayes_opt.best_params_)
    print("Best validation accuracy:", bayes_opt.best_score_)

    # Save the feature vectors obtained from IPCA
    save_checkpoint(X_train, output_folder)

    # Save the trained model as a checkpoint
    save_checkpoint(grid_search, checkpoint_file)

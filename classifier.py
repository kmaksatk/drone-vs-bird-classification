import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import scipy.fft as fft
import cv2
from IPython import display
import io
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
import keras
from keras.models import Sequential
from keras.layers import Dense
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier as RF


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print(f"Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Precision Score: {precision_score(y_train, pred, pos_label='drone') * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:")
        print(f"{clf_report}")
        print("_______________________________________________")
        disp = ConfusionMatrixDisplay(confusion_matrix(y_train, pred), display_labels = clf.classes_)
        disp.plot()
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print(f"Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Precision Score: {precision_score(y_test, pred, pos_label='drone') * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:")
        print(f"{clf_report}")
        print("_______________________________________________")
        disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels = clf.classes_)
        disp.plot()
        
def classify(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    print_score(model, X_train, y_train, X_test, y_test, train=False)
    print(f'Params: {model.get_params()}')

def gridsearch(model, param_grid, X_train, y_train, X_test, y_test, name):
    sc = StandardScaler()
    X_train_scale = sc.fit_transform(X_train)
    X_test_scale = sc.fit_transform(X_test)
    grid = GridSearchCV(model, param_grid, refit=True, verbose=1, cv=5)
    grid.fit(X_train_scale, y_train)
    best_params = grid.best_params_
    print(f"Best params: {best_params}")
    if name == 'SVC': 
        clf = SVC(**best_params)
    elif name == 'RF': 
        clf = RF(**best_params)
    clf.fit(X_train_scale, y_train)
    print_score(clf, X_train_scale, y_train, X_test_scale, y_test, train=False)
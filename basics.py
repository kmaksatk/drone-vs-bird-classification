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

def plot_trajectory(x, y):
#     plt.figure(figsize=(6,6))
#     plt.plot(x, y)
#     plt.title("Object flight path")
    
    

    fig = Figure(figsize=(6,6))
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.plot(x, y)
    ax.axis('off')
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    w,h = canvas.get_width_height()
    image.shape = (w, h, 3)
    return image

def turn_angle(x, y):
    T = len(x) - 1
    angle_sum = 0
    for i in range(1, len(x)):
        if x[i]==x[i-1]:
            angle_sum += np.pi/2
            continue
        angle_sum += np.arctan((y[i]-y[i-1])/(x[i]-x[i-1]))
    return angle_sum/T

def frequency(x):
#     T = len(M) - 1
#     fft_data = np.fft.fft2(M)
#     freqs = np.fft.fftfreq(len(M))
#     peak_coef = np.argmax(np.abs(fft_data))
#     peak_freq = freqs[peak_coef]
#     return abs(peak_freq * T)
    fft_data = np.fft.fft(x)
    abs_fft_data = np.abs(fft_data)**2
    fft_freq = np.fft.fftfreq(len(x), 1./30)
    fft_freq = fft_freq[fft_freq > 0]
    peak_coef = np.argmax(abs_fft_data)
    peak_freq = fft_freq[peak_coef]
    return peak_freq

def curvature(x, y):
    t = len(x)-2
    k_t = 0
    for i in range(1, len(x)-1):
        a = np.linalg.norm(np.array((x[i+1],y[i+1]))-np.array((x[i-1],y[i-1])))
        b = np.linalg.norm(np.array((x[i],y[i]))-np.array((x[i-1],y[i-1])))
        c = np.linalg.norm(np.array((x[i+1],y[i+1]))-np.array((x[i],y[i])))
        k_i = np.arccos((a**2-b**2-c**2)/(2*b*c))
        if np.isnan(k_i):
            continue
        k_t += k_i
    return k_t/t

def velocity(matrix):
    t = len(matrix)-1
    dist = 0
    for i in range(1, len(matrix)):
        dist += np.linalg.norm(matrix[i]-matrix[i-1])
    return dist/t

def acceleration(matrix):
    t = len(matrix)-2
    dist = 0
    for i in range(1, len(matrix)-1):
        first = np.linalg.norm(matrix[i]-matrix[i-1])
        second = np.linalg.norm(matrix[i+1]-matrix[i])
        dist += second-first
    return dist/t
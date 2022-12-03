# Drone-vs-Bird-classification
This is my final project for the ML702 course at MBZUAI. It focuses on trajectory classification of two objects: drones and birds. 

All the experiments are done in the .ipynb notebook. You can view it directly from the repo or follow the instructions below. 

#IMPORTANT:
DO NOT RERUN THE EXPERIMENTS, YOU MAY GET DIFFERENT RESULTS

# Installation
Run the following script to launch notebook in your browser. 

```
git clone https://github.com/kmaksatk/drone-vs-bird-classification

cd drone-vs-bird-classification

conda create -n maksat-env python=3.9 -y

conda activate maksat-env

conda install --file requirements.txt

jupyter notebook
```
# File descriptions
<ul>
  <li> basics.py - computes basic features
  <li> glcm.py - computes glcm matrix and its features
  <li> classifier.py - classification and grid search functions
  <li> dataaugmenter.py - function for data augmentation
  <li> TrajectoryFeatureExtractions+GLCM.ipynb - experiments with default data (do not rerun)
  <li> AugmentedBasicFeatures.ipynb - computing features for augmented dataset, generates csvs in Data folder
  <li> AugmentedTrajectoryFeatureExtractions+GLCM.ipynb - experiments with augmented data (do not rerun)
</ul>

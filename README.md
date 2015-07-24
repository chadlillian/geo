# geo
machine learning to analyze geo-tagged photos

Requires 
  1. Caffe Convolutional Neural Network http://caffe.berkeleyvision.org/
  2. Trained network (for Caffe) http://places.csail.mit.edu/downloadCNN.html

This set of programs does the following:
  1.  Queries the Panoramio API for geo-tagged photos, and stores the JSON response in a pymongo db
  2.  Analyzes the photos with the Caffe CNN to identify features and updates the db
  3.  Uses the DBSCAN clustering algorithm to identify high density regions for each feature
  4.  More geographic processing to follow.

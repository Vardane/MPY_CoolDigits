# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 10:41:18 2018

@author: yfoucault002
"""

import numpy   as np 
#import scipy.misc # to visualize only   

from sklearn import svm



# =============================================================================
# 0 - Load data
# =============================================================================

x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
y = np.loadtxt("train_y.csv", delimiter=",") 
x = x.reshape(-1, 64, 64) # reshape 
y = y.reshape(-1, 1) 
#scipy.misc.imshow(x[0]) # to visualize only  # RuntimeError: Could not execute image viewer.


# =============================================================================
# 1 - Baseline classifier
# =============================================================================

# reshape arrays to match shape expected by svm scikit learn function
x = np.reshape(x,(50000,4096))
y = np.ravel(y)


bsClf = svm.LinearSVC()   # baseline classifier
bsClf.fit(x, y) 
# using default settings:
#(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#     verbose=0)
dec = bsClf.decision_function([[1]])
dec.shape[1]

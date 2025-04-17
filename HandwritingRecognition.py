import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import linear_model, preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy.io

# NOTICE: can't upload file to github too large. download from https://www.nist.gov/itl/products-and-services/emnist-dataset

def load_emnist(file_path='emnist-digits.mat'):
    """
    Loads training and test data with ntr and nts training and test samples
    The `file_path` is the location of the `eminst-balanced.mat`.
    """    
    
    # Load the MATLAB file
    mat = scipy.io.loadmat(file_path)
    
    # Get the training data
    Xtr = mat['dataset'][0][0][0][0][0][0][:]
    ntr = Xtr.shape[0]
    ytr = mat['dataset'][0][0][0][0][0][1][:].reshape(ntr).astype(int)
    
    # Get the test data
    Xts = mat['dataset'][0][0][1][0][0][0][:]
    nts = Xts.shape[0]
    yts = mat['dataset'][0][0][1][0][0][1][:].reshape(nts).astype(int)
    
    print("%d training samples, %d test samples loaded" % (ntr, nts))

    return [Xtr, Xts, ytr, yts]

# Loading digit data
Xtr_dig, Xts_dig, ytr_dig, yts_dig = load_emnist(file_path='emnist-digits.mat')

# Loading letter data
Xtr_let, Xts_let, ytr_let, yts_let = load_emnist(file_path='emnist-letters.mat')

def plt_digit(x,y=None):
    nrow = 28
    ncol = 28
    xsq = x.reshape((nrow,ncol))
    plt.imshow(xsq.T,  cmap='Greys_r')
    plt.xticks([])
    plt.yticks([])    
    if y != None:
        plt.title('%d' % y)        

# Plot 8 random samples from the training data of the digits
# Select random digits
nplt = 8
Iperm = np.random.permutation(nplt)
plt.figure(figsize=(10,20))
for i in range(nplt):
    ind = Iperm[i]
    plt.subplot(1, nplt, i+1)
    plt_digit(Xtr_dig[ind,:], ytr_dig[ind])

# Plot 8 random samples from the training data of the letters
nplt = 8
Iperm = np.random.permutation(nplt)
plt.figure(figsize=(10,20))
for i in range(nplt):
    ind = Iperm[i]
    plt.subplot(1, nplt, i+1)
    plt_digit(Xtr_let[ind,:], ytr_let[ind])

# Creating a non-digit class
remove_list = np.array([9,12,15])

# Create arrays with labels 9, 12 and 15 removed
Xtr_let_rem = Xtr_let[np.all(ytr_let[:,None] != remove_list[None,:], axis=1)]
ytr_let_rem = ytr_let[np.all(ytr_let[:,None] != remove_list[None,:], axis=1)]
Xts_let_rem =Xts_let[np.all(yts_let[:,None] != remove_list[None,:], axis=1)]
yts_let_rem = yts_let[np.all(yts_let[:,None] != remove_list[None,:], axis=1)]

# Number of training and test digits and letters
ntr_dig = 5000
ntr_let = 1000
nts_dig = 5000
nts_let = 1000

# Create sub-sampled training and test data
Itr_dig = np.random.permutation(ntr_dig)
Its_dig = np.random.permutation(nts_dig)
Itr_let = np.random.permutation(ntr_let)
Its_let = np.random.permutation(nts_let)

Xtr1_dig, ytr1_dig = Xtr_dig[Itr_dig], ytr_dig[Itr_dig]
Xts1_dig, yts1_dig = Xts_dig[Its_dig], yts_dig[Its_dig]
Xtr1_let, ytr1_let = Xtr_let[Itr_let], ytr_let[Itr_let]
Xts1_let, yts1_let = Xts_let[Its_let], yts_let[Its_let]

# Create combined letter and digit training and test data
Xtr, ytr = np.concatenate((Xtr1_dig, Xtr1_let)), np.concatenate((ytr1_dig, np.full(ntr_let, 10)))
Xts, yts = np.concatenate((Xts1_dig, Xts1_let)), np.concatenate((yts1_dig, np.full(nts_let, 10)))

# Rescale the data to the interval from -1.0 to 1.0
Xtr1 = (Xtr)/127.5 - 1.0
Xts1 = (Xts)/127.5 - 1.0

# SVM Classifier
# Create a classifier: a support vector classifier
svc = svm.SVC(probability=False, kernel="rbf", C=2.8, gamma=.0073, verbose=1)

# Fit the classifier on the training data. 
svc.fit(Xtr1, ytr)

# Measure error on the test data
yhat_ts = svc.predict(Xts1)
acc = np.mean(yhat_ts == yts)
print(acc)

# Confusion Matrix
# Print and plot the normalized confusion matrix
C = confusion_matrix(yts, yhat_ts, normalize='true',) # each row sums to one

disp = ConfusionMatrixDisplay(C)
disp.plot()

# Print the error rates
print('digits mislabeled:', np.sum((yhat_ts == 10) & (yts < 10)), '/', nts_dig)
print('letters mislabeled:', np.sum((yhat_ts < 10) & (yts == 10)), '/', nts_let)

# Optimizing gamma and C - CROSS VALIDATION
# Create combined trained and test data X and y.
X = np.vstack((Xtr1, Xts1))
y = np.hstack((ytr, yts))

# Create a pre-defined test split object 
import sklearn.model_selection
test_fold = np.concatenate((np.full(ytr.size, -1), np.zeros(yts.size)))
ps = sklearn.model_selection.PredefinedSplit(test_fold)

# Create a GridSearchCV classifier
C_test = [0.1, 1, 10]
gam_test = [0.001, 0.01, 0.1]
parameters = {'kernel':['rbf'], 'C': C_test, 'gamma': gam_test} # the numpy website said to do ('rbf') but threw an error saying that it had to be an array
clf = GridSearchCV(svm.SVC(), parameters, cv=ps, verbose=10)
clf.fit(X,y)

# Print the best parameter and score of the classifier
print("best score:", clf.best_score_)
print("best parameter:", clf.best_params_)

# Print the mean test score for each parameter value.
print(clf.cv_results_['mean_test_score'])

import numpy as np
import seaborn as sns
from functions import process_email
from functions import get_vocab_list
import scipy.io
from sklearn import svm

print('\nPreprocessing sample email.\n')

# Extracting features.
email_contents = open("emailSample1.txt", "r").read()
test_fv = process_email(email_contents)

print("Length of feature vector is %d"%len(test_fv))
print("Number of npn-zero entries is : %d"%sum(test_fv==1))

# Training SVM for spam classification.

# Training set.
datafile = 'spamTrain.mat'
mat = scipy.io.loadmat(datafile)
X,y = mat['X'],mat['y']

# We don't have to insert column of 1's in X because SVM's package automatically does that.

# Test set
datafile = 'spamTest.mat'
mat = scipy.io.loadmat(datafile)
Xtest,ytest = mat['Xtest'],mat['ytest']

pos = np.array([X[i] for i in range(X.shape[0]) if y[i]==1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i]==0])

print("Total number of training emails = ",X.shape[0])
print("Number of training spam emails = ",pos.shape[0])
print("Number of training non-spam emails = ",neg.shape[0])

# First we make an instance of SVM with c=0.1 and 'Linear' kernel
linear_svm = svm.SVC(C=0.1,kernel='linear')
# Now we fit the SVm to our X matrix, given the labels y
linear_svm.fit(X,y.flatten())

#Predicting training set.
train_predictions = linear_svm.predict(X).reshape((y.shape[0],1))
train_accuracy = 100. * float(sum(train_predictions==y))/y.shape[0]
print("Training set accuracy = %0.2f%%"%train_accuracy)

# Predictiong test set.
test_predictions = linear_svm.predict(Xtest).reshape((ytest.shape[0],1))
test_accuracy = 100. * float(sum(test_predictions==ytest))/ytest.shape[0]
print("Test set accuracy = %0.2f%%"%test_accuracy)


# Determine the words most likely to indicate an email is a spam.

vocab_dict_flipped = get_vocab_list(reverse=True)

# Sort indices from most important to least important.
sort_indices = np.argsort(linear_svm.coef_,axis=None)[::-1]

print("The 15 most important words to classify a spam email are:")
print([vocab_dict_flipped[x] for x in sort_indices[:15]])
print()
print("The 15 least important words to classify a spam email are:")
print([vocab_dict_flipped[x] for x in sort_indices[-15:]])
print()


# Most common word
most_common_word = vocab_dict_flipped[sort_indices[0]]
print('# of spam containing \"%s\" = %d%d = %0.2f%%'%(most_common_word,sum(pos[:,1190]),pos.shape[0],100.*float(sum(pos[:,1190]))/pos.shape[0]))

print('# of NON spam containing \"%s\" = %d/%d = %0.2f%%'%(most_common_word, sum(neg[:,1190]),neg.shape[0],100.*float(sum(neg[:,1190]))/neg.shape[0]))








 






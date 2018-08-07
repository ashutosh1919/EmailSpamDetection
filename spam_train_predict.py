import numpy as np
import seaborn as sns
from functions import process_email
from functions import get_vocab_list
import scipy.io
from functions import svm_train
from functions import Model
from functions import svm_predict
from sklearn.metrics import accuracy_score

# Reading Email Contents.
file_contents = open('emailSample1.txt').read()
# Processing the email.
features = process_email(file_contents)
print("Length of feature vector : %d"%len(features))
print("Number of non-zero entries : %d"%sum(features==1))

# Train linear SVM for spam classification.
# Loading the spam email data set.
datafile_train = scipy.io.loadmat('spamTrain.mat')
X = np.matrix(datafile_train['X'])
y = np.matrix(datafile_train['y']) 

print("\nTraining Linear SVM (Spam Classification)\n")
print("This may take 1 to 2 Miinutes ....\n")

C = 0.1
model = svm_train(X,y,C,func_name='linear_kernel')

prediction_train = svm_predict(model,X)

print('Training Accuracy : %f\n'%accuracy_score(y, prediction_train))


# Test spam Classification.
# Load the dataset.
datafile_test = scipy.io.loadmat('spamTest.mat')
Xtest = np.matrix(datafile_test['Xtest'])
ytest = np.matrix(datafile_test['ytest'])

print('\n\n Evaluating the trained Linear SVM on a test set ...\n')

prediction_test = svm_predict(model,Xtest)
print('Test Accuracy : %f\n'%accuracy_score(ytest, prediction_test))








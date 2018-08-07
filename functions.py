import numpy as np
import re
from stemming.porter2 import stem
import nltk, nltk.stem.porter
from nltk import PorterStemmer
from math import ceil
from numpy.random.mtrand import rand
from scipy.constants.constants import alpha
import random

class Model:
    def __init__(self,X,y,func_name,b,alphas,w):
        self.X = X
        self.y = y
        self.func_name = func_name
        self.b = b
        self.alphas = alphas
        self.w = w;
    def get_X(self):
        return self.X
    def get_y(self):
        return self.y
    def get_func_name(self):
        return self.func_name
    def get_b(self):
        return self.b
    def get_alphas(self):
        return self.alphas
    def get_w(self):
        return self.w

def get_vocab_list(reverse=False):
    # Read the fixed vocabulary list.
    vocab_list = {}
    with open('vocab.txt','r') as f:
        for line in f:
            (val,key) = line.split()
            if not reverse:
                vocab_list[key] = int(val)
            else:
                vocab_list[int(val)] = key
    return vocab_list
            

def process_email(email_contents):
    # Load Vocabulary list.
    vocab_list = get_vocab_list()
    #Init return values
    result = np.zeros((len(vocab_list),1))
    # Lower case
    email_contents = email_contents.lower()
    # Strip all HTML tags and replace them with space.
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    # Handle Numbers
    email_contents = re.sub('[0-9]+','number',email_contents)
    # Handle URLs
    email_contents = re.sub('(http|https)://[^\s]*','httpaddr',email_contents)
    # Handle Email Addresses
    email_contents = re.sub('[^\s]+@[^\s]+','emailaddr',email_contents)
    # Handle $ sign
    email_contents = re.sub('[$]+','dollar',email_contents)
    # Tokenize Email
    # Output the email to screen as well.
    print('\n==== Processed Email ====\n\n')
    # Process file.
    #Stemmer to reduce words
    stemmer = nltk.stem.porter.PorterStemmer()
    token_list = []
    # Generating tokens.
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email_contents)
    # Check each token and use stemmer to reduce it.
    for token in tokens:
        # Remove any non alpha-numeric character.
        token = re.sub('[^a-zA-Z0-9]','',token)
        # Stem the token
        stemmed = stemmer.stem(token)
        if not len(token):
            continue
        token_list.append(stemmed)
        
    index_list = [vocab_list[token] for token in token_list if token in vocab_list]
    for idx in index_list:
        result[idx] = 1
    return result
    
def linear_kernel(x1,x2):
    # Ensure that x1 and x2 are column vectors.
    x1 = x1.flatten()
    x2 = x2.flatten()
    sim = np.inner(x1,x2)
    return sim

def gaussian_kernel(x1,x2,sigma):
    x1 = x1.flatten()
    x2 = x2.flatten()
    # You need to return the following variables correctly.
    sim = 0
    
    sim = np.exp(-np.sum(np.power((x1-x2),2))/(2*(sigma**2)))
    return sim
    
    
def svm_train(X,Y,C,func_name='linear_kernel',tol=1e-3,max_passes=5):
    # Data params.
    m = X.shape[0]
    n = X.shape[1]
    
    # Map 0 to -1 because SVM has property of -1 as neg and 1 as pos.
    #Y[Y==0] = -1
    
    for i in range(0,Y.shape[0]):
        if(Y[i,0]==0):
            Y[i,0] = -1
    
    # Variables
    alphas = np.zeros(shape=(m,1))
    b = 0
    E = np.zeros(shape=(m,1))
    passes = 0
    L = 0
    H = 0
    
    # We have different functions to train the SVM model to train in linear or gaussian style.
    if(func_name == 'linear_kernel'):
        print('\n\nApplying Linear Kernel.\n')
        K = np.inner(X,X)
    elif(func_name == 'gaussian_kernel'):
        #Vectorized RBF Kernel.
        # This is equivalent to computing kernel on evry pair of examples.
        print('\n\nApplying Gaussian Kernel.\n')
        X2 = np.sum(np.square(X),axis=1)
        K = X2 + (X2.transpose() + (-2)*np.inner(X,X))
        K = np.power(gaussian_kernel(1,0),K)
    else:
        # Pre-compute the kernel matrix.
        # The following process may be slow due to lack of vectorization.
        print('\n\nApplying Other Kernel.\n')
        K = np.zeros(m)
        for i in range(0,m):
            for j in range(i,m):
                K[i,j] = linear_kernel(X[i,:].transpose(), X[j,:].transpose())
                K[j,i] = K[i,j]   # Matrix is symmetric.
        
    #print('K Value : ')
    #print(K)
    # Training the parameters.
    dots = 12
    while passes<max_passes:
        num_changed_alphas = 0
        for i in range(0,m):
            # Calculate Ei = f(x(i)) - y(i)
            E[i,0] = b + np.sum(np.multiply(alphas,np.multiply(Y,K[:,i]))) - Y[i,0]
            
            if (Y[i,0]*E[i,0]<-tol and alphas[i,0]<C) or (Y[i,0]*E[i,0]>tol and alphas[i,0]>0):
                # In practice, there are many heuristics one can use to select i and j. In the simplified code, we select them randomly.
                j = ceil(m*random.random())
                while j==i:
                    j = ceil(m*random.random())
                    
                #print(j)
                
                if(j>=m):
                    continue
                # Calculate Ej = f(x(j)) - y(j)
                E[j,0] = b + np.sum(np.multiply(alphas,np.multiply(Y,K[:,j]))) - Y[j,0]
                # Save old alphas
                alpha_i_old = alphas[i,0]
                alpha_j_old = alphas[j,0]
                # Compute L and H.
                if(Y[i,0]==Y[j,0]):
                    L = max(0,alphas[j,0]+alphas[i,0]-C)
                    H = min(C,alphas[j,0]+alphas[i,0])
                else:
                    L = max(0,alphas[j,0]-alphas[i,0])
                    H = min(C,C+alphas[j,0]-alphas[i,0])
                    
                if(L==H):
                    continue
                
                # Compute eta.
                eta = 2*K[i,j] - K[i,i] - K[j,j]
                if(eta>=0):
                    continue
                
                # Compute and clip new value for alpha j.
                alphas[j,0] = alphas[j,0] - (Y[j,0]*(E[i,0]-E[j,0]))/eta
                
                # Clip
                alphas[j,0] = min(H,alphas[j,0])
                alphas[j,0] = max(L,alphas[j,0])
                
                if(abs(alphas[j,0]-alpha_j_old)<tol):
                    alphas[j,0] = alpha_j_old
                    continue
                
                # Determine value for alpha i.
                alphas[i,0] = alphas[i,0] + Y[i,0]*Y[j,0]*(alpha_j_old-alphas[j,0])
                
                # Compute b1 and b2.
                b1 = b - E[i,0] - Y[i,0]*(alphas[i,0]-alpha_i_old)*K.transpose()[i,j] - Y[j,0]*(alphas[j,0]-alpha_j_old)*K.transpose()[i,j]
                b2 = b - E[j,0] - Y[i,0]*(alphas[i,0]-alpha_i_old)*K.transpose()[i,j] - Y[j,0]*(alphas[j,0]-alpha_j_old)*K.transpose()[j,j]
                
                # compute b.
                if(0<alphas[i,0] and alphas[i,0]<C):
                    b = b1
                elif(0<alphas[j,0] and alphas[j,0]<C):
                    b = b2
                else:
                    b = (b1+b2)/2
                    
                num_changed_alphas = num_changed_alphas+1
                    
        if(num_changed_alphas == 0):
            passes = passes+1
        else:
            passes = 0
            
        print('.',end="")
        dots = dots+1
        if dots>78:
            dots = 0
            print()
        
    print("\n\n Done! \n\n")
    
    # Save the model
    idx = alphas>0
    #print(idx)
    X_top = []
    for i in range(0,idx.shape[0]):
        if(idx[i,0]==True):
            temp = []
            for j in range(0,X.shape[1]):
                temp.append(X[i,j])
            X_top.append(temp)
    X_top = np.matrix(X_top)
    
    Y_top = []
    for i in range(0,idx.shape[0]):
        if(idx[i,0]==True):
            temp = []
            for j in range(0,Y.shape[1]):
                temp.append(Y[i,j])
            Y_top.append(temp)
    Y_top = np.matrix(Y_top)
    #print(Y_top)
    
    alph_top = []
    for i in range(0,idx.shape[0]):
        if(idx[i,0]==True):
            temp = []
            for j in range(0,alphas.shape[1]):
                temp.append(alphas[i,j])
            alph_top.append(temp)
    alph_top = np.matrix(alph_top)
    
    model = Model(X_top,Y_top,func_name,b,alph_top,np.dot(np.multiply(alphas,Y).transpose(),X).transpose())
    
    return model

def svm_predict(model,X):
    # Check if we are getting only column vector, if so, then we only have to predict for single example.
    if(X.shape[1]==1):
        X = X.transpose()
    
    m = X.shape[0]
    p = np.zeros(shape=(m,1))
    pred = np.zeros(shape=(m,1))
    
    if(model.func_name=='linear_kernel'):
        # We can use weights and bias directly if working with the linear kernel.
        p = np.dot(X,model.w) + model.b
    elif(model.func_name=='gaussian_kernel'):
        # Vectorized RBF Kernel.
        # This is equivalent to computing the kernel on every pair of examples.
        X1 = (np.square(X)).sum(axis=1)
        X2 = ((np.square(model.X)).sum(axis=1)).transpose()
        K = X1 + (X2 + (-2)*np.dot(X,model.X.transpose()))
        K = np.power(gaussian_kernel(1, 0),K)
        K = np.multiply(model.y.transpose(),K)
        K = np.multiply(model.alphas.transpose(),K)
        p = K.sum(axis=1)
    else:
        # Other Non-linear Kernel.
        for i in range(0,m):
            prediction = 0
            for j in range(0,model.X.shape[0]):
                prediction = prediction + model.alphas[j,0]*model.y[j,0]*linear_kernel(X[i,:].transpose(), model.X[j,:].transpose())[0][0]
            p[i,0] = prediction + model.b
        
    for i in range(0,m):
        if(p[i,0]>=0):
            pred[i,0] = 1
        else:
            pred[i,0] = 0
    
    return pred       
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn import svm
import matplotlib.pyplot as plt


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args
    
    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.ones(n_features + 1)
    
    
    
    dat_i = np.ones([n_features+1,1])
    grad = np.zeros([n_features + 1,1])
    
    for i in range(n_data):
        #for error:
        dat_i[0:n_features,0] = train_data[i,:]
        theta_i = sigmoid(np.dot(np.transpose(initialWeights),dat_i))
        label = labeli[i]
        error = error + label*np.log(theta_i) + (1-label)*np.log(1-theta_i)
        
        #for error_grad:
        grad = grad + (theta_i - label)*dat_i
        
    
    error = -1*error/n_data
    error_ = error[0]
    error_grad = grad[:,0]/n_data
    #print(error_)
    
    return error_, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    n_data = data.shape[0]
    n_features = data.shape[1]
    dat_i = np.ones(n_features+1)
    n_class = W.shape[1]
    
    for i in range(n_data):
        tmp = data[i,:]
        dat_i[0:n_features] = tmp
        theta_max = 0
        opt_class = 999
        for j in range(n_class):
            w_j = W[:,j]    
            theta_ij = sigmoid(np.dot(np.transpose(w_j),dat_i))
            if theta_ij > theta_max:
                opt_class = j
                theta_max = theta_ij
        label[i,0] = opt_class
    
    return label

def mlrObjFunction(initialWeights_b, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights_b: the weight vector of size (D + 1) x 10
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, labeli = args
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))
    
    dat_i = np.ones((n_data,n_feature+1))  
    dat_i[:,0:n_feature]=train_data
    
    initialWeights_b_ = initialWeights_b.reshape([716,10])
    wT_x = dat_i@initialWeights_b_
    
    exp_wT_x = np.exp(wT_x)
    #print(np.sum(exp_wT_x))
    norm_val = np.sum(exp_wT_x,axis=1)
    P = np.zeros((n_data,exp_wT_x.shape[1]))
    
    for i in range(10):
        P[:,i] = exp_wT_x[:,i]/norm_val
    
    error = (-1*np.sum(np.multiply(labeli,np.log(P))))
    
    err = P - labeli
    error_grad = (dat_i.T)@err
    
    error_grad_ = error_grad.reshape(7160)
    #print(error)

    return error, error_grad_


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    n_data = data.shape[0]
    n_features = data.shape[1]
    dat_i = np.ones(n_features+1)
    n_class = W.shape[1]
    
    for i in range(n_data):
        tmp = data[i,:]
        dat_i[0:n_features] = tmp
        theta_max = 0
        opt_class = 999
        for j in range(n_class):
            w_j = W[:,j]    
            theta_ij = sigmoid(np.dot(np.transpose(w_j),dat_i))
            if theta_ij > theta_max:
                opt_class = j
                theta_max = theta_ij
        label[i,0] = opt_class
    
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    #print(i)
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options={'maxiter': 100})
    W[:, i] = nn_params.x.reshape((n_feature + 1,))
    
print('\n\n--------------Logistic One v All-------------------\n\n')
# Find the accuracy on Training Dataset
predicted_label_blr_train = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_blr_train == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_blr_valid = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_blr_valid == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_blr_test = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_blr_test == test_label).astype(float))) + '%')
print('\n\n--------------End of Logistic One v All-------------------\n\n')

print('\n\n--------------SVM-------------------\n\n')
model_predict = np.zeros(10000)
train_predict = np.zeros(50000)
valid_predict = np.zeros(10000)
test_predict = np.zeros(10000)

"""SVM Model Evaluation begins here and has been commented out from execution.
#Project report explains the reason for choosing the particular model

model_select_sample = 10000
feat_size = train_data.shape[1]
model_select_data = np.zeros([model_select_sample,feat_size])
model_select_label = np.zeros([model_select_sample])
temp=0
n_sample = 1000
for i in range(10):
    tmp = np.where(train_label == i)
    t = tmp[0]
    rndm = np.random.choice(t, replace=False, size=1000)
    rndm=np.sort(rndm)
    model_select_data[temp:temp + n_sample, :] = train_data[rndm,:]
    model_select_label[temp:temp + n_sample] = train_label[rndm,0]
    temp = temp + n_sample

model1 = svm.SVC(kernel='linear')
model2 = svm.SVC(kernel='rbf',gamma=1.0)
model3 = svm.SVC(kernel='rbf',gamma='scale')
model4 = svm.SVC(kernel='rbf',gamma='scale',C=1)
model5 = svm.SVC(kernel='rbf',gamma='scale',C=10)
model6 = svm.SVC(kernel='rbf',gamma='scale',C=20)
model7 = svm.SVC(kernel='rbf',gamma='scale',C=30)
model8 = svm.SVC(kernel='rbf',gamma='scale',C=40)
model9 = svm.SVC(kernel='rbf',gamma='scale',C=50)
model10 = svm.SVC(kernel='rbf',gamma='scale',C=60)
model11 = svm.SVC(kernel='rbf',gamma='scale',C=70)
model12 = svm.SVC(kernel='rbf',gamma='scale',C=80)
model13 = svm.SVC(kernel='rbf',gamma='scale',C=90)
model14 = svm.SVC(kernel='rbf',gamma='scale',C=100)

def svm_eval(model):
    mdl = model
    mdl.fit(model_select_data,model_select_label)  
    #SVM predictions:

    train_predict = mdl.predict(train_data)
    valid_predict = mdl.predict(validation_data)
    test_predict = mdl.predict(test_data) 
    model_predict = mdl.predict(model_select_data)
    
    train_error = 100 * np.mean((train_predict == train_label[:,0]).astype(float))
    valid_error = 100 * np.mean((valid_predict == validation_label[:,0]).astype(float))
    test_error  = 100 * np.mean((test_predict == test_label[:,0]).astype(float)) 
    model_error = 100 * np.mean((model_predict == model_select_label).astype(float))
    return train_error, valid_error, test_error, model_error

e = np.zeros([14,4])
e[0,0],e[0,1],e[0,2],e[0,3] = svm_eval(model1)
print("Model 1 evaluated")
e[1,0],e[1,1],e[1,2],e[1,3] = svm_eval(model2)
print("Model 2 evaluated")
e[2,0],e[2,1],e[2,2],e[2,3] = svm_eval(model3)
print("Model 3 evaluated")
e[3,0],e[3,1],e[3,2],e[3,3] = svm_eval(model4)
print("Model 4 evaluated")
e[4,0],e[4,1],e[4,2],e[4,3] = svm_eval(model5)
print("Model 5 evaluated")
e[5,0],e[5,1],e[5,2],e[5,3] = svm_eval(model6)
print("Model 6 evaluated")
e[6,0],e[6,1],e[6,2],e[6,3] = svm_eval(model7)
print("Model 7 evaluated")
e[7,0],e[7,1],e[7,2],e[7,3] = svm_eval(model8)
print("Model 8 evaluated")
e[8,0],e[8,1],e[8,2],e[8,3] = svm_eval(model9)
print("Model 9 evaluated")
e[9,0],e[9,1],e[9,2],e[9,3] = svm_eval(model10)
print("Model 10 evaluated")
e[10,0],e[10,1],e[10,2],e[10,3] = svm_eval(model11)
print("Model 11 evaluated")
e[11,0],e[11,1],e[11,2],e[11,3] = svm_eval(model12)
print("Model 12 evaluated")
e[12,0],e[12,1],e[12,2],e[12,3] = svm_eval(model13)
print("Model 13 evaluated")
e[13,0],e[13,1],e[13,2],e[13,3] = svm_eval(model14)
print("Model 14 evaluated")

opt_modl = np.zeros([11,5])
opt_modl[:,0] = [1,10,20,30,40,50,60,70,80,90,100]
opt_modl[:,1:] = e[3:,:]
values = ['Default','10','20','30','40','50','60','70','80','90','100']
plt.plot(opt_modl[:,0], opt_modl[:,1], label = "Full Train Set")
plt.plot(opt_modl[:,0], opt_modl[:,2], label = "Full Validation Set")
plt.plot(opt_modl[:,0], opt_modl[:,3], label = "Full Test Set")
plt.plot(opt_modl[:,0], opt_modl[:,4], label = "Model Selection Set")
plt.legend()
plt.xlabel("C Values (Kernel=rbf,gamma=default)")
plt.ylabel("% Accuracy")
plt.xticks(opt_modl[:,0],values)
plt.savefig('error_v_C.png',bbox_inches='tight',dpi=600)
np.savetxt("errors.csv", e, delimiter=",")
"""

#Selecting optimum model as [kernel=rbf, gamma=default, C=10]:
mdl = svm.SVC(kernel='rbf',gamma='scale',C=10)
mdl.fit(train_data,train_label[:,0])
#SVM predictions:

train_predict = mdl.predict(train_data)
valid_predict = mdl.predict(validation_data)
test_predict = mdl.predict(test_data)

print('\n Training set Accuracy:' + str(100 * np.mean((train_predict == train_label[:,0]).astype(float))) + '%')
print('\n Validation set Accuracy:' + str(100 * np.mean((valid_predict == validation_label[:,0]).astype(float))) + '%')
print('\n Testing set Accuracy:' + str(100 * np.mean((test_predict == test_label[:,0]).astype(float))) + '%')

print('\n\n--------------End of SVM-------------------\n\n')


print('\n\n--------------Multi Class logistic-------------------\n\n')
#Script for Extra Credit Part
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1)* n_class)
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_mlr_train = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_mlr_train == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_mlr_valid = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_mlr_valid == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_mlr_test = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_mlr_test == test_label).astype(float))) + '%')
print('\n\n--------------End of Multi Class logistic-------------------\n\n')




"""
The section below is to evaluate class-level accuracies for Logistic models
class_train_acc = np.zeros(10)
class_test_acc = np.zeros(10)
class_valid_acc = np.zeros(10)


#For class accuracies only::
for i in range(n_class):
    tmp = np.where(train_label == i)
    t = tmp[0]
    
    pred_train = predicted_label_mlr_train[t]
    actl_train = train_label[t]
    class_train_acc[i] = (100 * np.mean((pred_train == actl_train).astype(float))).astype(int)

    tmp = np.where(test_label == i)
    t = tmp[0]
    
    pred_test = predicted_label_mlr_test[t]
    actl_test = test_label[t]
    class_test_acc[i] = (100 * np.mean((pred_test == actl_test).astype(float))).astype(int)
    
    tmp = np.where(validation_label == i)
    t = tmp[0]
    
    pred_valid = predicted_label_mlr_valid[t]
    actl_valid = validation_label[t]
    class_valid_acc[i] = (100 * np.mean((pred_valid == actl_valid).astype(float))).astype(int)
"""
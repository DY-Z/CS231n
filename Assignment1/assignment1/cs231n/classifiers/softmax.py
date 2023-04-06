import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  scores = X.dot(W)
  for i in range(X.shape[0]):
      #To be numerically stable
      stable = scores[i] - np.max(scores[i])
      softMax = np.exp(stable)
      softMax /= np.sum(softMax)
      crossEntrophy = -np.log(softMax[y[i]])
      loss += crossEntrophy
      #compute the gradient
      for j in range(W.shape[1]):
          if j == y[i]:
              dW[:,j] += (softMax[j]-1)*X[i]
          else:
              dW[:,j] += softMax[j]*X[i]
  #divided by num of training images
  loss /= X.shape[0]
  dW /= X.shape[0]
  
  #regularization
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  numTrain = X.shape[0]
  #compute the loss first
  scores = X.dot(W)
  #maxByRow = np.max(scores, axis = 1)# max value of each row
  #numerically stable
  stable = scores - np.max(scores, axis=1, keepdims=True)
  softMax = np.exp(stable)/np.sum(np.exp(stable), axis = 1, keepdims = True)
  toSum = -np.log(softMax[np.arange(numTrain), y], dtype = np.float64)
  loss += np.sum(toSum)
  
  #then the grad
  #compute the softmax loss for ecah X_i, i.e., L_i
  #lossByRow = softMax[np.arange(numTrain), y]
  gradMatrix = softMax.reshape(numTrain, -1)#N*C
  gradMatrix[np.arange(numTrain),y] -= 1
  dW = np.dot(X.T.reshape(X.shape[1], numTrain), gradMatrix)
  
  #divided by the num of training images;
  #add the regulation
  loss /= numTrain
  dW /= numTrain
  
  loss += reg*np.sum(W*W, dtype = np.float64)
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


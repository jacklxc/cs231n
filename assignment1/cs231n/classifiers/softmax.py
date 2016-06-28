import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    maximum = np.max(scores)
    gdenom = np.sum(np.exp(scores))
    denom = 0.0
    for j in xrange(num_classes):
      denom += np.exp(scores[j] - maximum)
      dl = np.exp(scores[j]) / gdenom
      if(j==y[i]):
        dl-=1
      dW[:,j] += dl * X[i]

    loss -= np.log(np.exp(correct_class_score - maximum) / denom)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
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
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_score = scores[np.arange(num_train),y]
  maximum = np.max(scores,axis=1)
  num = np.exp(correct_class_score - maximum)
  foo = scores - np.tile(maximum,(10,1)).T
  denom = np.sum(np.exp(foo),axis=1)
  probs = num / denom
  loss = np.mean(-np.log(probs))
  loss += 0.5 * reg * np.sum(W * W)

  #dscores = np.tile(probs,(10,1)).T
  dscores = np.exp(foo) / denom[:,None] ####
  dscores[np.arange(num_train),y] -= 1
  dscores /= num_train 
  dW = np.dot(X.T, dscores)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


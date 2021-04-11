from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass
    num_train = X.shape[0] # 500 inputs
    num_classes = W.shape[1] # 10 classes

    for i in range(num_train):
      scores = np.dot(X[i], W)

      # numeric stability: https://cs231n.github.io/linear-classify/#softmax 
      stable_scores = scores - np.max(scores)

      # same as: -stable_scores[y[i]] + np.log(np.sum(np.exp(stable_scores)))
      loss_i = -np.log(np.exp(stable_scores[y[i]]) / np.sum(np.exp(stable_scores)))
      loss += loss_i
     
      # calculate dW
      # if, rPj/rS_sj, else, rPj/rS_syi
      for j in range(num_classes):
        softmax_output = np.exp(stable_scores[j]) / np.sum(np.exp(stable_scores))
        if j == y[i]: # p_j*(1 - p_yi)
          dW[:, j] -= (1 - softmax_output) * X[i]
        else: # -p_j*p_yi
          dW[:, j] -= -softmax_output * X[i]
    
    loss /= num_train
    loss += reg*np.sum(W*W) # L2 Regularization
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pass
    scores = np.dot(X, W)
    stable_scores = scores - np.max(scores, axis=1).reshape(-1,1) # for broadcasting
    loss = np.sum(-np.log(np.exp(stable_scores[range(num_train), y]) / np.sum(np.exp(stable_scores), axis=1)))
    
    softmax_output = np.exp(stable_scores)/np.sum(np.exp(stable_scores), axis = 1) # (500,10)



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

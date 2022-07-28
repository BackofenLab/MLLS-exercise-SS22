'''
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Sebastian Lapuschkin
@contact: sebastian.lapuschkin@hhi.fraunhofer.de, wojciech.samek@hhi.fraunhofer.de
@date: 14.08.2015
@version: 1.2+
@copyright: Copyright (c)  2015-2017, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''
import torch
import copy
import sys
import time
from .module import Module
import numpy
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"):
    import cupy
    import cupy as np
na = np.newaxis
import torch

def cross_entropy_loss(yHat, y_label):
    loss_list = []

    for en,y in enumerate(y_label):



        if y == 1:
            loss_list.append(-np.log(yHat[en]))
        else:
            loss_list.append(-np.log(1 - yHat[en]))
            
    return np.array(loss_list)
      
def sigmoid(z):

    #print(z)
    #print(1.0 / (1.0 + np.exp(-z)))

    return 1.0 / (1.0 + np.exp(-z))

# -------------------------------
# Sequential layer
# -------------------------------
class Sequential(Module):
    '''
    Top level access point and incorporation of the neural network implementation.
    Sequential manages a sequence of computational neural network modules and passes
    along in- and outputs.
    '''

    def __init__(self,modules):
        '''
        Constructor

        Parameters
        ----------
        modules : list, tuple, etc. enumerable.
            an enumerable collection of instances of class Module
        '''
        Module.__init__(self)
        self.modules = modules

	    #make sure to migrate py-modules and nn-modules to the same backend
        if imp.find_spec("cupy"):
            self.to_cupy()
        else:
            self.to_numpy()

    def to_cupy(self):
        global np
        for m in self.modules:
            m.to_cupy()
        np = cupy # ensure correct numerics backend

    def to_numpy(self):
        global np
        for m in self.modules:
            m.to_numpy()
        np = numpy # ensure correct numerics backend

    def drop_softmax_output_layer(self):
        '''
        This function removes the softmax output layer from the model, if there is any.
        '''
        from .softmax import SoftMax
        if isinstance(self.modules[-1],  SoftMax):
            print('removing softmax output mapping')
            del self.modules[-1]
        else:
            print('output layer is not softmax. nothing to do')



    def forward(self,X,lrp_aware=False):
        '''
        Realizes the forward pass of an input through the net

        Parameters
        ----------
        X : numpy.ndarray
            a network input.

        lrp_aware : bool
            controls whether the forward pass is to be computed with awareness for multiple following
            LRP calls. this will sacrifice speed in the forward pass but will save time if multiple LRP
            calls will follow for the current X, e.g. wit different parameter settings or for multiple
            target classes.

        Returns
        -------
        X : numpy.ndarray
            the output of the network's final layer
        '''

        for m in self.modules:

            X = m.forward(X,lrp_aware=lrp_aware)

            
        return X


    def backward(self,DY):
        for m in self.modules[::-1]:
            DY = m.backward(DY)
        return DY


    def update(self,lrate):
        for m in self.modules:
            m.update(lrate)


    def clean(self):
        '''
        Removes temporary variables from all network layers.
        '''
        for m in self.modules:
            m.clean()


    def train(self, X, Y,  Xval = [], Yval = [],  batchsize = 25, iters = 10000, lrate = 0.005, lrate_decay = None, lfactor_initial=1.0 , status = 250, convergence = -1, transform = None, silent=False):
    
    

    
        '''
        Provides a method for training the neural net (self) based on given data.

        Parameters
        ----------

        X : numpy.ndarray
            the training data, formatted to (N,D) shape, with N being the number of samples and D their dimensionality

        Y : numpy.ndarray
            the training labels, formatted to (N,C) shape, with N being the number of samples and C the number of output classes.

        Xval : numpy.ndarray
            some optional validation data. used to measure network performance during training.
            shaped (M,D)

        Yval : numpy.ndarray
            the validation labels. shaped (M,C)

        batchsize : int
            the batch size to use for training

        iters : int
            max number of training iterations

        lrate : float
            the initial learning rate. the learning rate is adjusted during training with increased model performance. See lrate_decay

        lrate_decay : string
            controls if and how the learning rate is adjusted throughout training:
            'none' or None disables learning rate adaption. This is the DEFAULT behaviour.
            'sublinear' adjusts the learning rate to lrate*(1-Accuracy**2) during an evaluation step, often resulting in a better performing model.
            'linear' adjusts the learning rate to lrate*(1-Accuracy) during an evaluation step, often resulting in a better performing model.

        lfactor_initial : float
            specifies an initial discount on the given learning rate, e.g. when retraining an established network in combination with a learning rate decay,
            it might be undesirable to use the given learning rate in the beginning. this could have been done better. TODO: do better.
            Default value is 1.0

        status : int
            number of iterations (i.e. number of rounds of batch forward pass, gradient backward pass, parameter update) of silent training
            until status print and evaluation on validation data.

        convergence : int
            number of consecutive allowed status evaluations with no more model improvements until we accept the model has converged.
            Set <=0 to disable. Disabled by DEFAULT.
            Set to any value > 0 to control the maximal consecutive number (status * convergence) iterations allowed without model improvement, until convergence is accepted.

        transform : function handle
            a function taking as an input a batch of training data sized [N,D] and returning a batch sized [N,D] with added noise or other various data transformations. It's up to you!
            default value is None for no transformation.
            expected syntax is, with X.shape == Xt.shape == (N,D)
            def yourFunction(X):
                Xt = someStuff(X)
                return Xt
        '''

        def randperm(N,b):
            '''
            helper method for picking b unique random indices from a range [0,N[.
            we do not use numpy.random.permutation or numpy.random.choice
            due to known severe performance issues with drawing without replacement.
            if the ratio of N/b is high enough, we should see a huge performance gain.

            N : int
                range of indices [0,N[ to choose from.m, s = divmod(seconds, 60)


            b : the number of unique indices to pick.
            '''
            assert(b <= N) # if this fails no valid solution can be found.
            I = numpy.arange(0)
            while I.size < b:
                I = numpy.unique(numpy.append(I,numpy.random.randint(0,N,[b-I.size,])))
            return np.array(I)

        t_start = time.time()
        untilConvergence = convergence;    learningFactor = lfactor_initial
        bestAccuracy = 0.0;                bestLayers = copy.deepcopy(self.modules)
        bestLoss = np.Inf;                 bestIter = 0

        N = X.shape[0]
        for d in range(iters):


            samples = randperm(N,batchsize)

            #transform batch data (maybe)
            if transform == None:
                batch = X[samples,:]
            else:
                batch = transform(X[samples,:])

            #forward and backward propagation steps with parameter update
            Ypred = self.forward(batch)

            Ypred = np.array(torch.sigmoid(torch.tensor(Ypred)))

            self.backward(Ypred - np.array(Y[samples])) #l1-loss
            self.update(lrate*learningFactor)

            #periodically evaluate network and optionally adjust learning rate or check for convergence.
            if (d+1) % status == 0:
                if not len(Xval) == 0 and not len(Yval) == 0: #if given, evaluate on validation data
                    Ypred = self.forward(Xval)
                    
                    Ypred = sigmoid(Ypred)
                    
                    pred = [1 if x > 0.5 else 0 for x in Ypred]
                    Yval2 = [y[0] for y in Yval]
                    acc = sum([1 if pred[en] == Yval2[en] else 0 for en, _ in enumerate(pred)])/len(Yval2)
                    

                    if not np == numpy: acc = np.asnumpy(acc); l1loss = np.asnumpy(l1loss)
                    if not silent: print('Accuracy after {0} iterations on validation set: {1}'.format(d+1, acc*100))

                else: #evaluate on the training data only
                    Ypred = self.forward(X)

                    acc = np.mean(np.argmax(Ypred, axis=1) == np.argmax(Y, axis=1))

                    l1loss = np.abs(Ypred - Y).sum()/Y.shape[0]
                    if not numpy == np: acc = np.asnumpy(acc); l1loss = np.asnumpy(l1loss)
                    if not silent: print('Accuracy after {0} iterations on training data: {1}'.format(d+1,acc*100))


                #save current network parameters if we have improved
                #if acc >= bestAccuracy and l1loss <= bestLoss:
                # only go by loss

                t_elapsed =  time.time() - t_start
                percent_done = float(d+1)/iters #d+1 because we are after the iteration's heavy lifting
                t_remaining_estimated = t_elapsed/percent_done - t_elapsed

                t_m, t_s = divmod(t_remaining_estimated, 60)
                t_h, t_m = divmod(t_m, 60)
                t_d, t_h = divmod(t_h, 24)

                timestring = '{}d {}h {}m {}s'.format(int(t_d), int(t_h), int(t_m), int(t_s))
                if not silent: print('    Estimate time until current training ends : {} ({:.2f}% done)'.format(timestring, percent_done*100))

            elif (d+1) % (status/10) == 0:
                # print 'alive' signal
                #sys.stdout.write('.')
                #l1loss = np.abs(Ypred - np.array(Y[samples,:])).sum()/Ypred.shape[0]
                loss = cross_entropy_loss(Ypred, Y[samples,:]).sum()/Ypred.shape[0]
                
                
                if not np == numpy: l1loss = np.asnumpy(l1loss)
                if not silent:
                    sys.stdout.write('batch# {}, lrate {}, bce-loss {:.4}\n'.format(d+1,lrate*learningFactor,loss))
                    sys.stdout.flush()



    def set_lrp_parameters(self,lrp_var=None,param=None):
        for m in self.modules:
            m.set_lrp_parameters(lrp_var=lrp_var,param=param)

    def lrp(self,R,lrp_var=None,param=None):
        '''
        Performs LRP by calling subroutines, depending on lrp_var and param or
        preset values specified via Module.set_lrp_parameters(lrp_var,lrp_param)

        If lrp parameters have been pre-specified (per layer), the corresponding decomposition
        will be applied during a call of lrp().

        Specifying lrp parameters explicitly when calling lrp(), e.g. net.lrp(R,lrp_var='alpha',param=2.),
        will override the preset values for the current call.

        How to use:

        net.forward(X) #forward feed some data you wish to explain to populat the net.

        then either:

        net.lrp() #to perform the naive approach to lrp implemented in _simple_lrp for each layer

        or:

        for m in net.modules:
            m.set_lrp_parameters(...)
        net.lrp() #to preset a lrp configuration to each layer in the net

        or:

        net.lrp(somevariantname,someparameter) # to explicitly call the specified parametrization for all layers (where applicable) and override any preset configurations.

        Parameters
        ----------
        R : numpy.ndarray
            final layer relevance values. usually the network's prediction of some data points
            for which the output relevance is to be computed
            dimensionality should be equal to the previously computed predictions

        lrp_var : str
            either 'none' or 'simple' or None for standard Lrp ,
            'epsilon' for an added epsilon slack in the denominator
            'alphabeta' or 'alpha' for weighting positive and negative contributions separately. param specifies alpha with alpha + beta = 1
            'flat' projects an upper layer neuron's relevance uniformly over its receptive field.
            'ww' or 'w^2' only considers the square weights w_ij^2 as qantities to distribute relevances with.

        param : double
            the respective parameter for the lrp method of choice

        Returns
        -------

        R : numpy.ndarray
            the first layer relevances as produced by the neural net wrt to the previously forward
            passed input data. dimensionality is equal to the previously into forward entered input data

        Note
        ----

        Requires the net to be populated with temporary variables, i.e. forward needed to be called with the input
        for which the explanation is to be computed. calling clean in between forward and lrp invalidates the
        temporary data
        '''

        for m in self.modules[::-1]:
            R = m.lrp(R,lrp_var,param)
        return R

# -*- coding: utf-8 -*-
#Layer Buffet~ choose what you like to eat!

'''
For now we have 
    1.HiddenLayer_ReLU
    2.HiddenLayer_PReLU
    3.HiddenLayer_PReLU_DropOut
    4.OutLayer
in buffet
'''

import theano
import theano.tensor as T
import numpy
import numpy as np

def PReLU(x, a):
    return theano.tensor.switch(x<0, a*x, x)

def ReLU(x):
    return theano.tensor.switch(x<0, 0, x)

def dropout(In,rng, p=0.5):     
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=In.shape, dtype=theano.config.floatX)
    return In * T.cast(mask, theano.config.floatX)


class HiddenLayer_PReLU(object):
    def __init__(self,rng,In,NIn,NOut,W_hidden=None,b_hidden=None,a_hidden=None):
        if W_hidden is None :
            W_hidden = theano.shared(
                value=numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6./(NIn+NOut)),high=numpy.sqrt(6./(NIn+NOut)),
                size=(NIn,NOut)),dtype=theano.config.floatX),name='W_hidden',borrow=True)
        if b_hidden is  None :
            b_hidden = theano.shared(
                value=numpy.zeros((NOut,),dtype=theano.config.floatX),
                name='b_hidden',borrow=True)
        if a_hidden is None :
            a_hidden = theano.shared(
                value=numpy.zeros((NOut,),dtype=theano.config.floatX)+0.25,
                name='a_hidden',borrow=True)
        self.W_hidden = W_hidden
        #self.b_hidden = b_hidden
        #self.a_hidden = a_hidden
        Out = T.dot(In,W_hidden)+b_hidden
        self.Out = PReLU(Out,a_hidden)
        self.params = [self.W_hidden,b_hidden,a_hidden]#[self.W_hidden,self.b_hidden,self.a_hidden]

class HiddenLayer_PReLU_DropOut(object):
    def __init__(self,rng,train_test,In,NIn,NOut,W_hidden=None,b_hidden=None,a_hidden=None,p=0.5):
        if W_hidden is None :
            W_hidden = theano.shared(
                value=numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6./(NIn+NOut)),high=numpy.sqrt(6./(NIn+NOut)),
                size=(NIn,NOut)),dtype=theano.config.floatX),name='W_hidden',borrow=True)
        if b_hidden is  None :
            b_hidden = theano.shared(
                value=numpy.zeros((NOut,),dtype=theano.config.floatX),
                name='b_hidden',borrow=True)
        if a_hidden is None :
            a_hidden = theano.shared(
                value=numpy.zeros((NOut,),dtype=theano.config.floatX)+0.25,
                name='a_hidden',borrow=True)
        self.W_hidden = W_hidden
        #self.b_hidden = b_hidden
        #self.a_hidden = a_hidden
        Out = PReLU(T.dot(In,W_hidden)+b_hidden,a_hidden)
        Out_drop = dropout(np.cast[theano.config.floatX](1./p) * Out,p=0.5,rng=rng)
        self.Out = T.switch(T.neq(train_test, 0), Out_drop, Out)
        self.params = [self.W_hidden,b_hidden,a_hidden]#[self.W_hidden,self.b_hidden,self.a_hidden]

class HiddenLayer_PReLU_DropConnect(object):
    def __init__(self,rng,train_test,In,NIn,NOut,W_hidden=None,b_hidden=None,a_hidden=None,p=0.5):
        if W_hidden is None :
            W_hidden = theano.shared(
                value=numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6./(NIn+NOut)),high=numpy.sqrt(6./(NIn+NOut)),
                size=(NIn,NOut)),dtype=theano.config.floatX),name='W_hidden',borrow=True)
        if b_hidden is  None :
            b_hidden = theano.shared(
                value=numpy.zeros((NOut,),dtype=theano.config.floatX),
                name='b_hidden',borrow=True)
        if a_hidden is None :
            a_hidden = theano.shared(
                value=numpy.zeros((NOut,),dtype=theano.config.floatX)+0.25,
                name='a_hidden',borrow=True)
        self.W_hidden = W_hidden
        #self.b_hidden = b_hidden
        #self.a_hidden = a_hidden
        Out = PReLU(T.dot(In,W_hidden)+b_hidden,a_hidden)
        Out_drop = PReLU(T.dot(In, dropout(W_hidden, p=0.5, rng=rng))+b_hidden,a_hidden)
        self.Out = T.switch(T.neq(train_test, 0), Out_drop, Out)
        self.params = [self.W_hidden,b_hidden,a_hidden]#[self.W_hidden,self.b_hidden,self.a_hidden]

class HiddenLayer_ReLU(object):
    def __init__(self,rng,In,NIn,NOut,W_hidden=None,b_hidden=None):
        if W_hidden is None :
            W_hidden = theano.shared(
                value=numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6./(NIn+NOut)),high=numpy.sqrt(6./(NIn+NOut)),
                size=(NIn,NOut)),dtype=theano.config.floatX),name='W_hidden',borrow=True)
        #initial bias with 0.1 (put z to the linear side, not zero side) learn faster
        if b_hidden is  None :
            b_hidden = theano.shared(
                value=numpy.zeros((NOut,),dtype=theano.config.floatX)+0.1,
                name='b_hidden',borrow=True)
        self.W_hidden = W_hidden
        #self.b_hidden = b_hidden
        Out = T.dot(In,W_hidden)+b_hidden
        self.Out = ReLU(Out)
        self.params = [self.W_hidden,b_hidden]#[self.W_hidden,self.b_hidden]

class OutLayer(object):
    def __init__(self,rng,In,NIn,NOut,W_out=None,b_out=None):
        if W_out is None:
            W_out = theano.shared(
                value=numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6./(NIn+NOut)),high=numpy.sqrt(6./(NIn+NOut)),
                size=(NIn,NOut)),dtype=theano.config.floatX),name='W_out',borrow=True)
        if b_out is None:
            b_out = theano.shared(
                value=numpy.zeros((NOut,),dtype=theano.config.floatX),name='b_out',borrow=True)
        self.W_out = W_out
        #self.b_out = b_out
        self.Out = T.dot(In,W_out)+b_out
        self.params = [self.W_out,b_out]#[self.W_out,self.b_out]

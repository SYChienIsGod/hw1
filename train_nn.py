# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:32:02 2015

@author: Jason
"""

import paths

import numpy
import cPickle
import theano

#%% Read pickled data



rng = numpy.random.RandomState(6789)
f = file(paths.pathToSaveMFCCTrain,'rb')
raw_data = cPickle.load(f)
f.close()

raw_scaling = []
raw_means = []
for i in range(raw_data.shape[1]):
    scaling = numpy.sqrt(numpy.var(raw_data[:,i]));
    raw_scaling.append(scaling)
    raw_data[:,i] = raw_data[:,i]/scaling
    meaning = numpy.mean(raw_data[:,i])
    raw_means.append(meaning)
    raw_data[:,i] = raw_data[:,i]-meaning
    
testing_data_sel = rng.uniform(size=(raw_data.shape[0],)) < 0.1
training_data_sel = testing_data_sel==0
training_data = raw_data[training_data_sel,:]
testing_data = raw_data[testing_data_sel,:]

f = file(paths.pathToSave48Labels,'rb')
raw_labels = cPickle.load(f)
f.close()
training_labels = raw_labels[training_data_sel]
testing_labels = raw_labels[testing_data_sel]


training_x_shared = theano.shared(numpy.asarray(training_data,dtype=theano.config.floatX),borrow=True)
training_y_shared = theano.tensor.cast(theano.shared(numpy.asarray(training_labels,dtype=theano.config.floatX),borrow=True),'int32')
testing_x_shared = theano.shared(numpy.asarray(testing_data,dtype=theano.config.floatX),borrow=True)
testing_y_shared = theano.tensor.cast(theano.shared(numpy.asarray(testing_labels,dtype=theano.config.floatX),borrow=True),'int32')

#%% Here we go
NIn = 39
NHidden_1 = 100
NHidden = 100
NOut = 48

L1_weighting = 0.001
L2_weighting = 0.0001

batch_size = 1000

NBatches = int(numpy.floor(training_x_shared.get_value(borrow=True).shape[0]/batch_size))
NTestBatches = int(numpy.floor(testing_x_shared.get_value(borrow=True).shape[0]/batch_size))

x = theano.tensor.matrix('x')
y = theano.tensor.ivector('y')
idx = theano.tensor.lscalar('idx')

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

W_hidden_1 = theano.shared(value=numpy.asarray(rng.uniform(low=0,high=numpy.sqrt(6./(NIn+NHidden_1)),size=(NIn,NHidden_1)),dtype=theano.config.floatX),name='W_hidden_1',borrow=True)
b_hidden_1 = theano.shared(value=numpy.zeros((NHidden_1,),dtype=theano.config.floatX),name='b_hidden_1',borrow=True)
act_hidden_1 = theano.tensor.tanh(theano.tensor.dot(x,W_hidden_1)+b_hidden_1);
W_hidden = theano.shared(value=numpy.asarray(rng.uniform(low=0,high=numpy.sqrt(6./(NHidden_1+NHidden)),size=(NHidden_1,NHidden)),dtype=theano.config.floatX),name='W_hidden',borrow=True)
b_hidden = theano.shared(value=numpy.zeros((NHidden,),dtype=theano.config.floatX),name='b_hidden',borrow=True)
act_hidden_2 = theano.tensor.tanh(theano.tensor.dot(act_hidden_1,W_hidden)+b_hidden);
W_out = theano.shared(value=numpy.zeros((NHidden,NOut),dtype=theano.config.floatX),name='W_out',borrow=True)
b_out = theano.shared(value=numpy.zeros((NOut,),dtype=theano.config.floatX),name='b_out',borrow=True)
softmax = theano.tensor.nnet.softmax(theano.tensor.dot(act_hidden_2,W_out)+b_out)
prediction = theano.tensor.argmax(softmax,axis=1)

def NLL(label):
    return -theano.tensor.mean(theano.tensor.log(softmax)[theano.tensor.arange(label.shape[0]), label])
    
def errors(label):
    return theano.tensor.mean(theano.tensor.eq(prediction,label))


L1_reg = abs(W_hidden).sum()+abs(W_out).sum()
L2_reg = (W_hidden ** 2).sum()+(W_out ** 2).sum()

cost_function = NLL(y) # +L1_reg*L1_weighting+L2_reg*L2_weighting

params = [ W_hidden_1, b_hidden_1, W_hidden, b_hidden, W_out, b_out]
grads = [];
for p in params:
    grads.append(theano.tensor.grad(cost_function,p))
    
Learning_Rate = 0.2

updates = [];
for p,g in zip(params,grads):
    updates.append((p,p-Learning_Rate * g))
    
training_proc = theano.function(inputs=[idx], outputs=cost_function, updates=updates,givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]})

test_on_training_proc = theano.function(inputs=[idx], outputs=errors(y), givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]})

test_on_testing_proc = theano.function(inputs=[idx], outputs=errors(y), givens={x:testing_x_shared[idx*batch_size:(idx+1)*batch_size],y:testing_y_shared[idx*batch_size:(idx+1)*batch_size]})

NEpochs = 200;
iteration = 0;

g_W_hidden = theano.function(inputs=[idx],outputs=grads[0],givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]});
g_W_out = theano.function(inputs=[idx],outputs=grads[0],givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]});

#%%
for epoch in xrange(NEpochs):
    for minibatch_i in xrange(NBatches):
        iteration = iteration+1;
        avg_cost = training_proc(minibatch_i)  
        #gradient_W_hidden = g_W_hidden(minibatch_i);
        #print 'Gradient W_hidden: %f' % numpy.sum(numpy.abs(gradient_W_hidden))
        #gradient_W_out = g_W_out(minibatch_i);
        #print 'Gradient W_out: %f' % numpy.sum(numpy.abs(gradient_W_out))
        if iteration % 200 == 0:
            print 'Epoch %i, Minibatch %i' % (epoch,minibatch_i)
            test_errors = [test_on_testing_proc(i) for i in xrange(NTestBatches)]
            print 'Current Accuracy: %f ' % numpy.mean(test_errors)

#%% Predict and write result

f = file(paths.pathToSaveMFCCTest,'rb')
evaluation_data = cPickle.load(f)
f.close()

for i in range(evaluation_data.shape[1]):
    scaling = raw_scaling[i]
    evaluation_data[:,i] = evaluation_data[:,i]/scaling
    meaning = raw_means[i]
    evaluation_data[:,i] = evaluation_data[:,i]-meaning

f = file(paths.pathToSaveTestIds,'rb')
evaluation_ids = cPickle.load(f)
f.close()

evaluation_x_shared = theano.shared(numpy.asarray(evaluation_data,dtype=theano.config.floatX),borrow=True)

predict_proc = theano.function(inputs=[],outputs=prediction,givens={x:evaluation_x_shared})

evaluation_y_pred = predict_proc()

ph48_39 = numpy.loadtxt(paths.pathToMapPhones,dtype='str_',delimiter='\t')
ph48_39_dict = dict(ph48_39)
phi_48 = dict(zip(numpy.arange(0,48),ph48_39[:,0]))
evaluation_y_pred_str = [ph48_39_dict[phi_48[pred]] for pred in evaluation_y_pred]
import csv
with open('prediction_3.csv','wb') as csvfile:
    csvw = csv.writer(csvfile,delimiter=',')
    csvw.writerow(['Id','Prediction'])
    for id_,pred_ in zip(evaluation_ids,evaluation_y_pred_str):
        csvw.writerow([id_,pred_])
   
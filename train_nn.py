# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:32:02 2015
@author: Jason
"""
"""
Ver. 0.01 by PHHung
Ver. 0.02 by HYTseng
some modification
1.use relu instead of tanh -> use Prelu instead
2.initialize W from -x to + x (I don't think form 0 to +x is a good idea)
3.lower learning rate <= because in previous version, at later epoch(100+) accuracy 
                            is jumping between 0.55 and 0.59 (which means learnig rate too high)
4.smaller batch_size <= learn faster
5.add some comment (something we should discuss)
6.add timer to record computation time
7.display accuracy @every epoch (instead of @every 200 iteration) 
submission result (prediction_4.csv)
@200 epoch validate accuracy is 0.595 (best record 0.600)
submission accuracy is 0.62645 
lower than our best record 0.62898 (FBANK)
I guess that is because in 0.628 we use FBANK
but i use MFCC here 
if we compare with MFCC best record in our team (0.622)
there is still some improvement from 0.622 to 0.626
maybe we should use FBANK instead of MFCC in the future
"""


import paths
import time
import numpy
import cPickle
import theano

#%% Read pickled data

start_time = time.clock()

rng = numpy.random.RandomState(6789)
f = file(paths.pathToSaveFBANKTrain,'rb')
raw_data = cPickle.load(f)
f.close()

raw_scaling = []
raw_means = []

# Data normalize
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


training_x_shared = theano.shared(
	numpy.asarray(training_data,dtype=theano.config.floatX),borrow=True)
training_y_shared = theano.tensor.cast(
	theano.shared(numpy.asarray(training_labels,dtype=theano.config.floatX),borrow=True),'int32')
testing_x_shared = theano.shared(
	numpy.asarray(testing_data,dtype=theano.config.floatX),borrow=True)
testing_y_shared = theano.tensor.cast(
	theano.shared(numpy.asarray(testing_labels,dtype=theano.config.floatX),borrow=True),'int32')

#%% Here we go
NIn = 69 # FBANK:69 MFCC:39
NHidden_1 = 100
NHidden_2 = 100
#NHidden_3 = 100
#NHidden_4 = 100
NOut = 48

L1_weighting = 0.001
L2_weighting = 0.0001

#PHHung : usually use 64 or 128 in practice (I guess~) (To Tune)
batch_size = 512 #1000

NBatches = int(numpy.floor(
	training_x_shared.get_value(borrow=True).shape[0]/batch_size))
NTestBatches = int(numpy.floor(
	testing_x_shared.get_value(borrow=True).shape[0]/batch_size))

x = theano.tensor.matrix('x')
y = theano.tensor.ivector('y')
idx = theano.tensor.lscalar('idx')

#PHHung relu is your good friend~
def Prelu(x, a):
    return theano.tensor.switch(x<0, a*x, x)


#PHHung : uniform distribution from 0 to sqrt(6/NIn+NHidden)?
#         or from -sqrt(6/NIn+NHidden) to sqrt(6/NIn+NHidden)?

#%% Hidden layer 1
W_hidden_1 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NIn+NHidden_1)),high=numpy.sqrt(6./(NIn+NHidden_1)),
		size=(NIn,NHidden_1)),dtype=theano.config.floatX),name='W_hidden_1',borrow=True)
b_hidden_1 = theano.shared(
	value=numpy.zeros((NHidden_1,),dtype=theano.config.floatX),name='b_hidden_1',borrow=True)
a_hidden_1 = theano.shared(
	value=numpy.zeros((NHidden_1,),dtype=theano.config.floatX)+0.25,name='a_hidden_1',borrow=True)
act_hidden_1 = Prelu(theano.tensor.dot(x,W_hidden_1)+b_hidden_1, a_hidden_1)

#%% Hidden layer 2
W_hidden_2 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_1+NHidden_2)),high=numpy.sqrt(6./(NHidden_1+NHidden_2)),
		size=(NHidden_1,NHidden_2)),dtype=theano.config.floatX),name='W_hidden_2',borrow=True)
b_hidden_2 = theano.shared(
	value=numpy.zeros((NHidden_2,),dtype=theano.config.floatX),name='b_hidden_2',borrow=True)
a_hidden_2 = theano.shared(
	value=numpy.zeros((NHidden_1,),dtype=theano.config.floatX)+0.25,name='a_hidden_2',borrow=True)
act_hidden_2 = Prelu(theano.tensor.dot(act_hidden_1,W_hidden_2)+b_hidden_2, a_hidden_2)
'''
#%% Hidden layer 3
W_hidden_3 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_2+NHidden_3)),high=numpy.sqrt(6./(NHidden_2+NHidden_3)),
		size=(NHidden_2,NHidden_3)),dtype=theano.config.floatX),name='W_hidden_3',borrow=True)
b_hidden_3 = theano.shared(
	value=numpy.zeros((NHidden_3,),dtype=theano.config.floatX),name='b_hidden_3',borrow=True)
act_hidden_3 = relu(theano.tensor.dot(act_hidden_2,W_hidden_3)+b_hidden_3)
#%% Hidden layer 4
W_hidden_4 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_3+NHidden_4)),high=numpy.sqrt(6./(NHidden_3+NHidden_4)),
		size=(NHidden_3,NHidden_4)),dtype=theano.config.floatX),name='W_hidden_4',borrow=True)
b_hidden_4 = theano.shared(
	value=numpy.zeros((NHidden_4,),dtype=theano.config.floatX),name='b_hidden_4',borrow=True)
act_hidden_4 = relu(theano.tensor.dot(act_hidden_3,W_hidden_4)+b_hidden_4)
'''
#PHHung : initialize W with all zero?
#W_out = theano.shared(value=numpy.zeros((NHidden,NOut),dtype=theano.config.floatX),name='W_out',borrow=True)
W_out = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_2+NOut)),high=numpy.sqrt(6./(NHidden_2+NOut)),
		size=(NHidden_2,NOut)),dtype=theano.config.floatX),name='W_out',borrow=True)
b_out = theano.shared(
	value=numpy.zeros((NOut,),dtype=theano.config.floatX),name='b_out',borrow=True)

#PHHung : Can we use nnet.softmax? (I guess not... maybe we have to write one)
softmax = theano.tensor.nnet.softmax(theano.tensor.dot(act_hidden_2,W_out)+b_out)
prediction = theano.tensor.argmax(softmax,axis=1)

def NLL(label):
    return -theano.tensor.mean(
    	theano.tensor.log(softmax)[theano.tensor.arange(label.shape[0]), label])
    
def errors(label):
    return theano.tensor.mean(theano.tensor.eq(prediction,label))


#L1_reg = abs(W_hidden).sum()+abs(W_out).sum()
#L2_reg = (W_hidden ** 2).sum()+(W_out ** 2).sum()

cost_function = NLL(y) # +L1_reg*L1_weighting+L2_reg*L2_weighting

#params = [ W_hidden_1, b_hidden_1, W_hidden_2, b_hidden_2, W_hidden_3, b_hidden_3, W_hidden_4, b_hidden_4, W_out, b_out]
params = [ W_hidden_1, b_hidden_1, a_hidden_1, W_hidden_2, b_hidden_2, a_hidden_2, W_out, b_out]
grads = [];
for p in params:
    grads.append(theano.tensor.grad(cost_function,p))

#PHHung : LR = 0.2 may be a litte too big!!     
Learning_Rate = 0.02 #0.2

updates = [];
for p,g in zip(params,grads):
    updates.append((p,p-Learning_Rate * g))
    
training_proc = theano.function(
	inputs=[idx], outputs=cost_function, updates=updates,
	givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]})

test_on_training_proc = theano.function(
	inputs=[idx], outputs=errors(y), 
	givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]})

test_on_testing_proc = theano.function(
	inputs=[idx], outputs=errors(y), 
	givens={x:testing_x_shared[idx*batch_size:(idx+1)*batch_size],y:testing_y_shared[idx*batch_size:(idx+1)*batch_size]})

NEpochs = 2000;
iteration = 0;

g_W_hidden = theano.function(
	inputs=[idx],outputs=grads[0],
	givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]});
g_W_out = theano.function(
	inputs=[idx],outputs=grads[0],
	givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]});

best_accuracy = 0

#%%
for epoch in xrange(NEpochs):
    for minibatch_i in xrange(NBatches):
        iteration = iteration+1;
        avg_cost = training_proc(minibatch_i)  
        #gradient_W_hidden = g_W_hidden(minibatch_i);
        #print 'Gradient W_hidden: %f' % numpy.sum(numpy.abs(gradient_W_hidden))
        #gradient_W_out = g_W_out(minibatch_i);
        #print 'Gradient W_out: %f' % numpy.sum(numpy.abs(gradient_W_out))
        #if iteration % 200 == 0:
        #    print 'Epoch %i, Minibatch %i' % (epoch,minibatch_i)
        #    test_errors = [test_on_testing_proc(i) for i in xrange(NTestBatches)]
        #    print '                         Current Accuracy: %f ' % numpy.mean(test_errors)
    test_errors = [test_on_testing_proc(i) for i in xrange(NTestBatches)]
    print 'Epoch %i Current Accuracy: %f ' % (epoch,numpy.mean(test_errors))
    if best_accuracy < numpy.mean(test_errors):
    	best_accuracy = numpy.mean(test_errors)
    	print('best accuracy %f'%best_accuracy)
#%% Predict and write result

f = file(paths.pathToSaveFBANKTest,'rb')
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

stop_time = time.clock()

print('Total time = %.2fmins'%((stop_time-start_time)/60.))

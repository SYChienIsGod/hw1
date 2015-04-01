# -*- coding: utf-8 -*-

"""
Ver. zero by Jan

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

Ver. 0.03 by PHHung
	1.add momentum_sgd

Ver. 0.03a by Jan
	1. replaced softmax layer by manual computation as requested 
    (no difference to theano's softmax but I left it in place as its probably faster)
    
Ver. 0.03b by Jan
	1. Switched to 39 Phonemes prediction

Ver. 0.03c by PHHung
    deeper and wider network model

Ver. 0.03d by PHHung
    add L2 regression

Ver. 0.03e by HYTseng
    input with FBANK and MFCC, add learning_rate decay

Ver. 0.03g by Jan
    fix for the learning rate decay
    
Ver. 0.03h by Jan
    moved learning rate update to epoch-wise procedure

Ver. 0.04 by HYTseng
	add model blending

Ver. 0.04a by PHHung
	add DropOut (no bug now i guess -.-)
	separate layer define from train_nn (to Layer_Buffet.py)
	it will be more clear when create model
	tune parameter by argv[] (write scrip to train multi-model more easily)

"""

import sys
import paths
import time
import numpy
import cPickle
import theano
import theano.tensor as T
import numpy as np
import Layer_Buffet as LB

"""=======================Parameters to tune==========================="""
if len(sys.argv)==1:
	seed = 6789
else:
	seed = int(sys.argv[1])
print('random seed=%i'%seed)
batch_size = 512 
L1_weighting = 0.001 # not use now
L2_weighting = 0.0001
Learning_Rate = numpy.float32(0.01)
Learning_Rate_Decay = numpy.float32(0.9999)
NEpochs =4000;
rng = numpy.random.RandomState(seed)




"""========================Prepare dataset=============================""" 
start_time = time.clock()
#%% Read pickled data
f = file(paths.pathToSaveFBANKTrain,'rb')
raw_data = cPickle.load(f)
f.close()
f = file(paths.pathToSaveMFCCTrain,'rb')
raw_data_1 = cPickle.load(f)
f.close()

raw_data = numpy.append(raw_data[:], raw_data_1[:], 1)

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

# Save normailze parameters (scaling & means) for predict
f = file('normalize_parameter_scaling.save', 'wb')
cPickle.dump(raw_scaling, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
f = file('normalize_parameter_means.save', 'wb')
cPickle.dump(raw_means, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

testing_data_sel = rng.uniform(size=(raw_data.shape[0],)) < 0.1
training_data_sel = testing_data_sel==0
training_data = raw_data[training_data_sel,:]
testing_data = raw_data[testing_data_sel,:]

f = file(paths.pathToSave39Labels,'rb')
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

x = theano.tensor.matrix('x')
y = theano.tensor.ivector('y')
idx = theano.tensor.lscalar('idx')

NBatches = int(numpy.floor(
	training_x_shared.get_value(borrow=True).shape[0]/batch_size))
NTestBatches = int(numpy.floor(
	testing_x_shared.get_value(borrow=True).shape[0]/batch_size))




"""========================Create model==================================="""
NIn = raw_data.shape[1] # FBANK:69 MFCC:39
NHidden_1 = 128
NHidden_2 = 256
NHidden_3 = 512
NHidden_4 = 256
NHidden_5 = 128
NOut = 39

# training or testing , different behavior in DropOut
train_test = T.iscalar('train_test')

# Model , choose layer from layer buffet! 
Hidden_layer_1 = LB.HiddenLayer_PReLU(rng,x,NIn,NHidden_1)
Hidden_layer_2 = LB.HiddenLayer_PReLU(rng,Hidden_layer_1.Out,NHidden_1,NHidden_2)
Hidden_layer_3 = LB.HiddenLayer_PReLU(rng,Hidden_layer_2.Out,NHidden_2,NHidden_3)
Hidden_layer_4 = LB.HiddenLayer_PReLU_DropOut(rng,train_test,Hidden_layer_3.Out,NHidden_3,NHidden_4)
Hidden_layer_5 = LB.HiddenLayer_PReLU_DropOut(rng,train_test,Hidden_layer_4.Out,NHidden_4,NHidden_5)
Out_layer = LB.OutLayer(rng,Hidden_layer_5.Out,NHidden_5,NOut)

softmax = theano.tensor.nnet.softmax(Out_layer.Out)
prediction = theano.tensor.argmax(softmax,axis=1)
#softmax_lin_act = theano.tensor.dot(act_hidden_3,W_out)+b_out # The linear activation just before softmax
#softmax_exp_act = theano.tensor.exp(softmax_lin_act-theano.tensor.max(softmax_lin_act,axis=1,keepdims=True)) 
''' Exponentiation. The subtraction is a numerical trick. 
    For each data point we subtract the maximum activation which is equal to 
    dividing both the numerator and the denominator of the softmax fraction. 
    Keepdims means that the result of the max function still has 39 (equal) entries
    instead of a single entry. (Jan)
'''
#softmax_manual = softmax_exp_act/theano.tensor.sum(softmax_exp_act,axis=1,keepdims=True) # The actual softmax fraction
L2_reg = (Hidden_layer_1.W_hidden**2).sum()+(Hidden_layer_2.W_hidden**2).sum()+(Hidden_layer_3.W_hidden**2).sum()+(Hidden_layer_4.W_hidden**2).sum()+(Hidden_layer_5.W_hidden**2).sum()+(Out_layer.W_out ** 2).sum()
params =  Hidden_layer_1.params + Hidden_layer_2.params + Hidden_layer_3.params + Hidden_layer_4.params + Hidden_layer_5.params + Out_layer.params 





"""===========================Train & validate============================"""
def NLL(label):
    return -T.mean(
    	T.log(softmax)[T.arange(label.shape[0]), label])
    
def errors(label):
    return T.mean(theano.tensor.eq(prediction,label))

cost_function = NLL(y) + L2_reg*L2_weighting # +L1_reg*L1_weighting+L2_reg*L2_weighting

# Create a shared variable for the learning rate
learning_rate_theano = theano.shared(Learning_Rate, name='learning_rate')

#momentum gd 
def momentum_sgd(cost, params, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate_theano*param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*theano.tensor.grad(cost, param)))
    return updates
    
update_proc = momentum_sgd(cost_function,params,0.8)

# The learning rate update is moved into its own function to allow epoch-wise updates
learning_rate_update = theano.function(inputs=[],outputs=learning_rate_theano,updates=[(learning_rate_theano,learning_rate_theano*Learning_Rate_Decay)])

#train
training_proc = theano.function(
	inputs=[idx], outputs=cost_function, updates=update_proc,
	givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],
			y:training_y_shared[idx*batch_size:(idx+1)*batch_size],
			train_test: np.cast['int32'](1)}) 
#validate
test_on_training_proc = theano.function(
	inputs=[idx], outputs=errors(y), 
	givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],
			y:training_y_shared[idx*batch_size:(idx+1)*batch_size],
			train_test: np.cast['int32'](0)}) 
#test
test_on_testing_proc = theano.function(
	inputs=[idx], outputs=errors(y), 
	givens={x:testing_x_shared[idx*batch_size:(idx+1)*batch_size],
			y:testing_y_shared[idx*batch_size:(idx+1)*batch_size],
			train_test: np.cast['int32'](0)}) 

best_accuracy = 0
best_accuracy_epoch =0 

#%%
for epoch in xrange(NEpochs):
    if epoch == 0:
        print('%d dimensions of input feature'%NIn)
    for minibatch_i in xrange(NBatches):
        avg_cost = training_proc(minibatch_i)  

    test_errors = [test_on_testing_proc(i) for i in xrange(NTestBatches)]    
    current_accuracy = numpy.mean(test_errors)
    current_learning_rate = learning_rate_update()
    print 'Epoch %i, Current Accuracy: %f, Avg. cost: %f, Learning rate: %f' % (epoch,current_accuracy,avg_cost,current_learning_rate)

    if best_accuracy < current_accuracy :
    	best_accuracy = current_accuracy
    	best_accuracy_epoch = epoch
    	print('best accuracy %f'%best_accuracy)
    	#save best model
    	f = file('model_best.save', 'wb')
    	cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    	f.close()  		
    if (epoch % 10 == 0) :
        stop_time = time.clock()
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Current best accuracy=%f @epoch%i Total time=%.2fmins'%(best_accuracy,best_accuracy_epoch,(stop_time-start_time)/60.))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

stop_time = time.clock()
print('=========Finish!!  Best accuracy=%f @epoch%i Total time=%.2fmins========'%
	(best_accuracy,best_accuracy_epoch,(stop_time-start_time)/60.))




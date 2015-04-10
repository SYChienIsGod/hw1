# -*- coding: utf-8 -*-
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
 
Ver. 0.04b by Jan
    training data selection got its own random number generator
    
Ver. 0.04c by Jan
    training data is permuted after loading
    
Ver. 0.04d by Jan
    added bagging function to draw random subsets from the trainig data
    
Ver. 0.04e by HY
    combine three frames in one

Ver. 0.04f by PHHung (Bug)
    combine 7 frames in one
    
Ver. 0.04g by Jan
    Make sure that a sentence is either in the validation or in the training set
    
Ver. 0.04h by Jan
    Bootstrap sample aggregating now fills the bootstrap samples up by oversampling

Ver. 0.04i by Jan
    Disabled bootstrap oversampling by default

Ver. 0.04j by Jan
    Set temporary data to None to save memory
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

def bootstrap_sample_aggregating (rng, fraction, training_data, training_labels, oversampling = False):
    if fraction <= 0.5:
        fraction = 0.51
    elif fraction > 1:
        fraction = 1
    bag_selection = rng.uniform(size=(training_data.shape[0],)) < fraction
    indices = numpy.where(bag_selection)[0]
    if not oversampling:
        return training_data[indices,:], training_labels[indices]
    oversampling = rng.uniform(size=(indices.shape[0],)) < 1/fraction-1
    indices = numpy.append(indices[oversampling], indices)
    return training_data[indices,:], training_labels[indices]

"""=======================Parameters to tune==========================="""
if len(sys.argv)==1:
    seed = 6789
    batch_size = 512
else:
    seed = int(sys.argv[1])
    batch_size = int(sys.argv[2])
print('random seed=%i batch_size=%i'%(seed,batch_size))
L1_weighting = 0.001 # not use now
L2_weighting = 0.0001
Learning_Rate = numpy.float32(0.12)
Learning_Rate_Decay = numpy.float32(0.9995)
NEpochs =1000;
rng = numpy.random.RandomState(seed)


"""========================Prepare dataset=============================""" 
start_time = time.clock()
#%% Read pickled data
f = file(paths.pathToSaveFBANKTrain,'rb')
raw_data = cPickle.load(f)
f.close()
#f = file(paths.pathToSaveMFCCTrain,'rb')
#raw_data_1 = cPickle.load(f)
#f.close()

#FBANK*(3+1+3)
raw_data_temp_0 = numpy.append(raw_data[3:-3,:], raw_data[2:-4,:], 1)
raw_data_temp_1 = numpy.append(raw_data_temp_0[:,:], raw_data[1:-5,:], 1)
raw_data_temp_0 = None
raw_data_temp_2 = numpy.append(raw_data_temp_1[:,:], raw_data[:-6,:], 1)
raw_data_temp_1 = None
raw_data_temp_3 = numpy.append(raw_data_temp_2[:,:], raw_data[4:-2,:], 1)
raw_data_temp_2 = None
raw_data_temp_4 = numpy.append(raw_data_temp_3[:,:], raw_data[5:-1,:], 1)
raw_data_temp_3 = None
raw_data = numpy.append(raw_data_temp_4[:,:], raw_data[6:,:], 1)
raw_data_temp_4 = None

f = file(paths.pathToSave39Labels,'rb')
raw_labels = cPickle.load(f)
raw_labels = raw_labels[3:-3] #WHY? raw_labels = raw_labels[1:-1]
assert(raw_labels.shape[0] == raw_data.shape[0])
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

# Save normailze parameters (scaling & means) for predict
f = file('normalize_parameter_scaling.save', 'wb')
cPickle.dump(raw_scaling, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
f = file('normalize_parameter_means.save', 'wb')
cPickle.dump(raw_means, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

# Make sure that the random number generator for the training data selection is always identical
# training_data_sel_rng = numpy.random.RandomState(12345)
# testing_data_sel = training_data_sel_rng.uniform(size=(raw_data.shape[0],)) < 0.1
# training_data_sel = testing_data_sel==0

#%% Training / Test Data Split
trainingIds = numpy.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))
trainingSentences_all = [trainingId.split('_')[1] for trainingId in trainingIds ]
trainingSentences = set(trainingSentences_all)

trainingSentenceDist = dict(zip(trainingSentences,numpy.zeros((len(trainingSentences),),dtype='int')))

lastFrameNo = 100
for id_ in trainingIds:
    data = id_.split('_')
    frameNo = int(data[2])
    if frameNo > lastFrameNo:
        lastFrameNo = frameNo
        continue
    trainingSentenceDist[data[1]] +=1
    lastFrameNo = frameNo

# Selection Process

npTrainSent = numpy.asarray(list(trainingSentences))
training_data_sel_rng = numpy.random.RandomState(123456) 
validationSelection = training_data_sel_rng.uniform(size=(npTrainSent.shape[0],)) < 0.15
validationSent = npTrainSent[validationSelection]
trainingSent = npTrainSent[validationSelection==0]
print "Validation Sentences X Training Sentences: %i" % len(set(validationSent).intersection(set(trainingSent)))

NTrainingSent = numpy.sum([trainingSentenceDist[tS] for tS in trainingSent])
NValidationSent = numpy.sum([trainingSentenceDist[vS] for vS in validationSent])
print "#Validation Sentences: %i, #Training Sentences: %i" % (NValidationSent,NTrainingSent)

trainingIdx = numpy.in1d(numpy.asarray(trainingSentences_all), trainingSent)
validationIdx = numpy.in1d(numpy.asarray(trainingSentences_all), validationSent)
print "#Frames in Training Set: %i, in Validation Set: %i" % (numpy.sum(trainingIdx), numpy.sum(validationIdx))
training_data_sel=trainingIdx[3:-3]
testing_data_sel=validationIdx[3:-3]
training_data = raw_data[training_data_sel,:]
testing_data = raw_data[testing_data_sel,:]

training_labels = raw_labels[training_data_sel]
testing_labels = raw_labels[testing_data_sel]

# Permute the data (reorder it to shuffle the mini batch content)
random_permutation = rng.permutation(training_labels.shape[0])
training_data = training_data[random_permutation,:]
training_labels = training_labels[random_permutation]

# Obtain a bootstrap sample
bag_data, bag_labels = bootstrap_sample_aggregating(rng,0.67,training_data,training_labels, oversampling=False)
# Enable the following lines to train on bagged data
training_data = bag_data
training_labels = bag_labels
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
#%%
NIn = raw_data.shape[1] # FBANK:69 MFCC:39
NHidden_1 = 512
NHidden_2 = 256
NHidden_3 = 256
NHidden_4 = 128
NHidden_5 = 128
NHidden_6 = 128
NHidden_7 = 64
NOut = 39

# training or testing , different behavior in DropOut
train_test = T.iscalar('train_test')

# Model , choose layer from layer buffet! 
Hidden_layer_1 = LB.HiddenLayer_PReLU_DropOut(rng,train_test,x,NIn,NHidden_1)
Hidden_layer_2 = LB.HiddenLayer_PReLU(rng,Hidden_layer_1.Out,NHidden_1,NHidden_2)
Hidden_layer_3 = LB.HiddenLayer_PReLU(rng,Hidden_layer_2.Out,NHidden_2,NHidden_3)
Hidden_layer_4 = LB.HiddenLayer_PReLU(rng,Hidden_layer_3.Out,NHidden_3,NHidden_4)
Hidden_layer_5 = LB.HiddenLayer_PReLU(rng,Hidden_layer_4.Out,NHidden_4,NHidden_5)
Hidden_layer_6 = LB.HiddenLayer_PReLU(rng,Hidden_layer_5.Out,NHidden_5,NHidden_6)
Hidden_layer_7 = LB.HiddenLayer_PReLU(rng,Hidden_layer_6.Out,NHidden_6,NHidden_7)
Out_layer = LB.OutLayer(rng,Hidden_layer_7.Out,NHidden_7,NOut)

#softmax = theano.tensor.nnet.softmax(Out_layer.Out)
softmax_lin_act = Out_layer.Out # The linear activation just before softmax
softmax_exp_act = theano.tensor.exp(softmax_lin_act-theano.tensor.max(softmax_lin_act,axis=1,keepdims=True)) 
''' Exponentiation. The subtraction is a numerical trick. 
    For each data point we subtract the maximum activation which is equal to 
    dividing both the numerator and the denominator of the softmax fraction. 
    Keepdims means that the result of the max function still has 39 (equal) entries
    instead of a single entry. (Jan)
'''
softmax_manual = softmax_exp_act/theano.tensor.sum(softmax_exp_act,axis=1,keepdims=True) # The actual softmax fraction
softmax = softmax_manual
prediction = theano.tensor.argmax(softmax,axis=1)
L2_reg = (Hidden_layer_1.W_hidden**2).sum()+(Hidden_layer_2.W_hidden**2).sum()+(Hidden_layer_3.W_hidden**2).sum()+(Hidden_layer_4.W_hidden**2).sum()+(Hidden_layer_5.W_hidden**2).sum()+(Hidden_layer_6.W_hidden**2).sum()+(Hidden_layer_7.W_hidden**2).sum()+(Out_layer.W_out ** 2).sum()
params =  Hidden_layer_1.params + Hidden_layer_2.params + Hidden_layer_3.params + Hidden_layer_4.params + Hidden_layer_5.params + Hidden_layer_6.params + Hidden_layer_7.params + Out_layer.params 

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
    	f = file('model_best_%i.save'%seed, 'wb')
    	cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    	f.close()  		
    if (epoch % 10 == 0) :
        stop_time = time.clock()
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Current best epoch=%i accuracy=%f @epoch%i Total time=%.2fmins'%(epoch,best_accuracy,best_accuracy_epoch,(stop_time-start_time)/60.))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

stop_time = time.clock()
print('=========Finish!!  Best accuracy=%f @epoch%i Total time=%.2fmins========'%
	(best_accuracy,best_accuracy_epoch,(stop_time-start_time)/60.))

'''
For blending
model
1.in/128/256/512/256/128/out
2.FBANK+MFCC

different between model 1~8
1.random seed from 1111 to 8888 (different initial parameter)
2.different batch size 1~5:256 6,7:1000 8:512
3.accuracy 1~5:~0.686 6,7:~0.692 8:~0.694
'''

import paths
import time
import numpy
import cPickle
import theano
import theano.tensor as T
import numpy as np

#%% Read pickled data

start_time = time.clock()

rng = numpy.random.RandomState(1111) #2222 ~ 8888
srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))

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

#%% Here we go
NIn = raw_data.shape[1] # FBANK:69 MFCC:39
NHidden_1 = 128
NHidden_2 = 256
NHidden_3 = 512
NHidden_4 = 256
NHidden_5 = 128

NOut = 39

L1_weighting = 0.001
L2_weighting = 0.0001

#PHHung : usually use 64 or 128 in practice (I guess~) (To Tune)
batch_size = 256 #1000

NBatches = int(numpy.floor(
	training_x_shared.get_value(borrow=True).shape[0]/batch_size))
NTestBatches = int(numpy.floor(
	testing_x_shared.get_value(borrow=True).shape[0]/batch_size))

x = theano.tensor.matrix('x')
y = theano.tensor.ivector('y')
idx = theano.tensor.lscalar('idx')

def Prelu(x, a):
    return theano.tensor.switch(x<0, a*x, x)

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

#def drop(input, p=0.5, rng=rng):     
#    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
#    return input * mask

#PHHung : uniform distribution from 0 to sqrt(6/NIn+NHidden)?
#         or from -sqrt(6/NIn+NHidden) to sqrt(6/NIn+NHidden)? --> ReLU doesn't have a gradient once the input is negative... so I thought that may be useful (Jan)
#PHHung : z=w*x+b , the input of activate function is z , but here we are initializing w , I thought that is different thing (w & z)

#To Drop or not to Drop, that is the question XD
#dropout=T.iscalar('dropout')
#p=0.5

# Hidden layer 1
W_hidden_1 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NIn+NHidden_1)),high=numpy.sqrt(6./(NIn+NHidden_1)),
		size=(NIn,NHidden_1)),dtype=theano.config.floatX),name='W_hidden_1',borrow=True)
b_hidden_1 = theano.shared(
	value=numpy.zeros((NHidden_1,),dtype=theano.config.floatX),name='b_hidden_1',borrow=True)
a_hidden_1 = theano.shared(
	value=numpy.zeros((NHidden_1,),dtype=theano.config.floatX)+0.25,name='a_hidden_1',borrow=True)
act_hidden_1 = Prelu(theano.tensor.dot(x,W_hidden_1)+b_hidden_1, a_hidden_1)
#drop_output_1 = drop(numpy.cast[theano.config.floatX](1./p) * act_hidden_1)
#output_1 = T.switch(T.neq(dropout, 0), drop_output_1, act_hidden_1)

# Hidden layer 2
W_hidden_2 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_1+NHidden_2)),high=numpy.sqrt(6./(NHidden_1+NHidden_2)),
		size=(NHidden_1,NHidden_2)),dtype=theano.config.floatX),name='W_hidden_2',borrow=True)
b_hidden_2 = theano.shared(
	value=numpy.zeros((NHidden_2,),dtype=theano.config.floatX),name='b_hidden_2',borrow=True)
a_hidden_2 = theano.shared(
	value=numpy.zeros((NHidden_2,),dtype=theano.config.floatX)+0.25,name='a_hidden_2',borrow=True)
act_hidden_2 = Prelu(theano.tensor.dot(act_hidden_1,W_hidden_2)+b_hidden_2, a_hidden_2)
#drop_output_2 = drop(numpy.cast[theano.config.floatX](1./p) * act_hidden_2)
#output_2 = T.switch(T.neq(dropout, 0), drop_output_2, act_hidden_2)

# Hidden layer 3
W_hidden_3 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_2+NHidden_3)),high=numpy.sqrt(6./(NHidden_2+NHidden_3)),
		size=(NHidden_2,NHidden_3)),dtype=theano.config.floatX),name='W_hidden_3',borrow=True)
b_hidden_3 = theano.shared(
	value=numpy.zeros((NHidden_3,),dtype=theano.config.floatX),name='b_hidden_3',borrow=True)
a_hidden_3 = theano.shared(
        value=numpy.zeros((NHidden_3,),dtype=theano.config.floatX)+0.25,name='a_hidden_2',borrow=True)
act_hidden_3 = Prelu(theano.tensor.dot(act_hidden_2,W_hidden_3)+b_hidden_3, a_hidden_3)
#drop_output_3 = drop(numpy.cast[theano.config.floatX](1./p) * act_hidden_3)
#output_3 = T.switch(T.neq(dropout, 0), drop_output_3, act_hidden_3)

# Hidden layer 4
W_hidden_4 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_3+NHidden_4)),high=numpy.sqrt(6./(NHidden_3+NHidden_4)),
		size=(NHidden_3,NHidden_4)),dtype=theano.config.floatX),name='W_hidden_4',borrow=True)
b_hidden_4 = theano.shared(
	value=numpy.zeros((NHidden_4,),dtype=theano.config.floatX),name='b_hidden_4',borrow=True)
a_hidden_4 = theano.shared(
        value=numpy.zeros((NHidden_4,),dtype=theano.config.floatX)+0.25,name='a_hidden_4',borrow=True)
act_hidden_4 = Prelu(theano.tensor.dot(act_hidden_3,W_hidden_4)+b_hidden_4,a_hidden_4)
#drop_output_4= drop(numpy.cast[theano.config.floatX](1./p) * act_hidden_4)
#output_4 = T.switch(T.neq(dropout, 0), drop_output_4, act_hidden_4)

# Hidden layer 5
W_hidden_5 = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_4+NHidden_5)),high=numpy.sqrt(6./(NHidden_4+NHidden_5)),
		size=(NHidden_4,NHidden_5)),dtype=theano.config.floatX),name='W_hidden_5',borrow=True)
b_hidden_5 = theano.shared(
	value=numpy.zeros((NHidden_5,),dtype=theano.config.floatX),name='b_hidden_5',borrow=True)
a_hidden_5 = theano.shared(
        value=numpy.zeros((NHidden_5,),dtype=theano.config.floatX)+0.25,name='a_hidden_5',borrow=True)
act_hidden_5 = Prelu(theano.tensor.dot(act_hidden_4,W_hidden_5)+b_hidden_5,a_hidden_5)
#drop_output_5 = drop(numpy.cast[theano.config.floatX](1./p) * act_hidden_5)
#output_5 = T.switch(T.neq(dropout, 0), drop_output_5, act_hidden_5)

W_out = theano.shared(
	value=numpy.asarray(rng.uniform(
		low=-numpy.sqrt(6./(NHidden_5+NOut)),high=numpy.sqrt(6./(NHidden_5+NOut)),
		size=(NHidden_5,NOut)),dtype=theano.config.floatX),name='W_out',borrow=True)
b_out = theano.shared(
	value=numpy.zeros((NOut,),dtype=theano.config.floatX),name='b_out',borrow=True)

#PHHung : Can we use nnet.softmax? (I guess not... maybe we have to write one) --> replaced (Jan)
#softmax_lin_act = theano.tensor.dot(act_hidden_7,W_out)+b_out # The linear activation just before softmax
#softmax_exp_act = theano.tensor.exp(softmax_lin_act-theano.tensor.max(softmax_lin_act,axis=1,keepdims=True)) 
''' Exponentiation. The subtraction is a numerical trick. 
    For each data point we subtract the maximum activation which is equal to 
    dividing both the numerator and the denominator of the softmax fraction. 
    Keepdims means that the result of the max function still has 39 (equal) entries
    instead of a single entry. (Jan)
'''
#softmax_manual = softmax_exp_act/theano.tensor.sum(softmax_exp_act,axis=1,keepdims=True) # The actual softmax fraction
softmax = theano.tensor.nnet.softmax(theano.tensor.dot(act_hidden_5,W_out)+b_out)
prediction = theano.tensor.argmax(softmax,axis=1)

def NLL(label):
    return -theano.tensor.mean(
    	theano.tensor.log(softmax)[theano.tensor.arange(label.shape[0]), label])
    
def errors(label):
    return theano.tensor.mean(theano.tensor.eq(prediction,label))


#L1_reg = abs(W_hidden).sum()+abs(W_out).sum()
L2_reg = (W_hidden_1 ** 2).sum()+(W_hidden_2 ** 2).sum()+(W_hidden_3 ** 2).sum()+(W_hidden_4 ** 2).sum()+(W_hidden_5 ** 2).sum()+(W_out ** 2).sum()

cost_function = NLL(y) + L2_reg*L2_weighting # +L1_reg*L1_weighting+L2_reg*L2_weighting

params = [ W_hidden_1, b_hidden_1, a_hidden_1, W_hidden_2, b_hidden_2, a_hidden_2, W_hidden_3, b_hidden_3,a_hidden_3, 
          W_hidden_4, b_hidden_4, a_hidden_4, W_hidden_5, b_hidden_5, a_hidden_5, W_out, b_out]

#params = [ W_hidden_1, b_hidden_1, a_hidden_1, W_hidden_2, b_hidden_2, a_hidden_2, W_out, b_out]
#grads = []
#for p in params:
#    grads.append(theano.tensor.grad(cost_function,p))
  
Learning_Rate = numpy.float32(0.01)
Learning_Rate_Decay = numpy.float32(0.9999)

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
#update_proc.append((learning_rate_theano, learning_rate_theano * Learning_Rate_Decay))  

#train
training_proc = theano.function(
	inputs=[idx], outputs=cost_function, updates=update_proc,
	givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]})#,dropout: np.cast['int32'](1)})
#validate
test_on_training_proc = theano.function(
	inputs=[idx], outputs=errors(y), 
	givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size],y:training_y_shared[idx*batch_size:(idx+1)*batch_size]})#,dropout: np.cast['int32'](0)})
#test
test_on_testing_proc = theano.function(
	inputs=[idx], outputs=errors(y), 
	givens={x:testing_x_shared[idx*batch_size:(idx+1)*batch_size],y:testing_y_shared[idx*batch_size:(idx+1)*batch_size]})#,dropout: np.cast['int32'](0)})

#softmax_theano_test = theano.function(inputs=[idx], outputs=softmax,givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size]})
#softmax_manual_test = theano.function(inputs=[idx], outputs=softmax_manual,givens={x:training_x_shared[idx*batch_size:(idx+1)*batch_size]})

NEpochs =500;
iteration = 0;

best_accuracy = 0
best_accuracy_epoch =0 

#%%
for epoch in xrange(NEpochs):
    if epoch == 0:
        print('%d dimensions of input feature'%NIn)
    for minibatch_i in xrange(NBatches):
        iteration = iteration+1;
        avg_cost = training_proc(minibatch_i)  
        
        # Test that the softmax computations are correct: (Checked and difference is 0, Jan)
        '''
        softmax_theano_result = softmax_theano_test(minibatch_i)
        softmax_manual_result = softmax_manual_test(minibatch_i)
        print 'Softmax maximum absolute difference: %f' % numpy.max(numpy.abs(softmax_theano_result-softmax_manual_result))
        '''

    test_errors = [test_on_testing_proc(i) for i in xrange(NTestBatches)]    
    current_accuracy = numpy.mean(test_errors)
    #print 'Epoch %i Current Accuracy: %f , learning rate : %f' % (epoch,current_accuracy,learning_rate_theano.get_value())

    if best_accuracy < current_accuracy :
    	best_accuracy = current_accuracy
    	best_accuracy_epoch = epoch
    	print('best accuracy %f'%best_accuracy)
    	#save best model
    	f = file('model_best_1.save', 'wb')
    	cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    	f.close()  		

    if (epoch % 50 == 0) :
        stop_time = time.clock()

        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Current best accuracy=%f @epoch%i Total time=%.2fmins'%(best_accuracy,best_accuracy_epoch,(stop_time-start_time)/60.))
        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

stop_time = time.clock()
print('=========Finish!!  Best accuracy=%f @epoch%i Total time=%.2fmins========'%
	(best_accuracy,best_accuracy_epoch,(stop_time-start_time)/60.))



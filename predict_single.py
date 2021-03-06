#
# After training, use predict_blending.py to predict result with uniform blending
# the model can be any size and depth, but the order of params must follow[w1, b1, a1, .... wout, bout]
import sys
import paths
import time
import numpy
import numpy as np
import cPickle
import theano
import theano.tensor as T

#========================= load validate and test data ===================================
f = file(paths.pathToSaveFBANKTrain,'rb')
validation_data = cPickle.load(f)
f.close()
#f = file(paths.pathToSaveMFCCTrain,'rb')
#validation_data_1 = cPickle.load(f)
#f.close()
f = file(paths.pathToSave39Labels,'rb')
validation_labels = cPickle.load(f)
f.close()

f = file(paths.pathToSaveFBANKTest,'rb')
evaluation_data = cPickle.load(f)
f.close()
#f = file(paths.pathToSaveMFCCTest,'rb')
#evaluation_data_1 = cPickle.load(f)
#f.close()
f = file(paths.pathToSaveTestIds,'rb')
evaluation_ids = cPickle.load(f)
f.close()


#for FBANK*(3+1+3)
validation_data_1 = np.append(validation_data[3:,:], np.zeros((3, validation_data.shape[1])),0)
validation_data_2 = np.append(validation_data[2:,:], np.zeros((2, validation_data.shape[1])),0)
validation_data_3 = np.append(validation_data[1:,:], np.zeros((1, validation_data.shape[1])),0)
validation_data_4 = validation_data
validation_data_5 = np.append(np.zeros((1, validation_data.shape[1])), validation_data[:-1,:],0)
validation_data_6 = np.append(np.zeros((2, validation_data.shape[1])), validation_data[:-2,:],0)
validation_data_7 = np.append(np.zeros((3, validation_data.shape[1])), validation_data[:-3,:],0)
validation_data = np.append(validation_data_4, validation_data_5,1)
validation_data = np.append(validation_data, validation_data_6,1)
validation_data = np.append(validation_data, validation_data_7,1)
validation_data = np.append(validation_data, validation_data_3,1)
validation_data = np.append(validation_data, validation_data_2,1)
validation_data = np.append(validation_data, validation_data_1,1)

evaluation_data_1 = np.append(evaluation_data[3:,:], np.zeros((3, evaluation_data.shape[1])),0)
evaluation_data_2 = np.append(evaluation_data[2:,:], np.zeros((2, evaluation_data.shape[1])),0)
evaluation_data_3 = np.append(evaluation_data[1:,:], np.zeros((1, evaluation_data.shape[1])),0)
evaluation_data_4 = evaluation_data
evaluation_data_5 = np.append(np.zeros((1, evaluation_data.shape[1])), evaluation_data[:-1,:],0)
evaluation_data_6 = np.append(np.zeros((2, evaluation_data.shape[1])), evaluation_data[:-2,:],0)
evaluation_data_7 = np.append(np.zeros((3, evaluation_data.shape[1])), evaluation_data[:-3,:],0)
evaluation_data = np.append(evaluation_data_4, evaluation_data_5,1)
evaluation_data = np.append(evaluation_data, evaluation_data_6,1)
evaluation_data = np.append(evaluation_data, evaluation_data_7,1)
evaluation_data = np.append(evaluation_data, evaluation_data_3,1)
evaluation_data = np.append(evaluation_data, evaluation_data_2,1)
evaluation_data = np.append(evaluation_data, evaluation_data_1,1)
#validation_data_0 = numpy.append(validation_data[3:-3,:], validation_data[2:-4,:], 1)
#validation_data_1 = numpy.append(validation_data_0[:,:], validation_data[1:-5,:], 1)
#validation_data_2 = numpy.append(validation_data_1[:,:], validation_data[:-6,:], 1)
#validation_data_3 = numpy.append(validation_data_2[:,:], validation_data[4:-2,:], 1)
#validation_data_4 = numpy.append(validation_data_3[:,:], validation_data[5:-1,:], 1)
#validation_data = numpy.append(validation_data_4[:,:], validation_data[6:,:], 1)

#evaluation_data_0 = numpy.append(evaluation_data[3:-3,:], evaluation_data[2:-4,:], 1)
#evaluation_data_1 = numpy.append(evaluation_data_0[:,:], evaluation_data[1:-5,:], 1)
#evaluation_data_2 = numpy.append(evaluation_data_1[:,:], evaluation_data[:-6,:], 1)
#evaluation_data_3 = numpy.append(evaluation_data_2[:,:], evaluation_data[4:-2,:], 1)
#evaluation_data_4 = numpy.append(evaluation_data_3[:,:], evaluation_data[5:-1,:], 1)
#evaluation_data = numpy.append(evaluation_data_4[:,:], evaluation_data[6:,:], 1)

'''#for FBANK*(1+1+1) 
validation_data_1 = np.append(validation_data[1:,:], np.zeros((1, validation_data.shape[1])),0)
validation_data_2 = validation_data
validation_data_3 = np.append(np.zeros((1,validation_data.shape[1])), validation_data[:-1,:],0)
validation_data = np.append(validation_data_2, validation_data_3, 1)
validation_data = np.append(validation_data, validation_data_1, 1)
evaluation_data_1 = np.append(evaluation_data[1:,:], np.zeros((1, evaluation_data.shape[1])),0)
evaluation_data_2 = evaluation_data
evaluation_data_3 = np.append(np.zeros((1,evaluation_data.shape[1])), evaluation_data[:-1,:],0)
evaluation_data = np.append(evaluation_data_2, evaluation_data_3, 1)
evaluation_data = np.append(evaluation_data, evaluation_data_1, 1)
'''

'''#for FBANK+MFCC
validation_data = np.append(validation_data[:], validation_data_1[:], 1)
evaluation_data = np.append(evaluation_data[:], evaluation_data_1[:], 1)
'''
print(evaluation_data.shape)
#========================= load models ====================================================
if len(sys.argv)==1:
    model_path = 'model_best.save'
else:
    model_path = sys.argv[1]
f = file(model_path, 'rb')  # modify your model path here~
params1 = cPickle.load(f)
f.close()

#========================= load scale and mean ============================================
f = file('normalize_parameter_scaling.save', 'rb')  # modify your model path here~
raw_scaling = cPickle.load(f)
f.close()
f = file('normalize_parameter_means.save', 'rb')  # modify your model path here~
raw_means = cPickle.load(f)
f.close()

#Normalize test data by training data's mean and scale
assert(validation_data.shape[1] == evaluation_data.shape[1]) 
for i in range(validation_data.shape[1]):
    scaling = raw_scaling[i]
    validation_data[:,i] = validation_data[:,i]/scaling
    evaluation_data[:,i] = evaluation_data[:,i]/scaling
    meaning = raw_means[i]
    validation_data[:,i] = validation_data[:,i]-meaning
    evaluation_data[:,i] = evaluation_data[:,i]-meaning

rng = numpy.random.RandomState(1978)
testing_data_sel = rng.uniform(size=(validation_data.shape[0],)) < 0.2
validation_data = validation_data[testing_data_sel,:]
validation_labels = validation_labels[testing_data_sel]

#========================= rebuild model ===============================================
def Prelu(x, a):
    return theano.tensor.switch(x<0, a*x, x)

def activation(params, x, model_id):
	act = x;
	print('%f-th model with depth %f'%(model_id, (len(params)-2)/3+1))
	for i in range((len(params)-2)/3):
		act = Prelu(theano.tensor.dot(act,params[i*3])+params[i*3+1], params[i*3+2])
	act = theano.tensor.nnet.softmax(theano.tensor.dot(act,params[-2])+params[-1])
	return act
	
x = theano.tensor.matrix('x')
y = theano.tensor.ivector('y')

#========================= uniform blending ================================================
softmax = activation(params1, x, 1)
prediction = theano.tensor.argmax(softmax,axis=1)
error = theano.tensor.mean(theano.tensor.eq(prediction,y))

#===================== compile theano function ================================================
validation_x_shared = theano.shared(numpy.asarray(validation_data,dtype=theano.config.floatX),borrow=True)
validation_labels_shared = theano.tensor.cast(theano.shared(numpy.asarray(validation_labels,dtype=theano.config.floatX),borrow=True),'int32')
validate_proc = theano.function(inputs=[],outputs=error,givens={x:validation_x_shared,y:validation_labels_shared})

evaluation_x_shared = theano.shared(numpy.asarray(evaluation_data,dtype=theano.config.floatX),borrow=True)
predict_proc = theano.function(inputs=[],outputs=prediction,givens={x:evaluation_x_shared})

#===================== GOOOOOOOOOOOOOOOOOOOO ================================================
print('Validating...')
validate_error = validate_proc()
print('--- error : %f'%numpy.mean(validate_error))
print('Predicting...')
evaluation_y_pred = predict_proc()
	
ph48_39 = numpy.loadtxt(paths.pathToMapPhones,dtype='str_',delimiter='\t')
ph48_39_dict = dict(ph48_39)
phi_48 = dict(zip(numpy.arange(0,48),ph48_39[:,0])) # [0,47] -> 48 phonemes
phi_39 = dict(zip(numpy.arange(0,39),list(set(ph48_39[:,1])))) # [0,38] -> 39 phonemes
evaluation_y_pred_str = [phi_39[pred] for pred in evaluation_y_pred]
import csv
with open('prediction_single.csv','wb') as csvfile:
    csvw = csv.writer(csvfile,delimiter=',')
    csvw.writerow(['Id','Prediction'])
    for id_,pred_ in zip(evaluation_ids,evaluation_y_pred_str):
        csvw.writerow([id_,pred_])

print('Done~')
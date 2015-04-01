#
# After training, use predict_blending.py to predict result with uniform blending
# the model can be any size and depth, but the order of params must follow[w1, b1, a1, .... wout, bout]

import paths
import time
import numpy
import cPickle
import theano

#========================= load validate and test data ===================================
f = file(paths.pathToSaveFBANKTrain,'rb')
validation_data = cPickle.load(f)
f.close()
f = file(paths.pathToSaveMFCCTrain,'rb')
validation_data_1 = cPickle.load(f)
f.close()
f = file(paths.pathToSave39Labels,'rb')
validation_labels = cPickle.load(f)
f.close()

f = file(paths.pathToSaveFBANKTest,'rb')
evaluation_data = cPickle.load(f)
f.close()
f = file(paths.pathToSaveMFCCTest,'rb')
evaluation_data_1 = cPickle.load(f)
f.close()
f = file(paths.pathToSaveTestIds,'rb')
evaluation_ids = cPickle.load(f)
f.close()

validation_data = numpy.append(validation_data[:], validation_data_1[:], 1)
evaluation_data = numpy.append(evaluation_data[:], evaluation_data_1[:], 1)


#========================= load models ====================================================
f = file('model/model_best_1.save', 'rb')  # modify your model path here~
params1 = cPickle.load(f)
f.close()
f = file('model/model_best_2.save', 'rb')  # modify your model path here~
params2 = cPickle.load(f)
f.close()
f = file('model/model_best_3.save', 'rb')  # modify your model path here~
params3 = cPickle.load(f)
f.close()
f = file('model/model_best_4.save', 'rb')  # modify your model path here~
params4 = cPickle.load(f)
f.close()
f = file('model/model_best_5.save', 'rb')  # modify your model path here~
params5 = cPickle.load(f)
f.close()
f = file('model/model_best_6.save', 'rb')  # modify your model path here~
params6 = cPickle.load(f)
f.close()
f = file('model/model_best_7.save', 'rb')  # modify your model path here~
params7 = cPickle.load(f)
f.close()
f = file('model/model_best_8.save', 'rb')  # modify your model path here~
params8 = cPickle.load(f)
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
softmax = activation(params1, x, 1)+ activation(params2, x, 2) + activation(params3, x, 3)+ activation(params4, x, 4)+ activation(params5, x, 5)+activation(params6, x, 6)+ activation(params6, x, 7)+ activation(params8, x, 8)
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
with open('prediction_blend.csv','wb') as csvfile:
    csvw = csv.writer(csvfile,delimiter=',')
    csvw.writerow(['Id','Prediction'])
    for id_,pred_ in zip(evaluation_ids,evaluation_y_pred_str):
        csvw.writerow([id_,pred_])

print('Done~')

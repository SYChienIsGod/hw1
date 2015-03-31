#
#After training, use predict.py to produce submission result
#
# ver 0.00a by Tseng, input with FBANK and MFCC

import paths
import time
import numpy
import cPickle
import theano

f = file(paths.pathToSaveFBANKTest,'rb')
evaluation_data = cPickle.load(f)
f.close()

g = file(paths.pathToSaveMFCCTest,'rb')
evaluation_data_1 = cPickle.load(g)
g.close()

f = file('model_best690.save', 'rb')  # modify your model path here~
params = cPickle.load(f)
f.close()

f = file('normalize_parameter_scaling.save', 'rb')  # modify your model path here~
raw_scaling = cPickle.load(f)
f.close()

f = file('normalize_parameter_means.save', 'rb')  # modify your model path here~
raw_means = cPickle.load(f)
f.close()

[ W_hidden_1, b_hidden_1, a_hidden_1, W_hidden_2, b_hidden_2, a_hidden_2, 
W_hidden_3, b_hidden_3,a_hidden_3, W_hidden_4, b_hidden_4, a_hidden_4,W_hidden_5, b_hidden_5, a_hidden_5, W_out, b_out] = params

evaluation_data = numpy.append(evaluation_data[:], evaluation_data_1[:], 1)

#Normalize test data by training data's mean and scale 
for i in range(evaluation_data.shape[1]):
    scaling = raw_scaling[i]
    evaluation_data[:,i] = evaluation_data[:,i]/scaling
    meaning = raw_means[i]
    evaluation_data[:,i] = evaluation_data[:,i]-meaning

f = file(paths.pathToSaveTestIds,'rb')
evaluation_ids = cPickle.load(f)
f.close()

#=========================rebuild model====================================================
def Prelu(x, a):
    return theano.tensor.switch(x<0, a*x, x)

x = theano.tensor.matrix('x')

act_hidden_1 = Prelu(theano.tensor.dot(x,W_hidden_1)+b_hidden_1, a_hidden_1)
act_hidden_2 = Prelu(theano.tensor.dot(act_hidden_1,W_hidden_2)+b_hidden_2, a_hidden_2)
act_hidden_3 = Prelu(theano.tensor.dot(act_hidden_2,W_hidden_3)+b_hidden_3, a_hidden_3)
act_hidden_4 = Prelu(theano.tensor.dot(act_hidden_3,W_hidden_4)+b_hidden_4,a_hidden_4)
act_hidden_5 = Prelu(theano.tensor.dot(act_hidden_4,W_hidden_5)+b_hidden_5,a_hidden_5)

softmax = theano.tensor.nnet.softmax(theano.tensor.dot(act_hidden_5,W_out)+b_out)
prediction = theano.tensor.argmax(softmax,axis=1)

#===========================================================================================


evaluation_x_shared = theano.shared(numpy.asarray(evaluation_data,dtype=theano.config.floatX),borrow=True)

predict_proc = theano.function(inputs=[],outputs=prediction,givens={x:evaluation_x_shared})

evaluation_y_pred = predict_proc()

ph48_39 = numpy.loadtxt(paths.pathToMapPhones,dtype='str_',delimiter='\t')
ph48_39_dict = dict(ph48_39)
phi_48 = dict(zip(numpy.arange(0,48),ph48_39[:,0])) # [0,47] -> 48 phonemes
phi_39 = dict(zip(numpy.arange(0,39),list(set(ph48_39[:,1])))) # [0,38] -> 39 phonemes
evaluation_y_pred_str = [phi_39[pred] for pred in evaluation_y_pred]
import csv
with open('prediction_3.csv','wb') as csvfile:
    csvw = csv.writer(csvfile,delimiter=',')
    csvw.writerow(['Id','Prediction'])
    for id_,pred_ in zip(evaluation_ids,evaluation_y_pred_str):
        csvw.writerow([id_,pred_])

print('Done~')

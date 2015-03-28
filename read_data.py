# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:50:15 2015
1
@author: Jason
"""



NFBANK = 69
NMFCC = 39
import paths
import numpy
import cPickle


#%% Process FBANK Training Data

fbank_train = numpy.loadtxt(paths.pathToFBANKTrain,dtype='float32',delimiter=' ',usecols=range(1,70))

fbank_train_ids = numpy.loadtxt(paths.pathToFBANKTrain,dtype='str_',delimiter=' ',usecols=(0,))

#%% FBank Test Data

fbank_test = numpy.loadtxt(paths.pathToFBANKTest,dtype='float32',delimiter=' ',usecols=range(1,70))
f = file(paths.pathToSaveFBANKTest,'wb')
cPickle.dump(fbank_test,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

#%%

fbank_test_ids = numpy.loadtxt(paths.pathToFBANKTest,dtype='str_',delimiter=' ',usecols=(0,))
f = file(paths.pathToSaveTestIds,'wb')
cPickle.dump(fbank_test_ids,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

#%% Remap MFCC Training Data

mfcc_train = numpy.loadtxt(paths.pathToMFCCTrain,dtype='float32',delimiter=' ',usecols=range(1,40))
mfcc_train_ids = numpy.loadtxt(paths.pathToMFCCTrain, dtype='str_',delimiter=' ', usecols =(0,))

mfcc2i_dict = dict(zip(mfcc_train_ids,range(mfcc_train_ids.shape[0])))

mfcc_train_remapped = numpy.asarray([mfcc_train[mfcc2i_dict[fbank_id]] for fbank_id in fbank_train_ids])
mfcc_train_ids_remapped = numpy.asarray([mfcc_train_ids[mfcc2i_dict[fbank_id]] for fbank_id in fbank_train_ids])

print 'Check MFCC remapping: %i==0' % numpy.sum(mfcc_train_ids_remapped!=fbank_train_ids)

f = file(paths.pathToSaveMFCCTrain,'wb')
cPickle.dump(mfcc_train_remapped,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

#%% Remap MFCC Testing Data

mfcc_test = numpy.loadtxt(paths.pathToMFCCTest,dtype='float32',delimiter=' ',usecols=range(1,40))
mfcc_test_ids = numpy.loadtxt(paths.pathToMFCCTest, dtype='str_',delimiter=' ', usecols =(0,))

mfcc2i_dict = dict(zip(mfcc_test_ids,range(mfcc_test_ids.shape[0])))

mfcc_test_remapped = numpy.asarray([mfcc_test[mfcc2i_dict[fbank_id]] for fbank_id in fbank_test_ids])
mfcc_test_ids_remapped = numpy.asarray([mfcc_test_ids[mfcc2i_dict[fbank_id]] for fbank_id in fbank_test_ids])

print 'Check MFCC remapping: %i==0' % numpy.sum(mfcc_test_ids_remapped!=fbank_test_ids)

f = file(paths.pathToSaveMFCCTest,'wb')
cPickle.dump(mfcc_test_remapped,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

#%% Load phonemes map and create dict


ph48_labels = numpy.loadtxt(paths.pathToLbl,dtype='str_',delimiter=',')

ph48_label_dict = dict(zip(ph48_labels[:,0],ph48_labels[:,1])) # Id -> 48 phonemes

ph48_39 = numpy.loadtxt(paths.pathToMapPhones,dtype='str_',delimiter='\t')
ph48_39_dict = dict(ph48_39) # 48 phonemes -> 39 phonemes
ph48_i = dict(zip(ph48_39[:,0],numpy.arange(0,48))) # 48 phonemes -> [0,47]
phi_48 = dict(zip(numpy.arange(0,48),ph48_39[:,0])) # [0,47] -> 48 phonemes

fbank_48labels = [ph48_i[ph48_label_dict[ident]] for ident in fbank_train_ids] # [0,47] for the fbank ids

f = file(paths.pathToSave48Labels,'wb')
cPickle.dump(numpy.array(fbank_48labels),f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

ph39_i = dict(zip(list(set(ph48_39[:,1])),numpy.arange(0,39))) # 39 phonemes -> [0,38]
phi_39 = dict(zip(numpy.arange(0,39),list(set(ph48_39[:,1])))) # [0,38] -> 39 phonemes

fbank_39labels = numpy.array([ph39_i[ph48_39_dict[ph48_label_dict[ident]]] for ident in fbank_train_ids]) # [0,38] for the fbank ids

f = file(paths.pathToSave39Labels,'wb')
cPickle.dump(numpy.array(fbank_39labels),f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

#%% Save data and labels


f = file(paths.pathToSaveFBANKTrain,'wb')
cPickle.dump(fbank_train,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

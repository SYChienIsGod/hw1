# hw1

Usage

>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python read_data.py
>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_nn.py

(choose CPU or GPU base on your device)

To run 

========Version discription====================================== 

Table
Version     Validate accuracy        Submmit accuracy     
zero        0.586875                 0.622                (MFCC,2 layer) 
0.01        0.595                    0.626                (MFCC,2 layer,ReLU) 
0.01-1      0.604                    X                    (FBANK,3 layer,ReLU) 
0.01-2      0.614 (@2000 epoch)      X                    (FBANK,4 layer,ReLU)
0.02        0.602 (@275  epoch)      X                    (FBANK,2 layer,PReLU)
0.02-1      0.622 (@2000 epoch)      ?                    (FBANK,4 layer,PReLU)
0.03        0.628*(@2000 epoch)      ?                    (FBANK,4 layer,PReLU,Momentum)

*still growing (I'll train a 4000 epoch ver later)

Zero edition: create by Jan 
	it takes about 8 mins on GTX760 for 200 epoch 
	prediction_3.csv on kaggle 
	accuracy 0.622 

Ver 0.01: modify by PHHung 
	it takes about 50 mins on GTX760 for 200 epoch (because of the smaller batch size) 
	prediction_4.csv on kaggle
	accuracy 0.626 

Ver 0.02: modify by HYTseng
	I train it on CPU XD. I would update the timing and accuracy when I get the GPU computation power.
        Major modify: PReLU: http://arxiv.org/pdf/1502.01852v1.pdf

Ver 0.03: modify by PHHung
        it takes about 3 hr on GTX760 for 2000 epoch
        Major modify: momentum

Ver 0.03a: Modified by Jan
        Built our own softmax function but left theano's in place as it's faster.
==================================================================

ToDo:
>learning rate decay
>drop out
>model average
>matplot function
>batch normalization?

Done:
>softmax

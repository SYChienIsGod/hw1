# hw1

Usage
To prepare dataset
>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python read_data.py
To train
>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_nn.py
To generate predict result
>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python predict.py

(choose CPU or GPU base on your device)


========Version discription====================================== 

Table
Version     Validate accuracy        Submmit accuracy     
zero        0.586875                 0.622            (MFCC,2 layer) 
0.01        0.595                    0.626            (MFCC,2 layer,ReLU) 
0.01-1      0.604                    X                (FBANK,3 layer,ReLU) 
0.01-2      0.614 (@2000 epoch)      X                (FBANK,4 layer,ReLU)
0.02        0.602 (@275  epoch)      X                (FBANK,2 layer,PReLU)
0.02-1      0.622 (@2000 epoch)      X                (FBANK,4 layer,PReLU)
0.03        0.628 (@2000 epoch)      0.63684          (as 0.02-1,Momentum)
0.03b	    0.671 (@2000 epoch)      0.65078          (as 0.03,39 Phonemes)
0.03c1      0.682 (@1000 epoch)      X                (as 0.03b,4L wider node)
0.03c2      0.690 (@279 epoch)       0.65481          (as 0.03b,5L)    
0.03d       0.692 (@378 epoch)       0.66202          (as 0.03c2,L2Regression)
0.03d1      0.700 (@511 epoch)       ?                (as 0.03d,7L)
0.03e       0.692 (@328 epoch)       0.66081          (as 0.03e?,FBANK+MFCC)
0.03f       0.706 (@551 epoch)       ?                (as 0.03e,7L,DropOut)   
0.03g       ?                        ?                (as 0.03f,LR decay)
  

Zero edition: create by Jan 
	it takes about 8 mins on GTX760 for 200 epoch 
	prediction_3.csv on kaggle 
	accuracy 0.622 

Ver 0.01: modify by PHHung 
	it takes about 50 mins on GTX760 for 200 epoch 
	prediction_4.csv on kaggle
	accuracy 0.626 

Ver 0.02: modify by HYTseng
	I train it on CPU XD. 
I would update the timing and accuracy when I get the GPU computation power.
        Major modify: PReLU: http://arxiv.org/pdf/1502.01852v1.pdf

Ver 0.03: modify by PHHung
        it takes about 3 hr on GTX760 for 2000 epoch
        Major modify: momentum
	accuracy 0.636 on kaggle

Ver 0.03a: Modified by Jan
        Built our own softmax function but left theano's in place as it's faster.

Ver 0.03b: Modified by Jan
	Prediction changed to 39 phonemes, should increase performance  
	accuracy 0.650 on kaggle

Ver 0.03c: Modified by PHHung
	1.save model as "model_best.save" when there is a better model
	2.separate training & prediction to two different file 
	=>you can terminate training process whenever you like, 
          and then go for prediction
	3.deeper & wider model
    	    
Ver 0.03d: Modified by PHHung
	add L2 regression in cost function,hope to prevent overfiting
	0.03d1: 7layer in/128/256/512/1024/512/256/128/out 

Ver 0.03e: by HYTseng
	(need modify both train_nn.py and predict.py)
	input with FBANK+MFCC, still overfitting, time to think about drop out?

Ver 0.03f: Modified by PHHung
	add DropOut : http://arxiv.org/pdf/1207.0580.pdf

Ver 0.03g: Jan / fix for the learning rate decay

==================================================================

ToDo:
>model average
>matplot function
>batch normalization?
>Use MFCC and FBANK together?

Done:
>momentum gd
>softmax
>learning rate decay
>DropOut

# hw1

Usage <br />

>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python read_data.py <br />
>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_nn.py  <br />

(choose CPU or GPU base on your device) <br />

To run  <br />

========Version discription======================================

Table
Version     Validate accuracy        Submmit accuracy     
zero        0.586875                 0.622                (MFCC,2 layer)
0.01        0.595                    0.626                (MFCC,2 layer,ReLU)
0.01-1      0.604                    X                    (FBANK,3 layer,ReLU)
0.01-2      0.601 (still growing,need more epoch)         (FBANK,4 layer,ReLU)                    


Zero edition: create by Jan<br /> 
>it takes about 8 mins on GTX760 for 200 epoch <br />
>prediction_3.csv on kaggle <br />
>accuracy 0.622 <br />

Ver 0.01: modify by PHHung <br />
>it takes about 50 mins on GTX760 for 200 epoch (because of the smaller batch size) <br />
>prediction_4.csv on kaggle <br />
>accuracy 0.626 <br />

==================================================================

ToDo:
>momentum sgd
>drop out
>model average
>matplot function


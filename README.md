# hw1

Usage <br />

>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python read_data.py <br />
>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_nn.py  <br />

(choose CPU or GPU base on your device) <br />

To run  <br />

========Version discription====================================== <br />
<br />
Table<br />
Version     Validate accuracy        Submmit accuracy <br />    
zero        0.586875                 0.622                (MFCC,2 layer) <br />
0.01        0.595                    0.626                (MFCC,2 layer,ReLU) <br />
0.01-1      0.604                    X                    (FBANK,3 layer,ReLU) <br />
0.01-2      0.601 (still growing,need more epoch)         (FBANK,4 layer,ReLU) <br />                   
<br />
<br />
Zero edition: create by Jan<br /> 
>it takes about 8 mins on GTX760 for 200 epoch <br />
>prediction_3.csv on kaggle <br />
>accuracy 0.622 <br />
<br />
Ver 0.01: modify by PHHung <br />
>it takes about 50 mins on GTX760 for 200 epoch (because of the smaller batch size) <br />
>prediction_4.csv on kaggle <br />
>accuracy 0.626 <br />
<br />
==================================================================<br />
<br />
ToDo:<br />
>momentum sgd<br />
>drop out<br />
>model average<br />
>matplot function<br />


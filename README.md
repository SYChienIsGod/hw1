# hw1

Usage <br />

>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python read_data.py <br />
>> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python train_nn.py  <br />

(choose CPU or GPU base on your device) <br />

To run  <br />

Zero edition: create by Jan<br /> 
	it takes about 8 mins on GTX760 for 200 epoch <br />
	prediction_3.csv on kaggle <br />
	accuracy 0.622 <br />
ver 0.01: modify by PHHung <br />
	it takes about 50 mins on GTX760 for 200 epoch (because of the smaller batch size) <br />
	prediction_4.csv on kaggle <br />
	accuracy 0.626 <br />

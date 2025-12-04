## Firstly i trained this model on all the features including - face,pose,left_hand and right_hand so (33*4) + (468*3) + (21*3) + (21*3) = 1662
## with 1662 feature i achieved 1.98 % accuarcy 

## There after i removed features of face (33 * 4) + (21 * 3) + (21 * 3) = 258 , now trained with 258 features then i got 57.8 % with overfitting.  

### To get more  accuaracy, there can be few more steps - 
1. Data Augumentation (let's see, going to apply without adding more training data) - 53%
2. Bidirectional LSTMs, GRUs and more complex architectures. 
3. HyperParameter Tuning 

### Now i got 71.16% test accuracy after normalizing the data, it was giving only aroud 45% initially it improves after data normalization
### Experiments: 
1. `3 LSTM` Layers with `128,64,64` neurons with `relu` activation function, batch normalization after each layer and dropout with `0.25 percent`
and there are 2 Dense layers with `128,32` neurons with `elu` activation function, batch normalization afer each layer and dropout `0.2 percent`
overfitting - minimul
<a href="exp1.png">exp1</a>

2. This time i used `GRU` and got `76.78` other things was same as *exp1* 
<a href="exp2.png">exp2</a>

3. Same architecture but this time with bidirectional GRU got `73.41%` but there is overfitting in this exp. <a href="exp3.png">exp3</a>
4. 2 LAYERS OF bidirectional GRU with 128,64 units and `relu` activation in layers, in dense layers(as it is as they before 128,32 units) as well. got `70.04%` <a href="exp4.png">exp4</a>
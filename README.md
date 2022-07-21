# Learning-with-type-2-fuzzy-activation-functions-to-improve-the-performance-of-deep-neural-networks
Matlab implementation of the composite learning algorithm in the following paper:

```
A. Beke and T. Kumbasar, "Learning with type-2 fuzzy activation functions to improve the performance of deep neural networks," Engineering Applications of Artificial Intelligence, vol. 85, pp. 372-384, 2019. 
```
We kindly ask that to cite the above mentioned paper if you use type-2 fuzzy activation functions and you publish papers about work that was performed using these codes.

# How to use type-2 fuzzy activation functions in your model  
-Download or clone the repository into a convenient directory. Open MATLAB 2019a or a newer version
-Add default layer folder to your path
-While constructing your model, just add 'SIT2FRU' or 'SIT2FMLayerOpt' class as an activation function and insert the appropriate neuron number as an input argument. In the following, we give an example:  
    FF_NetworkLayers = [ ...
    sequenceInputLayer(size(inputRegressor{1}',1),'Name','FF-Giris')
    fullyConnectedLayer(60,'Name','RegressionLayer')
    sequenceFoldingLayer
    fullyConnectedLayer(6,'Name','FF-Output2')
    SIT2FRU(6, 'SIT2')
    fullyConnectedLayer(1,'Name','FF-Output')
    SIT2FRU(1, 'SIT22')
    regressionLayer]; 
-'SIT2FRU' or 'SIT2FMLayerOpt' classes baically do the some job, the only difference 'SIT2FMLayerOpt' is the time optimized version. 

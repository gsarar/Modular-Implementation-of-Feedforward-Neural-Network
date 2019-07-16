# Modular-Implementation-of-Feedforward-Neural-Network
UCSD CSE 253 HW - Pair Programming with Osman Cihan Kilinc.

The code was developed using Python 3.6. 

Part b)

questionb() is a stand-alone method that makes the subsequent checks for all cases. For each case, it returns the symmetrical centralized difference (numerical approximation) 
and a boolean value that is true if the gradient found by the backpropagation algorithm is within the constraints with respect to the returned numerical approximation.

Except of part f, we only used the same code, where only by changing the config keys, the code can be run for parts c-e. 
In part f for our experiments with ReLU activation, due to weight explosion with ReLU in a neural network with multiple hidden layers, we initialized the weights by multipliying them in the beginning with 0.001. 
Below we provide the necessary configurations for parts c-e. The code for these parts is called neuralnet.py

Part c)
config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'tanh' 
config['batch_size'] = 20  
config['epochs'] = 500  
config['early_stop'] = True  
config['early_stop_epoch'] = 5  
config['L2_penalty'] = 0
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 

Part d)
config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'tanh' 
config['batch_size'] = 100  
config['epochs'] = 550  
config['early_stop'] = True  
config['early_stop_epoch'] = 5  
config['L2_penalty'] = 0.1 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 

config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'tanh' 
config['batch_size'] = 100  
config['epochs'] = 550  
config['early_stop'] = True  
config['early_stop_epoch'] = 5  
config['L2_penalty'] = 0.01 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.001 

config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'tanh' 
config['batch_size'] = 100  
config['epochs'] = 550  
config['early_stop'] = True  
config['early_stop_epoch'] = 5  
config['L2_penalty'] = 0.001 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 

config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'tanh' 
config['batch_size'] = 100  
config['epochs'] = 550  
config['early_stop'] = True  
config['early_stop_epoch'] = 5  
config['L2_penalty'] = 0.0001 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 


Part e)

i) Sigmoid
config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'sigmoid' 
config['batch_size'] = 10  
config['epochs'] = 500  
config['early_stop'] = True  
config['early_stop_epoch'] = 3  
config['L2_penalty'] = 0 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.001 

ii) ReLU

config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'ReLU' 
config['batch_size'] = 50  
config['epochs'] = 500  
config['early_stop'] = True  
config['early_stop_epoch'] = 3  
config['L2_penalty'] = 0 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 

iii) Comparison

config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'ReLU' 
config['batch_size'] = 50  
config['epochs'] = 500  
config['early_stop'] = False  
config['early_stop_epoch'] = 3  
config['L2_penalty'] = 0 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 

config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = sigmoid' 
config['batch_size'] = 50  
config['epochs'] = 500  
config['early_stop'] = False  
config['early_stop_epoch'] = 3  
config['L2_penalty'] = 0 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 

config = {}
config['layer_specs'] = [784, 50, 10]  
config['activation'] = 'tanh' 
config['batch_size'] = 50  
config['epochs'] = 500  
config['early_stop'] = False  
config['early_stop_epoch'] = 3  
config['L2_penalty'] = 0 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 

Part f)

i.
config = {}
config['layer_specs'] = [784, 25 10]  
config['activation'] = 'tanh'
config['batch_size'] = 100 
config['epochs'] = 550 
config['early_stop'] = True 
config['early_stop_epoch'] = 5 
config['L2_penalty'] = 0.1
config['momentum'] = True
config['momentum_gamma'] = 0.9 
config['learning_rate'] = 0.0001

ii.
config = {}
config['layer_specs'] = [784, 50, 10] 
config['activation'] = 'tanh'
config['batch_size'] = 100 
config['epochs'] = 550 
config['early_stop'] = True  
config['early_stop_epoch'] = 5  
config['L2_penalty'] = 0.1 
config['momentum'] = True  
config['momentum_gamma'] = 0.9  
config['learning_rate'] = 0.0001 

iii.
config = {}
config['layer_specs'] = [784, 25, 25, 10] 
config['activation'] = 'tanh' 
config['batch_size'] = 100 
config['epochs'] = 550  
config['early_stop'] = True  
config['early_stop_epoch'] = 5  
config['L2_penalty'] = 0.1  
config['momentum'] = True  
config['momentum_gamma'] = 0.9 
config['learning_rate'] = 0.0001 

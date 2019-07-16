import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

config = {}
config['layer_specs'] = [784, 10, 10, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'tanh' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 100  # Number of training samples per batch to be passed to network
config['epochs'] = 550  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0.1  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

EPS = np.finfo(np.float32).eps

def softmax(x):
	"""
	the code for softmax activation function that takes in a numpy array and returns a numpy array.
	"""
#x:n x class_number
#	x_hat=x - x.max(axis=0)
	exp_x=np.exp(x)
	denominator=np.sum(exp_x,axis=1,keepdims=True)
	output=exp_x/denominator
	return output

def load_data(fname):
	"""
	return 2 arrays X, Y given a pickle file. 
	X should be the input features and Y should be the one-hot encoded labels of each input image i.e
	```shape(X) = n,784``` and ```shape(Y) = n,10```
	"""
	data=pickle.load(open(fname, 'rb'))
	images=data[:,:784]
	orig_labels=data[:,-1]
	labels=np.zeros((data.shape[0],10),dtype=float)
	for i in range(data.shape[0]):
		labels[i,int(orig_labels[i])]=1    
	return images, labels


class Activation:
	def __init__(self, activation_type = "sigmoid"):
		self.activation_type = activation_type
		self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
	def forward_pass(self, a):
		if self.activation_type == "sigmoid":
			return self.sigmoid(a)

		elif self.activation_type == "tanh":
			return self.tanh(a)

		elif self.activation_type == "ReLU":
			return self.ReLU(a)
  
	def backward_pass(self, delta):
		if self.activation_type == "sigmoid":
			grad = self.grad_sigmoid()

		elif self.activation_type == "tanh":
			grad = self.grad_tanh()

		elif self.activation_type == "ReLU":
			grad = self.grad_ReLU()

		return grad * delta
	  
	def sigmoid(self, x):
		"""
		the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
		"""
		self.x = x
		output=1/(1+np.exp(-x))
		return output

	def tanh(self, x):
		"""
		the code for tanh activation function that takes in a numpy array and returns a numpy array.
		"""
		self.x = x
		numerator = (np.exp(x) - np.exp(-x))
		denominator = (np.exp(x) + np.exp(-x))
		output = numerator/denominator
		return output

	def ReLU(self, x):
		"""
		the code for ReLU activation function that takes in a numpy array and returns a numpy array.
		"""
		self.x = x
		output=np.maximum(x,0)
		return output

	def grad_sigmoid(self):
		"""
		the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
		"""
		sigmoid_out=1/(1+np.exp(-self.x))
		grad = np.multiply(sigmoid_out,(1-sigmoid_out))
		return grad

	def grad_tanh(self):
		"""
		the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
		"""
		numerator = (np.exp(self.x) - np.exp(-self.x))
		denominator = (np.exp(self.x) + np.exp(-self.x))
		tanh_out = numerator/denominator
		grad = 1 - tanh_out**2
		return grad

	def grad_ReLU(self):
		"""
		the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
		"""
		intermediate_out = np.maximum(self.x,0)
		intermediate_out /= intermediate_out.max()
		grad = np.ceil(intermediate_out)
		return grad


class Layer():
	def __init__(self, in_units, out_units):
		np.random.seed(42)
		self.w = np.random.randn(in_units, out_units) # Weight matrix
		self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
		self.x = None  # Save the input to forward_pass in this
		self.a = None  # Save the output of forward pass in this (without activation)
		self.d_x = None  # Save the gradient w.r.t x in this
		self.d_w = None  # Save the gradient w.r.t w in this
		self.d_b = None  # Save the gradient w.r.t b in this
		self.dw_velocity=np.zeros((in_units, out_units)).astype(np.float32) 
		self.db_velocity=np.zeros((1, out_units)).astype(np.float32) 

	def forward_pass(self, x):
		"""
		Write the code for forward pass through a layer. Do not apply activation function here.
		"""
		self.a = np.dot(x,self.w)+self.b
		self.x = x
		return self.a
  
	def backward_pass(self, delta):
		"""
		Write the code for backward pass. This takes in gradient from its next layer as input,
		computes gradient for its weights and the delta to pass to its previous layers.
		"""
		self.d_w = np.dot(self.x.T,delta)
		self.d_b = np.sum(delta, axis=0)
		self.d_x = np.dot(delta,self.w.T)
		return self.d_x

	  
class Neuralnetwork():
	def __init__(self, config):
		self.layers = []
		self.x = None  # Save the input to forward_pass in this
		self.y = None  # Save the output vector of model in this
		self.targets = None  # Save the targets in forward_pass in this variable
		for i in range(len(config['layer_specs']) - 1):
			self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
			if i < len(config['layer_specs']) - 2:
				self.layers.append(Activation(config['activation']))  
		self.acc_train = []
		self.acc_valid = []
		self.loss_train = []
		self.loss_valid = []

	def forward_pass(self, x, targets=None):
		"""
		Write the code for forward pass through all layers of the model and return loss and predictions.
		If targets == None, loss should be None. If not, then return the loss computed.
		"""
		counter=0       
		self.x = x
		self.targets = targets
		for layer in self.layers:
			x = layer.forward_pass(x)
			counter+=1          
		self.y = softmax(x)
		loss = self.loss_func(self.y, self.targets)     
		return loss, self.y

	def loss_func(self, logits, targets):
		'''
		find cross entropy loss between logits and targets
		'''
		output = -np.sum(targets * np.log(logits))/(logits.shape[0] * logits.shape[1]) 
		return output
	
	def backward_pass(self):
		'''
		implement the backward pass for the whole network. 
		hint - use previously built functions.
		'''
#		delta=(self.targets-self.y)/(self.y.shape[0] * self.y.shape[1])
		delta=(self.targets-self.y)
		for layer in reversed(self.layers):
			delta = layer.backward_pass(delta)		

def plot_graph(y1, y2, ylabel = 'Error', legend=['training loss','validation loss'], title = 'Cross-Entropy Loss (Error)', fname='loss_sigmoid500.png'):
	rdir = "./results/"
	plt.figure(figsize=(10,7))
	plt.plot(y1)
	plt.plot(y2)
	plt.legend(legend,fontsize='10');
	plt.grid(b=None, which='major', axis='both');
	plt.minorticks_on()
	#plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
	plt.ylabel(ylabel);
	plt.xlabel('Epoch number');
	plt.title(title);
	plt.savefig(rdir+fname)

def trainer(model, X_train, y_train, X_valid, y_valid, config):
	"""
	Write the code to train the network. Use values from config to set parameters
	such as L2 penalty, number of epochs, momentum, etc.
	"""
	model.train_acc=[]  
	model.val_acc=[] 
	model.train_loss=[]  
	model.val_loss=[] 
	b_size=config['batch_size'] 
	num_batches=int(X_train.shape[0]/b_size)
	early_stop_count = 0
	prev_loss_valid=float('inf')
	# last_loss_valid = float('inf')
	best_loss_valid =  float('inf')
	for epoch in range(config["epochs"]):
		#shuffle
		shuffle_ind=np.random.permutation(X_train.shape[0])
		X_train=X_train[shuffle_ind,:]
		y_train=y_train[shuffle_ind]
		for b_num in range(num_batches):
			batch = X_train[b_num*b_size:(b_num+1)*b_size,:]
			batch_target = y_train[b_num*b_size:(b_num+1)*b_size,:]
			loss, preds = model.forward_pass(x = batch, targets = batch_target) # Forward Pass - Forward Propagation
			#print(loss)
			model.backward_pass() # Find Gradients - Backpropagation
			# Apply Gradients
			for layer in model.layers:
				if isinstance(layer, Layer):
					if config["momentum"]:
						layer.dw_velocity = config['momentum_gamma']*layer.dw_velocity + config["learning_rate"]*layer.d_w - config["learning_rate"]*config["L2_penalty"]*layer.w
						layer.db_velocity = config['momentum_gamma']*layer.db_velocity + config["learning_rate"]*layer.d_b
					else: 
						layer.dw_velocity = config["learning_rate"]*layer.d_w - config["learning_rate"]*config["L2_penalty"]*layer.w
						layer.db_velocity = config["learning_rate"]*layer.d_b
					layer.w += layer.dw_velocity
					layer.b += layer.db_velocity
		loss_train, preds_train = model.forward_pass(x = X_train, targets = y_train) # Forward Pass - Forward Propagation
		acc_train = np.sum(np.argmax(preds_train,axis=1)==np.argmax(y_train,axis=1))/X_train.shape[0]
		print("acc_train = ", acc_train)
		print("loss_train = ", loss_train)
		model.train_acc.append(acc_train)
		model.train_loss.append(loss_train)       
		loss_valid, preds_valid = model.forward_pass(x = X_valid, targets = y_valid) 
		acc_valid=np.sum(np.argmax(preds_valid,axis=1)==np.argmax(y_valid,axis=1))/X_valid.shape[0]
		print("acc_valid = ", acc_valid)
		print("loss_valid= ", loss_valid)
		model.val_acc.append(acc_valid)
		model.val_loss.append(loss_valid)       
        
		if best_loss_valid >= loss_valid:
			best_model = copy.deepcopy(model) # copy model
			best_loss_valid = loss_valid
            
		if loss_valid > prev_loss_valid:
			early_stop_count += 1
			prev_loss_valid = loss_valid
			print("WORSE LOSS VALID!",early_stop_count)
			if early_stop_count >= config['early_stop_epoch']:
				break
		else:
			early_stop_count=0
			prev_loss_valid = loss_valid
	return best_model
  
def test(model, X_test, y_test, config):
	"""
	Write code to run the model on the data passed as input and return accuracy.
	"""
	# forward pass
	# get loss and accuracy
	loss, preds = model.forward_pass(x = X_test, targets = y_test) # Forward Pass - Forward Propagation
	accuracy=np.sum(np.argmax(preds,axis=1)==np.argmax(y_test,axis=1))/X_test.shape[0]
	return accuracy

def questionb():
	''' wrapper for quesiton b in the second programming assignment of cse 253 of winter 2019'''
	train_data_fname = 'MNIST_train.pkl'
	valid_data_fname = 'MNIST_valid.pkl'
	test_data_fname = 'MNIST_test.pkl'
	model = Neuralnetwork(config)
	X_train, y_train = load_data(train_data_fname)
	X_valid, y_valid = load_data(valid_data_fname)
	X_test, y_test = load_data(test_data_fname)
	selected_sample = (X_train[0].reshape((1, 784)), y_train[0])
	return check_model(model, selected_sample)

#Gradient Check
def select_indices(model):
	''' randomly selects the indices of the 2 weights from input-to-hidden layer, 2 weights from hidden-to-output layer, 1 bias from input-to-hidden layer and 1 bias from hidden-to output layer '''
	selected_indices = [[np.random.choice(model.layers[0].w.shape[0]), np.random.choice(model.layers[0].w.shape[1])]]
	selected_indices.append([np.random.choice(model.layers[0].w.shape[0]), np.random.choice(model.layers[0].w.shape[1])])
	selected_indices.append([np.random.choice(model.layers[0].b.shape[1])])
	selected_indices.append([np.random.choice(model.layers[-1].w.shape[0]), np.random.choice(model.layers[-1].w.shape[1])])
	selected_indices.append([np.random.choice(model.layers[-1].w.shape[0]), np.random.choice(model.layers[-1].w.shape[1])])
	selected_indices.append([np.random.choice(model.layers[-1].b.shape[1])]) 
	return selected_indices

def select_weights(model, isPrint = False):
	''' randomly selects 2 weights from input-to-hidden layer, 2 weights from hidden-to-output layer, 1 bias from input-to-hidden layer and 1 bias from hidden-to output layer '''
	selected_indices = select_indices(model)
	inp_to_hid_0 = model.layers[0].w[selected_indices[0][0], selected_indices[0][1]]
	inp_to_hid_1 = model.layers[0].w[selected_indices[1][0], selected_indices[1][1]]
	hid_b = model.layers[0].b[0, selected_indices[2][0]]
	hid_to_out_0 = model.layers[-1].w[selected_indices[3][0], selected_indices[3][1]]
	hid_to_out_1 = model.layers[-1].w[selected_indices[4][0], selected_indices[4][1]]
	out_b = model.layers[-1].b[0, selected_indices[-1][0]]
	if isPrint:
		print(selected_indices)
		print(inp_to_hid_0)
		print(inp_to_hid_1)
		print(hid_b)
		print(hid_to_out_0)
		print(hid_to_out_1)
		print(out_b)
	return selected_indices, inp_to_hid_0, inp_to_hid_1, hid_b, hid_to_out_0, hid_to_out_1, out_b

def set_weights(model, layer_idx, weight_idx, value):
	''' given layer id, and the indices of the weight, changes the value of model's weight to the given value ''' 
	if len(weight_idx) == 2:
		model.layers[layer_idx].w[weight_idx[0], weight_idx[1]] = value
	else:
		model.layers[layer_idx].b[0][weight_idx[0]] = value
	return model

def get_grad(model, layer_idx, weight_idx):
	''' given layer id, and the indices of the weight, returns the weight's gradient ''' 
	if len(weight_idx) == 2:
		return model.layers[layer_idx].d_w[weight_idx[0], weight_idx[1]]
	else:
		return model.layers[layer_idx].d_b[weight_idx[0]]

def check_model(model, sample):
	''' the sanity check is executed within this method as asked in the homework'''
	selected_indices, inp_to_hid_0, inp_to_hid_1, hid_b, hid_to_out_0, hid_to_out_1, out_b = select_weights(model)
	# repeat check_gradient
	inp_to_hid_0_numApprox, inp_to_hid_0_sanityCheck, _ = check_gradient(model, sample, inp_to_hid_0, 0, selected_indices[0])
	inp_to_hid_1_numApprox, inp_to_hid_1_sanityCheck, _ = check_gradient(model, sample, inp_to_hid_1, 0, selected_indices[1])
	hid_b_numApprox, hid_b_sanityCheck, _ = check_gradient(model, sample, hid_b, 0, selected_indices[2])
	hid_to_out_0_numApprox, hid_to_out_0_sanityCheck, _ = check_gradient(model, sample, hid_to_out_0, -1, selected_indices[3])
	hid_to_out_1_numApprox, hid_to_out_1_sanityCheck, _ = check_gradient(model, sample, hid_to_out_1, -1, selected_indices[4])
	out_b_numApprox, out_b_sanityCheck, _ = check_gradient(model, sample, out_b, -1, selected_indices[5])
	return (inp_to_hid_0_numApprox, inp_to_hid_0_sanityCheck), (inp_to_hid_1_numApprox, inp_to_hid_1_sanityCheck), \
		(hid_b_numApprox, hid_b_sanityCheck),(hid_to_out_0_numApprox, hid_to_out_0_sanityCheck), (hid_to_out_1_numApprox, hid_to_out_1_sanityCheck), \
		(out_b_numApprox, out_b_sanityCheck)

def compareGradApprox(numApprox, grad, B_EPS):
    diffApprox = abs(grad-numApprox)
    if(diffApprox <= B_EPS**2):
    	return True
    else:
    	return False

def check_gradient(model, sample, weight, layer_idx, weight_idx):
	''' for each selected weight, returns the results of the sanity check ''' 
	B_EPS = 10**(-2)
	# DECREMENTED WEIGHT
	weight_inc = weight + B_EPS
	model = set_weights(model, layer_idx, weight_idx, weight_inc)
	loss_inc, preds_inc = model.forward_pass(sample[0], sample[1])
	model.backward_pass()
	# INCREMENTED WEIGHT
	weight_dec = weight - B_EPS
	model = set_weights(model, layer_idx, weight_idx, weight_dec)
	loss_dec, preds_dec = model.forward_pass(sample[0], sample[1])
	model.backward_pass()
	# NUMERICAL APPROXIMATION FROM EQUATION IN Q.B.
	numApprox = (loss_dec - loss_inc) / (2*B_EPS)
	# COMPARE TO MODEL's BACKWARD PASS METHOD
	model = set_weights(model, layer_idx, weight_idx, weight)
	loss_dec, preds_dec = model.forward_pass(sample[0], sample[1])
	model.backward_pass()
	grad = get_grad(model, layer_idx, weight_idx)/(preds_dec.shape[0] * preds_dec.shape[1])
	return numApprox, compareGradApprox(numApprox,grad,B_EPS), model

if __name__ == "__main__":
	train_data_fname = 'MNIST_train.pkl'
	valid_data_fname = 'MNIST_valid.pkl'
	test_data_fname = 'MNIST_test.pkl'

	### Train the network ###
	model = Neuralnetwork(config)
	X_train, y_train = load_data(train_data_fname)
	X_valid, y_valid = load_data(valid_data_fname)
	X_test, y_test = load_data(test_data_fname)
	trainer(model, X_train, y_train, X_valid, y_valid, config)
	test_acc = test(model, X_test, y_test, config)

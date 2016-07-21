import numpy as np
from load_data import *

train_data = processData(url='images/train')
test_data = processData(url='images/test')

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import SigmoidLayer, LinearLayer

network = FeedForwardNetwork()

# Create layer for neural network
# 3 layers: 1 in, 1 hidden, 1 out
inLayer = LinearLayer(100, name='inLayer')
outLayer = SigmoidLayer(2, name='outLayer')
hiddenLayer = SigmoidLayer(36, name='hiddenLayer')

network.addInputModule(inLayer)
network.addOutputModule(outLayer)
network.addModule(hiddenLayer)

from pybrain.structure import FullConnection

# Create connection between layers

in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

network.addConnection(in_to_hidden)
network.addConnection(hidden_to_out)

network.sortModules()

# Train network using Backpropagation

from pybrain.supervised.trainers import BackpropTrainer

trainer = BackpropTrainer(network, dataset=train_data, momentum=0.1, weightdecay=0.01, verbose=True)

# Train 500 times
trainer.trainEpochs(500)


# Calculate accuracy


sum = 0
for i in range(test_data['input'].shape[0]):
	result = np.round(network.activate(test_data['input'][i]))
	if (np.sum(result == test_data['target'][i]) == 2):
		sum += 1
		
print('Test accuracy: %d/%d' % (sum, test_data['target'].shape[0]))

		

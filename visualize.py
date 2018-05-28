import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


selectors = pickle.load( open( 'l2_l1_reg/020', 'rb' ) )

#we throw out the biases because they're not important for pruning
weights = [s for s in selectors if len(s.shape)==4]

weights = [w.reshape((w.shape[0],w.shape[1])) for w in weights]
weights = [np.abs(w) for w in weights]
# weights = [w/np.std(w) for w in weights]

max_width = 600


weights = [np.pad(w,((0,0),(0,600-w.shape[1])),'constant',constant_values=-0.01) for w in weights]

weights = np.concatenate(weights,axis=0)
print(weights.shape)

imgplot = plt.imshow(weights)
plt.show()

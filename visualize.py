import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


selectors = pickle.load( open( '001_reg/013', 'rb' ) )

#we throw out the biases because they're not important for pruning
weights = [s for s in selectors if len(s.shape)==4]

shaped_weights = [w.reshape((w.shape[0],w.shape[1])) for w in weights]
abs_weights = [np.abs(w) for w in shaped_weights]
normalized_weights = [w/np.std(w) for w in abs_weights]

max_width = 600


padded_weights = [np.pad(w,((0,0),(0,600-w.shape[1])),'constant',constant_values=0) for w in normalized_weights]

concatenated = np.concatenate(padded_weights,axis=0)
print(concatenated.shape)

imgplot = plt.imshow(concatenated)
plt.show()

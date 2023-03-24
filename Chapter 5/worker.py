import numpy as np 
import sys 

def square2(i, x, queue):
    print("In process {}".format(i,))
    queue.put(np.square(x))
    sys.stdout.flush()
    

def square(x):
    return x*x

def shit(k):
    return k*2
import numpy as np
import random

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    - Reference : http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/ 
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, gd = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        
        x[ix] += h
        random.setstate(rndstate)
        fx_p, _ = f(x)
        x[ix] -= 2*h
        random.setstate(rndstate)
        fx_n, _ = f(x)
        x[ix] += h
        grad = (fx_p - fx_n) / (2*h)
        
        diff = abs(grad - gd[ix]) / max(1, abs(grad), abs(gd[ix]))
        
        if diff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (gd[ix], grad)
            return

    
        it.iternext() # Step to next dimension

    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""

if __name__ == "__main__":
    sanity_check()

import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

#Basically, we will give the algorithm random parameters
#Cost function will be used to see if its a big error/difference or small difference to the real value/price
#Gradient Descent will be used to alter the w and b parameters till it gives us a really close answer to the real value/price that we gave the algorithm

#Later, these final parameters can be used to predict the value/price for unknown values of features. 

"""
Housing Price Prediction: the training dataset contains three examples with four features each. (size, bedrooms, floors, and age)
(the size is in sqft rathwe than 1000 sqft, which causes some problems (we'll see this later)
the sizes may be causing problems because the numbers are so much larger in comparison to the other features )

| Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  


"""

#Running the following code to create X_train and Y_train variables, which will contain our example values

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

#Displaying the input data / training data that is stored in the NumPy Array / Matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)


#We have parameters w and b that we need to initialise.
#w will be a vector with n elements, containing the parameter associated with a feature
# in our dataset, n=4. because we have 4 features
# b is a scalar parameter. (w1x1+w2x2....+wnxn + b ) like the y-intercept kinda

#initialising these parameters (close to optimal values for demonstration purposes)
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


#Model Prediction with Multiple Variables
# model's prediction with multiple variables is given by the linear model:
# 𝑓𝐰,𝑏(𝐱)=𝑤0𝑥0+𝑤1𝑥1+...+𝑤𝑛−1𝑥𝑛−1+𝑏
# In vector notation, it's 𝑓𝐰,𝑏(𝐱)=𝐰⋅𝐱+𝑏


#Single Prediction (element by element, loop iteration) 

def predict_single_loop(x, w, b): # Calling this function only once 
    # will mean that it only uses the training data of one example. (First row)
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0] # x.shape[0] tells u how many rows--  x.shape[1] tells u how many columns 
    # which tells it how many times it needs to iterate through the columns to multiply and add everything up.
    # seen in the model above (multiplying features with parameters, and adding b at the very end)
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i         
    p = p + b                
    return p

# get a row from our training data 
# Remember X_train contains the values of the features only, not the price)
x_vec = X_train[0,:] # Giving it one row (one training example's features to make a prediction out of)
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")


#Doing the single prediction again, but using VECTORS! Vectorised Version
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b      # dot product of x (features) and w (parameters)
    return p    

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")


# ------------
# As you can see from comparing the two methods above, both methods reached the same prediction for the same parameters.
# However, the vectorised version takes much less time as it makes use of optimisations provided by NumPy.
# Makes use of Parralization. (vector operations makes use of this. Calculations are done in parallel instead of iteratively)


#Computing Cost with Multiple Variables
# --- Computing the Cost Function (basically how big the error / difference between the prediction and actual value is)
# This difference is used to steer the parameters and adjust them so they'll give predictions closer to the actual value

# Equation for the cost function with multiple variables: 𝐽(𝐰,𝑏)=(1/2𝑚)*(∑𝑖=0 to 𝑚−1 of (𝑓𝐰,𝑏(𝐱(i)))−𝑦(𝑖))^2 )
# where 𝑓𝐰,𝑏(𝐱^(i))=𝐰⋅𝐱^(i)+𝑏. w and x^(i) here are vectors rather than scalars because we have multiple variables

# The following code uses a standard pattern where a 'for' loop over all 'm' number of rows/examples is used.
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0] # tells u the number of rows. X.shape[1] tells you the number of columns 
    cost = 0.0
    for i in range(m):          #for every training data
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot) #every row's value is calculated
        cost = cost + (f_wb_i - y[i])**2       #scalar # cost function, comparing every row's predicted value to the actual value
    cost = cost / (2 * m)                      #scalar    #cost function
    return cost #returning the error/difference 

# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')



#---- -----
# Gradient Descent with Multiple Variables

# For multiple variables: 
# Repeat until convergence. for j=0..n-1 where j is features
# wj = wj - alpha * ( delta J(w,b)/delta wj )
# b = b - alpha * (delta J(w,b) / delta b)

# n is the number of features.
# m is the number of training examples in the data set
# f w,b (x^(i)) is the model's prediction, where y^(i) is the target value

# delta J(w,b) / delta wj = 1/m * sum from i=0 to m-1 of (fw,b(x^(i)) - y^(i)) * xj^(i)
# delta J(w,b) / delta b = 1/m * sum from i=0 to m-1 of (fw,b(x^(i)) - y^(i))


# -- Calculating delta J(w,b) / delta wj and delta J(w,b) / delta b to use in the gradient descent equations to update the parameters
# In this version, 
# Outer loop over all m training examples
#   delta J(w,b) / delta b can be computed directly and accumulated
# In second inner loop over all n features
# delta J(w,b) / delta wj is computed for each wj

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples (Rows ), number of features (Columns ))
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

# Gradient Descent where you update the parameters now
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function. This line copies the w initial parameters 
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing


# initialize parameters
initial_w = np.zeros_like(w_init) # creates an array with the same shape as w_init, but with zero's
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape # discarding the number of features per row as we're only using m here
#Predictions using our new values, 0.2f decides the formatting of the answer, like 2 decimal places.
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4)) # two plots side by side.
# constrained layout = true makes sure the plots don't overlap
ax1.plot(J_hist) # cost function values are stored in the J list. ax1 is the first plot
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)") # Tail shows the end of the optimisation
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()

#since the graph ax2 shows that cost function is still declining and since our results aren't very accurate, we need to fix this
# Perhaps we can change the initialistion values we used for the gradient descent algorithm..? The next lab will improve this
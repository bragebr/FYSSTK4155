from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer


#Loss fuction
def loss(ao,Y):
    if c == 1: 
        #accuracy
        L = accuracy_score(ao,Y)
    else:
        #MSE
        L = (1.0/len(Y))*np.sum((Y - ao)**2, axis=0) 
    return L



#Derivative of loss function
def dL(ao,Y):
    if c == 1: 
        #derivative of Cross Entrypy
        dl = ao-Y
    else:
        #derivative of MSE
        dl = (2/len(Y))*(ao - Y)
    return dl


# Activation functions 
def activation(code, z):
    if code == 'sigmoid': 
        f = 1/(1 + np.exp(-z))
        
    elif code =='relu':
        zero = np.zeros(z.shape)
        f = np.max([z,zero], axis=0)  
        
    elif code == 'leakrelu':
        f = np.where(z > 0, z, a*z)
   
    elif code == 'elu':
        f = np.where(z > 0, z, a*(np.exp(z)-1)) 
        
    elif code == 'softmax':
        f = np.exp(z)/np.sum(np.exp(z),axis=0)
        
    elif code == 'linear':
        f = z
    elif code == 'normtanh':
            f = 1/2*(np.tanh(z)+1)
    return f


#Derivative of the activation functions
def dA(code, x):
    if code == 'sigmoid': 
        df = x*(1-x)
        
    elif code =='relu':
        df = 1 * (x > 0)
          
    elif code == 'leakrelu':
        df = np.where(x > 0, 1, a)  
         
    elif code == 'elu':
        df = np.where(x > 0, 1, activation('elu',x)*a)  
        
    elif code == 'softmax':
        df = x*(1-x) #if i=j) 
        #df = -x_i *x_j # if i != j
        # not properly implemented
        
    elif code == 'linear':
        df = 1
        
    elif code == 'normtanh':
        df = 1/(2*np.cosh(x)**2)       
    return df

    
   

def feed_forward(X):
        
    # weighted sum of inputs to the hidden layer
    zh = X @ Wh + Bh
    # activation in the hidden layer
            
    ah = activation(a_hidden,zh)

    # weighted sum of inputs to the output layer
    zo = ah @ Wo + Bo
    ao = activation(a_out,zo)

    # for backpropagation need activations in hidden and output layers
    return ah, ao


def backpropagation(X, Y):
    ah, ao = feed_forward(X)  

    # error in the output layer
    eo = dL(ao,Y) # derivative of the loss function
    
    # error in the hidden layer
    #eh = dL/dz * dz/da * da/dx #chain rule
    eh = eo @ Wo.T * dA(a_hidden,ah)
    
    # gradients for the output layer
    #dWo = dL() @ da() @ ah
    dWo = ah.T @ eo
    dBo = np.sum(eo, axis=0)
   
    # gradient for the hidden layer
    dWh = X.T @ eh
    dBh = np.sum(eh, axis=0)

    return dWo, dBo, dWh, dBh



def GDM(X,Y,eta,lmbd, Wo, Bo, Wh, Bh):
    # Reset weight gradient for test in While-loop
    dWo_t = np.random.randn(n_hidden_neurons, n_categories)
    
    Change = list(np.zeros(2*2))
    epoch = 0
    while np.linalg.norm(dWo_t) > lim :
        epoch +=1
        gradients = list(backpropagation(X, Y))
        
        for i in range(len(gradients)): 
            NewChange = eta * gradients[i] + momentum * Change[i]
            gradients[i] -= NewChange
            Change[i] = NewChange
            
        # regularization term gradients
        gradients[0] += lmbd * Wo
        gradients[2] += lmbd * Wh
        
        
        # Update weights with learn rate
        Wo -= eta * gradients[0]
        Bo -= eta * gradients[1]
        Wh -= eta * gradients[2]
        Bh -= eta * gradients[3]
        
        if epoch > itter_limit:
            print("Reach itteration limit of:  ", itter_limit, "itterations") 
            break
        
        # Calculate gradient for whole Train dataset 
        # to use as a stop criteria 
        dWo_t, dBo_t, dWh_t, dBh_t  = backpropagation(X_train, Y_train) 
                
    #return gradients #dWo, dBo, dWh, dBh
    return Wo, Bo, Wh, Bh



def SGD(X,Y,eta,lmbd, Wo, Bo, Wh, Bh):
    # Reset weight gradient for test in While-loop
    dWo_t = np.random.randn(n_hidden_neurons, n_categories)
    
    # loop over epochs, through NN, 
    # stops when norm of gradient is smaller than lim
    step = 0
    #list of momentums for ADAM
    M1 = list(np.zeros(2*2)) # 2 layers (h and o) *2 becaus two gradienst, dB and dW in each layer
    M2 = list(np.zeros(2*2))
    
    epoch = 0
    while np.linalg.norm(dWo_t) > lim :
        epoch +=1
        step = step+1
        b1 = 0.9
        b2 = 0.999
        delta = 10**-8
        for i in range(nbatch):
            r_i = M * np.random.randint(nbatch)
            xi = X[r_i:r_i + M]
            yi = Y[r_i:r_i + M]
            
            gradients = list(backpropagation(xi, yi)) # tuple of grads = dWo, dBo, dWh, dBh
            #ADAM
            for i in range(len(gradients)):
                g = gradients[i] 
                M1[i] = b1*M1[i] + (1.0-b1)*g
                M2[i] = b2*M2[i] + (1.0-b2)*np.multiply(g,g)
                term1 = M1[i] / (1.0-b1**step)
                term2 = M2[i] / (1.0-b2**step)
                gradients[i] -= eta*term1/(np.sqrt(term2)+delta)
           
            
        # regularization term gradients
        gradients[0] += lmbd * Wo
        gradients[2] += lmbd * Wh
        
        
        # Update weights with learn rate
        Wo -= eta * gradients[0]
        Bo -= eta * gradients[1]
        Wh -= eta * gradients[2]
        Bh -= eta * gradients[3]
        
        if epoch > itter_limit:
            print("Reach itteration limit of:  ", itter_limit, "itterations") 
            break
        
        # Calculate gradient for whole Train dataset 
        # to use as a stop criteria 
        dWo_t, dBo_t, dWh_t, dBh_t  = backpropagation(X_train, Y_train) 
                
    #return gradients #dWo, dBo, dWh, dBh
    return Wo, Bo, Wh, Bh

def predict(X):
    ah, ao = feed_forward(X)
  
    if c == 1: 
        #pred = np.argmax(ao, axis=1)
        pred = np.zeros(len(ao))
        for i in range(len(ao)):
            if np.max(ao[i,:]) > 0.5:
                pred[i] = 1 
            else:
              pred[i] = 0 
    
    else: pred = ao
       
    return pred



def Get_Data():
    ##Exponetial
    if data_type == "ExpReg": 
        noise = 0.1*np.random.rand(n,1)
        X = 2*np.random.rand(n,1) 
        #Y = 3*np.exp(2*X) + 4 + noise
        Y = np.exp(-X**2) + 1.5 * np.exp(-(X-2)**2) + 0.5 * np.random.normal(0,0.1,X.shape)

    ##Regression
    elif data_type == "Regress":
        noise = 0.1*np.random.rand(n,1)
        X = 2*np.random.rand(n,1)
        Y = 3*X + 4 + noise
        

    ##Binary:
    elif data_type == "Binary":
        #X = np.array([ [0, 0], [0, 1], [1, 0],[1, 1]],dtype=np.float64)
        X = np.array([ [0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], 
                       [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1],
                       [0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], 
                       [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]],
                     dtype=np.float64)
        #yAND = np.array( [ 0, 1 ,1, 1])
        yAND = np.array( [ 0, 1 ,1, 1, 0, 1 ,1, 1, 0, 1 ,1, 1,
                           0, 1 ,1, 1, 0, 1 ,1, 1, 0, 1 ,1, 1])
        # The XOR gate
        #yXOR = np.array( [ 0, 1 ,1, 0])
        yXOR = np.array( [ 0, 1, 1, 0, 0, 1, 1, 0, 0, 1 ,1, 0,
                           0, 1, 1, 0, 0, 1, 1, 0, 0, 1 ,1, 0])
        # The OR gate
        #yOR = np.array( [ 0, 1 , 1, 1])
        yOR = np.array( [ 0, 1 , 1, 1, 0, 1 , 1, 1, 0, 1 , 1, 1,
                          0, 1 , 1, 1, 0, 1 , 1, 1, 0, 1 , 1, 1])
        Y = np.reshape(yXOR,(len(yXOR),1))

    ##Brestcanser Data
    elif data_type == "Cancer":
        #Download breast cancer dataset
        cancer = load_breast_cancer()      
        #Feature matrix of 569 rows (samples) and 30 columns (parameters)
        Data  = cancer.data 
        # 0 for benign and 1 for malignant  
        Y = cancer.target       
        #labels  = cancer.feature_names[0:30]
        #print(labels)
        """
        ['mean radius0' 'mean texture1' 'mean perimeter2' 'mean area3'
         'mean smoothness4' 'mean compactness5' 'mean concavity6'
         'mean concave points7' 'mean symmetry8' 'mean fractal dimension9'
         'radius error10' 'texture error11' 'perimeter error12' 'area error13'
         'smoothness error14' 'compactness error15' 'concavity error16'
         'concave points error17' 'symmetry error18' 'fractal dimension error19'
         'worst radius20' 'worst texture21' 'worst perimeter22' 'worst area23'
         'worst smoothness24' 'worst compactness25' 'worst concavity26'
         'worst concave points27' 'worst symmetry28' 'worst fractal dimension29']
        Parameters that does not separat well: 
            Smootness, Compactness, Symmetry, Texture
        """
        """
        temp1=np.reshape(Data[:,20],(len(Data[:,20]),1)) 
        temp2=np.reshape(Data[:,23],(len(Data[:,23]),1))
        X=np.hstack((temp1,temp2))      
        temp=np.reshape(Data[:,22],(len(Data[:,22]),1))
        X=np.hstack((X,temp)) 
        
        
        """
        temp1=np.reshape(Data[:,0],(len(Data[:,0]),1)) # Radius
        temp2=np.reshape(Data[:,2],(len(Data[:,2]),1)) # Perimeter
        X=np.hstack((temp1,temp2))      
        temp=np.reshape(Data[:,3],(len(Data[:,3]),1))  # Area
        X=np.hstack((X,temp))       
        
        
        #X = Data[:,0:5]
        Y = np.reshape(Y, (len(Y),1))
    
    return X, Y


def Plot_XY (X,Y):
    y_pred = predict(X)
    
    plt.figure()

    if c == 0:
        plt.plot(X,Y,'+',label="Real")
        plt.title("Real and Target values")
    else: plt.title("Target values")
    plt.plot(X,y_pred,'.',label="Target")
    plt.legend()
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    plt.figure()
    plt.plot(X,Y,'.',label="Real")
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Real values")
    plt.legend()
    plt.show()
    return 0



####################################################

#Setting up variables
np.random.seed(0)

# Possible activation functions: 
# sigmoid, relu, leakrelu, elu, softmax, linear, normtanh
a_hidden = "leakrelu" # hidden layer
a_out = "normtanh"    # output layer

# Possible data types:
# Binary, ExpReg, Regress, Cancer
data_type = "Cancer"

# 1 if used: 
use_sgd = 0
use_gdm = 0
neuro_loops = 0

n = 500         # nuber of datapoint in regression
Nh = 2         # Number of neurons in hidden layer
epochs= 150     # Number of epochs, itterations throug NN
lim = 10**-5    # Gradient limit
itter_limit = 1000  # Itterations limit if gradient does not converge
eta_vals = np.logspace(-5, -1, 9)   #Learn rate values
lmbd_vals = np.logspace(-5, -1, 9)   # Penalty values
neuron_vals = np.linspace(1,20,num=20).astype(int)


M = 20   #size of each minibatch in SGD
momentum = 0.01 # Momentum in GD with momentum

#Hard-coded best values for Eta and lambda for plotting
eta_best = 10**-4
lmbd_best = 10**-5
Nh_best = 2 

a = 0.001 #Alpha parameter in elu and leaky relu

if data_type == "Binary" or data_type == "Cancer" : 
    c = 1
else: c = 0
# to chaneg between AND OR XOR go into the data_type definition 
# and change manually

#########################################
# Data:
X, Y = Get_Data()
#Split in test and train data set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

nbatch = int(len(X_train)/M) # Number of batches in SGD

#####################################
# Defining the neural network
n_inputs, n_features = X_train.shape # rows and columns of input
n_hidden_neurons = Nh # Chose what gives best results
n_categories = np.shape(X_train)[1] # outputs



########################################
#loop over learn-rates and penalties


#setting up matrixt to store results
train_results = np.zeros((len(eta_vals), len(lmbd_vals)))
test_results = np.zeros((len(eta_vals), len(lmbd_vals)))

for i, eta in enumerate(eta_vals):
    print("Learn Rate: ",eta)
    
    for j, lmbd in enumerate(lmbd_vals):      
        
        # Reset weights and bias in the hidden layer
        Wh = np.random.randn(n_features, n_hidden_neurons)
        Bh = np.zeros(n_hidden_neurons) + 0.01
    
        # Resets weights and bias in the output layer
        Wo = np.random.randn(n_hidden_neurons, n_categories)
        Bo = np.zeros(n_categories) + 0.01
        # Reset weight gradient for test in While-loop
        dWo_t = np.random.randn(n_hidden_neurons, n_categories)
         

        # calculate gradient
        if use_sgd:
            Wo,Bo,Wh,Bh = SGD(X_train, Y_train, eta, lmbd, Wo, Bo, Wh, Bh)
            
        elif use_gdm:
            Wo,Bo,Wh,Bh = GDM(X,Y,eta,lmbd, Wo, Bo, Wh, Bh)
            
        else:
            # loop over epochs, through NN, 
            # stops when norm of gradient is smaller than lim
            epoch = 0
            while np.linalg.norm(dWo_t) > lim :
                epoch +=1
                gradients = list(backpropagation(X_train, Y_train))
                
                # regularization term gradients
                gradients[0] += lmbd * Wo
                gradients[2] += lmbd * Wh
                
                # Update weights with learn rate
                Wo -= eta * gradients[0]
                Bo -= eta * gradients[1]
                Wh -= eta * gradients[2]
                Bh -= eta * gradients[3]
                
                
                # Calculate gradient for whole Train dataset 
                # to use as a stop criteria 
                dWo_t, dBo_t, dWh_t, dBh_t  = backpropagation(X_train, Y_train)
                
                if epoch > itter_limit:
                    print("Reach itteration limit of:  ", itter_limit, "itterations") 
                    break
        
 
        # Select best values for plotting (Hard coded from previous run)
        if (eta == eta_best) and (lmbd == lmbd_best):
            Plot_XY(X,Y)
             
        #Store loss value
        #print(predict(X_train))
        #print(loss(predict(X_train),Y_train))
        train_results[i][j] = loss(predict(X_train),Y_train) 
        test_results[i][j] = loss(predict(X_test),Y_test)

#########################################
#Plot results
lmbd_labels= np.round(np.log10(lmbd_vals),2)
eta_labels= np.round(np.log10(eta_vals),2)

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_results, annot=True, 
            fmt='.3g', ax=ax, cmap="viridis",
            xticklabels = lmbd_labels, yticklabels = eta_labels)
ax.set_title("Train Loss, Itterations: " + str(itter_limit))
ax.set_ylabel(" log( $\eta$ )")
ax.set_xlabel("log( $\lambda$ )")
plt.show()
   
sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_results, annot=True,
            fmt='.3g', ax=ax, cmap="viridis",
            xticklabels = lmbd_labels, yticklabels = eta_labels)
ax.set_title("Test Loss, Itterations: " + str(itter_limit))
ax.set_ylabel(" log( $\eta$ )")
ax.set_xlabel("log( $\lambda$ )")
plt.show()



if neuro_loops:
    eta = eta_best
    lmbd = lmbd_best
    train_results = np.zeros(len(neuron_vals))
    test_results = np.zeros(len(neuron_vals)) 
    
    for k, neu in enumerate(neuron_vals):  
        print("Number of neurons: " ,neu)
        # Defining the neural network
        n_inputs, n_features = X_train.shape # rows and columns of input
        n_hidden_neurons = neu # Chose what gives best results
        n_categories = np.shape(X_train)[1] # outputs
                
        
        # Reset weights and bias in the hidden layer
        Wh = np.random.randn(n_features, n_hidden_neurons)
        Bh = np.zeros(n_hidden_neurons) + 0.01

        # Resets weights and bias in the output layer
        Wo = np.random.randn(n_hidden_neurons, n_categories)
        Bo = np.zeros(n_categories) + 0.01
        # Reset weight gradient for test in While-loop
        dWo_t = np.random.randn(n_hidden_neurons, n_categories)
         

        # calculate gradient
        if use_sgd:
            Wo,Bo,Wh,Bh = SGD(X_train, Y_train, eta, lmbd, Wo, Bo, Wh, Bh)
            
        elif use_gdm:
            Wo,Bo,Wh,Bh = GDM(X,Y,eta,lmbd, Wo, Bo, Wh, Bh)
            
        else:
            # loop over epochs, through NN, 
            # stops when norm of gradient is smaller than lim
            epoch = 0
            while np.linalg.norm(dWo_t) > lim :
                epoch +=1
                gradients = list(backpropagation(X_train, Y_train))
                
                # regularization term gradients
                gradients[0] += lmbd * Wo
                gradients[2] += lmbd * Wh
                
                # Update weights with learn rate
                Wo -= eta * gradients[0]
                Bo -= eta * gradients[1]
                Wh -= eta * gradients[2]
                Bh -= eta * gradients[3]
                
                
                # Calculate gradient for whole Train dataset 
                # to use as a stop criteria 
                dWo_t, dBo_t, dWh_t, dBh_t  = backpropagation(X_train, Y_train)
                
                if epoch > itter_limit:
                    print("Reach itteration limit of:  ", itter_limit, "itterations") 
                    break
            
            
        # Select best values for plotting (Hard coded from previous run)
        if (neu == Nh_best):
            Plot_XY(X,Y)
             
        #Store loss value
        #print(predict(X_train))
        #print(loss(predict(X_train),Y_train))
        train_results[k] = loss(predict(X_train),Y_train)  
        test_results[k] = loss(predict(X_test),Y_test) 
    
    plt.figure()
    plt.plot(neuron_vals,train_results,'o-', label = "Train")    
    plt.plot(neuron_vals,test_results,'o-', label = "Test")    
    plt.xticks(neuron_vals,neuron_vals)
    plt.xlabel('Number of neurons')
    plt.ylabel('Accuracy')
    plt.title("Test and Train of Accuracy vs Number of Neurons")
    plt.legend()
    plt.grid(True)
    plt.show()
    
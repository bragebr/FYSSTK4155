from math import exp, sqrt
import numpy.random as rnd
import jax.numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jax import grad, jit

# GRADIENT_DESCENT() TAKES A TARGET COST FUNCTION WHICH IS DIFFERENTIATED BY THE grad FUNCTION IN THE jax LIBRARY
# OPTIMIZER AS A STRING, BEING THE NAME OF THE OPTIMIZER TYPE (GDM, RMSprop, etc), INITIAL GUESS guess, LEARNING RATE lr
# AND MAX NO. OF ITERATIONS niter. 

def GRADIENT_DESCENT(TARGET_FUNC, OPTIMIZER, guess, lr, niter):

    errdiff = 0.0; step = 0; dP = 0.0 # DECLARATIONS
    G2 = 0.0 ; first_moment = 0.0 ; second_moment = 0.0

    maxiter = niter
    tol = 1e-4 # TRY IMPLEMENTING STOPPING CRITERION
    eta = lr # LEARNING RATE
    delta = 1e-8 # AVOID DIVISION BY ZERO IN ADAPTIVE OPTIMIZERS
    rho = 0.99 # RMSprop
    momentum = 0.3 #GDM
    beta1 = 0.9 #ADAM
    beta2 = 0.999 #ADAM

    theta_init = guess # INITIAL GUESS

    training_gradient = jit(grad(TARGET_FUNC))


    if OPTIMIZER == 'GDM':

        while step < maxiter:

            g = training_gradient(theta_init)


            update = g*eta + (momentum * dP)
            theta = theta_init - update
            errdiff = np.linalg.norm(theta) - np.linalg.norm(theta_init)

            theta_init = theta

            dP = update

            step += 1



    elif OPTIMIZER == 'AdaGrad':

        while step < maxiter:# or np.abs(errdiff) > tol:
            gradient = training_gradient(theta_init)

            G2 += np.multiply(gradient,gradient)



            update = np.multiply(eta/(np.sqrt(G2)+delta),gradient)
            theta = theta_init - update
            errdiff = np.linalg.norm(theta) - np.linalg.norm(theta_init)
            theta_init = theta
            step += 1


    elif OPTIMIZER == 'RMSprop':

        while step < maxiter: # or np.abs(errdiff) >= tol:
            gradient = training_gradient(theta_init)
            G2 += (rho*G2)+(1-rho)* np.multiply(gradient,gradient)

	        # Taking the diagonal only and inverting
            update = np.multiply(eta/(np.sqrt(G2) + delta), gradient)
	        # Hadamard product
            theta = theta_init - update
            errdiff = np.linalg.norm(theta) - np.linalg.norm(theta_init)
            theta_init = theta
            step += 1


    elif OPTIMIZER == 'ADAM':
        step = 1
        while step < maxiter: #or np.abs(errdiff) >= tol:
            gradient = training_gradient(theta_init)
            # Computing moments first
            first_moment = beta1*first_moment + (1-beta1)*gradient
            second_moment = beta2*second_moment+(1-beta2)*np.multiply(gradient,gradient)

            first_term = first_moment/(1.0-beta1**step)
            second_term = second_moment/(1.0-beta2**step)

	        # Scaling with rho the new and the previous results
            update = eta*first_term/(np.sqrt(second_term)+delta)
            theta = theta_init - update

            errdiff = np.linalg.norm(theta) - np.linalg.norm(theta_init)
            theta_init = theta
            step += 1


    return theta, step


def SGD(TARGET_FUNC, OPTIMIZER, guess, X, y, lr, n_epochs):

    errdiff = 0.0; step = 0; dP = 0.0 # DECLARATIONS

    n_epochs = n_epochs
    M = 50
    m = int(n_epochs/M)
    t0,t1 = M,n_epochs

    def learning_schedule(t):
        return t0/(t + t1)


    tol = 1e-8

    delta = 1e-8 # AVOID DIVISION BY ZERO
    momentum = 0.3 # GDM
    rho = 0.99 # RMSprop
    eta = lr

    theta = guess # INITIAL GUESS

    training_gradient = jit(grad(TARGET_FUNC,argnums=2))


    if OPTIMIZER == 'GDM':

        for epoch in range(n_epochs):

            for i in range(m):

                r_i = M * rnd.randint(m)
                xi = X[r_i:r_i + M]
                yi = y[r_i:r_i + M]

                g = (1/m) * training_gradient(xi,yi,theta)
                #eta = learning_schedule(epoch*m + i)
                update = eta * g + (momentum * dP)
                theta -= update
                update = dP
                #theta_init = theta

    elif OPTIMIZER == 'AdaGrad':

        G2 = 0.0
        for epoch in range(n_epochs):

            for i in range(m):

                r_i = M * rnd.randint(m)
                xi = X[r_i:r_i + M]
                yi = y[r_i:r_i + M]

                g = (1/m) * training_gradient(xi,yi,theta)
                #eta = learning_schedule(epoch*m + i)

                G2 += np.multiply(g,g)



                update = np.multiply(eta/(np.sqrt(G2) + delta),g)
                theta -= update
                #theta_init = theta

    elif OPTIMIZER == 'RMSprop':
        G2 = 0.0; dP = 0.0
        theta_interim = 0.0; nesterov = 0.5
        for epoch in range(n_epochs):

            for i in range(m):

                r_i = M * rnd.randint(m)
                xi = X[r_i:r_i + M]
                yi = y[r_i:r_i + M]

                theta_interim = theta + nesterov * dP

                g = (1/m) * training_gradient(xi,yi,theta_interim)
                #eta = learning_schedule(epoch*m + i)

                G2 += (rho*G2)+(1-rho)* np.multiply(g,g)

    	        # Taking the diagonal only and inverting
                update = nesterov * dP - np.multiply(eta/(np.sqrt(G2)+delta),g)
    	        # Hadamard product
                theta += update
                dP = update
                #theta_init = theta

    elif OPTIMIZER == 'ADAM':

        beta1 = 0.9
        beta2 = 0.999
        first_moment = 0.0; second_moment = 0.0
        step = 0
        for epoch in range(n_epochs):
            step += 1
            for i in range(m):

                r_i = M * rnd.randint(m)
                xi = X[r_i:r_i + M]
                yi = y[r_i:r_i + M]

                g = (1/m) * training_gradient(xi,yi,theta)
                #eta = learning_schedule(epoch*m + i)

                first_moment = beta1*first_moment + (1-beta1)*g
                second_moment = beta2*second_moment+(1-beta2)*np.multiply(g,g)

                first_term = first_moment/(1.0-beta1**step)
                second_term = second_moment/(1.0-beta2**step)

    	        # Scaling with rho the new and the previous results
                update = eta*first_term/(np.sqrt(second_term)+delta)

                theta -= update

                #errdiff = np.linalg.norm(theta) - np.linalg.norm(theta_init)
                #theta_init = theta

    return theta

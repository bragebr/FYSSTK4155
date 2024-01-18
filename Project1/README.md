# Project 1 FYS-STK4155 Ellen Rekaa, Brage Brevig and Ines Santandreu.

The code for Project 1 is segmented into different parts and will therefore be explained here to ensure seamless usage of our code. 

# mylibrary.py
This file contains function calls that are relevant for the project. This is everything from creating random xy data to pass to the Franke function
to read - write function to make data files and processing them. A function for cross-validation is NOT included here as we had issues figuring out 
a good way to implement it as a callable function. It remains a separate code to the rest of the project. 

# plot_proj1.py
This file contains plot calls for the Franke function fitting and everything related to it. It is a raw (dummy) code where the user will have to 
choose filenames for saving the plots in the code itself. Plotting the 3D surfaces will also have to be done manually by changing which surface to 
visualize by commenting in or out the different models within the plot segment. 

# plot_real_data.py
Same as above, just for the real data. 

# main.py
The main file contains a step by step solving of OLS, Ridge and LASSO on a chosen set of data. Again, this code is NOT user friendly and the user
will have to go into the code to comment in or out the data she will study and change filenames similarly. 

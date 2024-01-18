import matplotlib.pyplot as plt
import mylibrary as mlb
import numpy as np
from matplotlib.font_manager import FontProperties
from imageio import imread
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sklearn.linear_model as skl

"""
-----------------
font for plots
-----------------
"""

font = FontProperties()
font.set_family('monospace')
fontsize = 18
FILEPATH = "Project1/Data/"
FILEPATH_FIG = "Project1/Figures/"

RealRidgeParams = mlb.Read(FILEPATH+"RidgeDat_real.csv")
RealLASSOParams = mlb.Read(FILEPATH+"LASSODAT_real.csv")
RealOLSParams = mlb.Read(FILEPATH+"OLSDAT_real.csv")

# HANDLES
RHLH = ["Degrees", "MSE", "R2", "Lambdas", "Opts"]
OLSH = ["Degrees", "MSE (TRAIN)", "MSE (TEST)", "R2", "Opts"]
# PARAMETER DICTIONARIES
RP = {RHLH[j]:RealRidgeParams[j] for j in range(len(RealRidgeParams))}
LP = {RHLH[j]:RealLASSOParams[j] for j in range(len(RealLASSOParams))}

OLSP = {OLSH[j]:RealOLSParams[j] for j in range(len(RealOLSParams))}

# PLOT LOG10 VALUES FOR EASIER VISUALIZATION
MSEOLSTRAIN = [np.log10(x) for x in OLSP["MSE (TRAIN)"]]
MSEOLSTTEST = [np.log10(x) for x in OLSP["MSE (TEST)"]]

MSERIDGE = [np.log10(x) for x in RP["MSE"]]
MSELASSO = [np.log10(x) for x in LP["MSE"]]

"""
PLOT MSE AND R2 FOR OLS FOR TERRAIN MODEL
"""


fig, (ax1,ax2) = plt.subplots(2,1)

ax1.set_title("MSE and R2 Score Under OLS Fitting", fontproperties = font)
ax1.plot(OLSP["Degrees"], MSEOLSTRAIN, label = "MSE (TRAIN)", alpha = 0.3, linestyle = "--")
ax1.plot(OLSP["Degrees"], MSEOLSTTEST, label = "MSE (TEST)", color = 'black', linestyle = '-')
ax1.set_ylabel("MSE")
ax1.plot(OLSP["Opts"][0], OLSP["Opts"][1], 'ro', label = "Optimal Fitting",)
ax1.legend(loc = 'upper right'); ax1.grid()

ax2.plot(OLSP["Degrees"], OLSP["R2"], label = "R2 Score", color = "red")
ax2.plot(OLSP["Opts"][0], OLSP["Opts"][2], 'bo', label = "Optimal Fitting",)
ax2.set_ylabel("R2 Score",fontproperties=font)
ax2.set_xlabel("Fitting Degree",fontproperties=font)
ax2.legend(loc='lower right'); ax2.grid()
plt.savefig(FILEPATH_FIG+"OLS_MSE_AND_R2_DEG_real.png")
#plt.show()


"""
PLOT MSE AND R2 FOR RIDGE TERRAIN FITTING
"""

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.set_title(r"MSE and R2 Scores as a Function of the Fitting Degree $d$",
fontproperties = font)
ax1.plot(RP["Degrees"], MSERIDGE,
label = 'RIDGE', color = 'black')
ax1.plot(LP["Degrees"], MSELASSO,
label = 'LASSO', color = 'red', linestyle = '--')

ax1.plot(RP["Opts"][0], RP["MSE"][int(RP["Opts"][0])], 'ro',
label = "Optimal Degree: Ridge")
ax1.plot(LP["Opts"][0], LP["MSE"][int(LP["Opts"][0])], 'bo',
label = "Optimal Degree: LASSO")
ax1.set_ylabel("MSE", fontproperties = font)
ax1.grid(); ax1.legend(loc = 'upper right')

ax2.plot(RP["Degrees"],RP["R2"],
label = 'RIDGE', color = 'black')
ax2.plot(LP["Degrees"],LP["R2"],
label = 'LASSO', color = 'red', linestyle='--')
ax2.set_ylabel("R2 Score", fontproperties = font)
ax2.set_xlabel(r"Polynomial Degree, $d$", fontproperties = font)
ax2.grid(); ax2.legend(loc = 'lower right')

plt.savefig(FILEPATH_FIG+"RIDGEANDLASSO_MSE_AND_R2_real.png")


#plt.show()

"""
PLOT ORIGINAL TERRAIN AND ESTIMATED SURFACES + COLORMAPS
"""

terrain = imread(FILEPATH+'SRTM_data_Norway_1.tif') # read terrain file
z = terrain

if z.shape[0] > z.shape[1]:
    N = z.shape[1]
else: N = z.shape[0]

# DETERMINE SEGMENTATION SIZE
N = 150

# MAKE XY GRID
x = np.linspace(0,N,N)
y = np.linspace(1.5*N,2.5*N,N)
xx,yy = np.meshgrid(x,y)

# SEGMENT TERRAIN DATA
z = z[:len(x),:len(y)]

# NORMALIZE DATA
std = np.std(z)
z = (z - np.mean(z))/std

# RECREATE DESIGN MATRICES WITH OPTIMAL FITTING DEGREES
X_ols = mlb.create_X(x,y,int(OLSP["Opts"][0]))
X_ridge = mlb.create_X(x,y,int(RP["Opts"][0]))
X_lasso = mlb.create_X(x,y,int(LP["Opts"][0]))

# COMPUTE OPTIMAL PARAMETERS
XT_X = X_ridge.T @ X_ridge
I_mat = np.identity(np.size(X_ridge[0,:]))
b_ridge = np.linalg.inv(XT_X + RP["Opts"][1] * I_mat) @ X_ridge.T @ z

# COMPUTE BEST MODEL
z_pred = X_ridge @ b_ridge
zols_pred = X_ols @ np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ z
zLASSO = skl.Lasso(LP["Opts"][1], max_iter = 100).fit(X_lasso,z)

zLASSO = X_lasso @ zLASSO.coef_.T


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_title("Original Terrain and OLS Estimated Surface")
ax.set_xlabel("x (arcsec)"); ax.set_ylabel("y (arcsec)")
ax.set_zlabel("Elevation")
#ax.plot_surface(xx,yy,z_pred, cmap = 'summer')
ax.scatter(xx,yy,z, alpha = 0.01, color = 'turquoise')
ax.plot_surface(xx,yy,zols_pred)
#ax.plot_surface(xx,yy,zLASSO, cmap = 'summer')
plt.savefig(FILEPATH_FIG+"LASSOTERRAIN.png")
plt.show()

# MAKE COLORMAPS

fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
ax1.set_title("Original Terrain Data", fontproperties = font)
ax1.set_ylabel("Y (arcsec)"); ax1.set_xlabel("X (arcsec)")
im1 = ax1.imshow(z, interpolation='None')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax2 = fig.add_subplot(122)
ax2.set_title("Estimated Terrain Data Under OLS Regression",
fontproperties=font)
ax2.set_xlabel("X (arcsec)", fontproperties=font)
im2 = ax2.imshow(zols_pred, interpolation = 'lanczos')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical');
plt.savefig(FILEPATH_FIG+"imshowterrain_OLS_trial.png")
plt.show()

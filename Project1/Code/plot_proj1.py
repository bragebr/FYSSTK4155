import matplotlib.pyplot as plt
import mylibrary as mlb
import numpy as np
from matplotlib.font_manager import FontProperties
from imageio import imread
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# READ IN LISTS FOR PLOTTING
RidgeParams = mlb.Read(FILEPATH+"RidgeDat.csv")
LASSOParams = mlb.Read(FILEPATH+"LASSOdat.csv")
OLSParams = mlb.Read(FILEPATH+"OLSDat.csv")
BVDAT = mlb.Read(FILEPATH+"biasvar.csv")


# HANDLES
RHLH = ["Degrees", "MSE", "R2", "Lambdas", "Opts"]
OLSH = ["Degrees", "MSE (TRAIN)", "MSE (TEST)", "R2", "Opts"]
BVH = ["Error", "Bias", "Variance"]
# PARAMETER DICTIONARIES
RP = {RHLH[j]:RidgeParams[j] for j in range(len(RidgeParams))}
LP = {RHLH[j]:LASSOParams[j] for j in range(len(LASSOParams))}
OLSP = {OLSH[j]:OLSParams[j] for j in range(len(OLSParams))}
BVP = {BVH[j]:BVDAT[j] for j in range(len(BVDAT))}

"""
PLOT MSE AND R2 FOR OLS FOR FRANKE FITTING
"""


fig, (ax1,ax2) = plt.subplots(2,1)

ax1.set_title("MSE and R2 Score Under OLS Fitting", fontproperties = font)
ax1.plot(OLSP["Degrees"], OLSP['MSE (TRAIN)'], label = "MSE (TRAIN)", alpha = 0.3, linestyle = "--")
ax1.plot(OLSP["Degrees"], OLSP['MSE (TEST)'], label = "MSE (TEST)", color = 'black', linestyle = '-')
ax1.set_ylabel("MSE")
ax1.plot(OLSP["Opts"][0], OLSP["Opts"][1], 'ro', label = "Optimal Fitting",)
ax1.legend(loc = 'upper right'); ax1.grid()

ax2.plot(OLSP["Degrees"], OLSP["R2"], label = "R2 Score", color = "red")
ax2.plot(OLSP["Opts"][0], OLSP["Opts"][2], 'bo', label = "Optimal Fitting",)
ax2.set_ylabel("R2 Score",fontproperties=font)
ax2.set_xlabel("Fitting Degree",fontproperties=font)
ax2.legend(loc='lower right'); ax2.grid()
plt.savefig(FILEPATH_FIG+"OLS_MSE_AND_R2_DEG.png")
#plt.show()


"""
PLOT BIAS-VAR TRADE OFF FOR FRANKE FITTING
"""


plt.title("Bias - Variance Trade-Off With Increasing Model Complexity",
fontproperties = font)
plt.plot(np.arange(1,len(BVP["Error"])+1), BVP["Error"], label = "Error")
plt.plot(np.arange(1,len(BVP["Error"])+1), BVP["Bias"], label = "Bias")
plt.plot(np.arange(1,len(BVP["Error"])+1), BVP["Variance"], label = "Variance")
plt.xlabel("Fitting Degree", fontproperties=font)
plt.legend(loc="lower right"); plt.grid()
#plt.show()


"""
PLOT MSE AND R2 FOR RIDGE AND LASSO FOR FRANKE FITTING
"""

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.set_title(r"MSE and R2 Scores as a Function of the Fitting Degree $d$",
fontproperties = font)
ax1.plot(RP["Degrees"], RP["MSE"],
label = 'RIDGE', color = 'black')
ax1.plot(LP["Degrees"], LP["MSE"],
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
label = 'LASSO', color = 'red', linestyle = '--')
ax2.set_ylabel("R2 Score", fontproperties = font)
ax2.set_xlabel(r"Polynomial Degree, $d$", fontproperties = font)
ax2.grid(); ax2.legend(loc = 'lower right')

plt.savefig(FILEPATH_FIG+"RIDGEANDLASSO_MSE_AND_R2.png")


#plt.show()

"""
PLOT ORIGINAL DATA SURFACE AND APPROXIMATED SURFACE
"""

n = 100 # number of sample points (virtual data)

np.random.seed(2023)

x,y = mlb.xy_data(n) # virtual x and y values
xs,ys = np.meshgrid(x,y) # meshed net of x and y values
z = mlb.FrankeFunction(xs,ys,n) # virtual data for benchmarking the code

OLSX = mlb.create_X(x,y,int(OLSP["Opts"][0]))
OLSb = np.linalg.inv(OLSX.T @ OLSX) @ OLSX.T @ z

RIDGEX = mlb.create_X(x,y,int(RP["Opts"][0]))
IDMAT = np.identity(np.size(RIDGEX[0,:]))
RIDGEb = np.linalg.inv(RIDGEX.T @ RIDGEX + RP["Opts"][1] * IDMAT) @ RIDGEX.T @ z

# OPTIONAL COLORMAP FOR COMPARISON
fig = plt.figure(figsize=(13,5))
ax1 = fig.add_subplot(121)
ax1.set_title("Original Franke Surface", fontproperties = font)
ax1.set_ylabel("Y (arcsec)"); ax1.set_xlabel("X (arcsec)")
im1 = ax1.imshow(z, interpolation='None')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

ax2 = fig.add_subplot(122)
ax2.set_title("Estimated Franke Surface Under Ridge Regression",
fontproperties=font)
ax2.set_xlabel("X (arcsec)", fontproperties=font)
im2 = ax2.imshow(RIDGEX @ RIDGEb, interpolation = 'lanczos')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical');
plt.savefig(FILEPATH_FIG+"imshowFranke_Ridge.png")
plt.show()

"""
fig1 = plt.figure()
ax = fig1.gca(projection='3d')

ax.set_title("Surface Comparison of Original Data and OLS Estimation")
ax.scatter(xs,ys,z, alpha = 0.1, color = 'turquoise')
ax.plot_surface(xs,ys, OLSX @ OLSb, cmap = 'coolwarm')
#ax.plot_surface(xs,ys, RIDGEX @ RIDGEb, cmap = 'summer')
ax.set_ylabel("Y"); ax.set_xlabel("X"); ax.set_zlabel("Z")
plt.savefig("OLSFrankeSurface.png")
plt.show()
"""

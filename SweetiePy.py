
import pandas as pd
import numpy as np
import scipy.optimize as spo
import scipy.stats as sp
import math
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
      

def LoadData(water: str, sucrose: str) -> np.array:
    
    """Load data from excel or text file
    
    Input should be two separate files, one file containing timestamps of licks
    for water, the other file the timestamps for sucrose. Make sure to have
    the timestamps in seconds, with every lick on a separate line.
    
    The output are two vectors, vecWater and vecSucrose, containing those
    timestamps in a NumPy array.
    
    @author: Jeroen Verharen
    """
    
    # Determine datatype and extract data
    if water[-4:] == 'xlsx' or water[-3:] == 'xls':
        vecWater = np.array(pd.read_excel(water, header = None))[:,0]
        vecSucrose = np.array(pd.read_excel(sucrose, header = None))[:,0]
    elif water[-3:] == 'txt':
        vecWater = np.array(pd.read_csv(water, sep = " ", header = None))[:,0]
        vecSucrose = np.array(pd.read_csv(sucrose, sep = " ", header = None))[:,0]
    else: 
        raise ValueError("No valid data type detected")
    
    
    # Print total licks imported and output vectors
    print("\nWater licks imported:",len(vecWater))
    print("Sucrose licks imported:",len(vecSucrose))
    return vecWater, vecSucrose

    

def PreProcess(vecWater: np.array, vecSucrose: np.array, time_cutoff: float = 5) -> np.array:
    
    """Perform a micro-structure analysis of licking behavior
    
    Inputs:
        vecWater and vecSucrose: two numpy arrays, created using LoadData,
                                 containing licks for water and for sucrose. 
        'time_cutoff': numerical value indicating the cuttoff (in s) that
                       ends a licking bout / choice (5s recommended).
    
    Output is a matrix, matChoices, containing all the choices the animal made:
        Column 1: timestamp of first lick
        Column 2: timestamp of last lick
        Column 3: # of licks within choice
        Column 4: choice for water (0) or sucrose (1)
        (With every row being a separate choice)
    
    @author: Jeroen Verharen
    """
    
    # pre-allocate empty arrays
    matBoutsWater = np.array([])
    matBoutsSucrose = np.array([])
    
    # Analyze for water spout
    size = 0  # initiate licking bout size at 0
    for i in range(0, len(vecWater)):  # loop through all water licks
        
        if size == 0:
            start = vecWater[i]  # save timestamp of first lick of bout
        
        size = size + 1  # increase size of bout with a lick
        
        if i < len(vecWater)-1:
            if vecWater[i+1]-vecWater[i] > time_cutoff:  # end of bout
                stop = vecWater[i]
                
                if len(matBoutsWater) == 0:
                    matBoutsWater = np.array([start, stop, size, 0])
                else: matBoutsWater = np.vstack((matBoutsWater, np.array([start, stop, size, 0])))
                
                size = 0  # reset bout size
                
            elif vecWater[i+1]-vecWater[i] < time_cutoff and i == len(vecWater)-2:  # end of bout for last licks of session
                stop = vecWater[i+1]
                size = size + 1
                matBoutsWater = np.vstack((matBoutsWater, np.array([start, stop, size, 0])))
         
        elif vecWater[i] - vecWater[i-1] > 5 and i == len(vecWater)-1:  # if the last choice of a session is a single lick
            matBoutsWater = np.vstack((matBoutsWater, np.array([start, start, 1, 0])))
            
    # Analyze for sucrose spout   
    size = 0  # initiate licking bout size at 0
    for i in range(0, len(vecSucrose)):
        
        if size == 0:
            start = vecSucrose[i]  # save timestamp of first lick of bout
        
        size = size + 1  # increase size of bout with a lick
        
        if i < len(vecSucrose)-1:
            if vecSucrose[i+1]-vecSucrose[i] > time_cutoff:  # end of bout
                stop = vecSucrose[i]
                
                if len(matBoutsSucrose) == 0:
                    matBoutsSucrose = np.array([start, stop, size, 1])
                else: matBoutsSucrose = np.vstack((matBoutsSucrose, np.array([start, stop, size, 1])))
                
                size = 0  # reset bout size
                
            elif vecSucrose[i+1]-vecSucrose[i] < time_cutoff and i == len(vecSucrose)-2:  # end of bout for last licks of session
                stop = vecSucrose[i+1]
                size = size + 1
                
                matBoutsSucrose = np.vstack((matBoutsSucrose, np.array([start, stop, size, 1])))
        elif vecSucrose[i] - vecSucrose[i-1] > 5 and i == len(vecSucrose)-1: # if the last choice of a session is a single lick
            matBoutsSucrose = np.vstack((matBoutsSucrose, np.array([start, start, 1, 1])))
    


    # merge matrices
    matChoices = np.vstack((matBoutsWater, matBoutsSucrose))  # merge matrices
    matChoices = matChoices[np.argsort(matChoices[:,0]),:]  # sort them based on 'start' timestamp
    
    print("\nA total of "+str(np.size(matChoices, axis=0))+" choices have been detected, of which "+str('{0:.2f}'.format(100*np.sum(matChoices[:,3] == 1)/np.size(matChoices, axis=0)))+"% for sucrose.\n")
    
    return matChoices









def CreateFigure(vecWater, vecSucrose, matChoices):
    
    """This functions plots the licks and choices for water and sucrose
    over time.
    
    @author: Jeroen Verharen
    """
    
    #Pre-allocate empty matrices
    vecTimeRange = np.arange(0, matChoices[-1,1], 100)
    matTotalLicks = np.empty((len(vecTimeRange),3))
    
    # First, create time series with total licks and preference
    for i in range(0, len(vecTimeRange)):
        water = sum(vecWater<vecTimeRange[i])
        sucrose = sum(vecSucrose<vecTimeRange[i])
        if water+sucrose > 0:
            pref = 100*sucrose/(water+sucrose)
        else: pref = 50
        matTotalLicks[i,:] = [water, sucrose, pref]
    
    # Create figure layout
    gridsize = (7,2)
    plt.figure(figsize=(15,7))
    ax1 = plt.subplot2grid(gridsize, (0,0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(gridsize, (3,0), colspan=2, rowspan=2)  
    ax3 = plt.subplot2grid(gridsize, (6,0), colspan=2, rowspan=1)
    
    # In ax1, draw in licks
    ax1.scatter(vecWater, np.zeros(len(vecWater)), marker = '|')
    ax1.scatter(vecSucrose, np.ones(len(vecSucrose))/10, marker = '|')
    ax1.set_yticks([0, 0.1])
    ax1.set_yticklabels(['water', 'sucrose'])
    ax1.set_ylim([-0.02, 0.14])    
    ax1.set_xlim([0, matChoices[-1,1]])
    ax1.set_xticks(np.arange(0,matChoices[-1,1],3600))
    ax1.set_xticklabels(np.round(np.arange(0,matChoices[-1,1],3600)/3600))
    ax1.set_xlabel('Time (h)')
    
    
    # In ax1, draw rectangles for choices
    for i in range(0, len(matChoices)):
        if matChoices[i,3] == 0:
            rect = patches.Rectangle((matChoices[i,0], 0.01), width = matChoices[i,1]-matChoices[i,0], height = 0.01, edgecolor='none', facecolor=(31/255,119/255,180/255))
        elif matChoices[i,3] == 1:
            rect = patches.Rectangle((matChoices[i,0], 0.11), width = matChoices[i,1]-matChoices[i,0], height = 0.01, edgecolor='none', facecolor=(255/255,127/255,14/255))
        ax1.add_patch(rect)


    # In ax2, show time series analysis
    ax2.plot(matTotalLicks[:,[0,1]])
    ax2.set_ylabel('Total licks')
    ax2.set_xlabel('Time (h)')
    ax2b = ax2.twinx()
    ax2b.plot(matTotalLicks[:,2], color='k', linestyle='--')
    ax2b.set_ylabel('Sucrose pref (%)')
    ax2b.set_ylim([0, 100])
    ax2b.set_xlim([0, matChoices[-1,1]/100])
    ax2b.set_xticks(np.arange(0,matChoices[-1,1]/100,36))
    ax2b.set_xticklabels(np.round(np.arange(0,matChoices[-1,1]/100,36)/36))

    
    # In ax3, show choices over time with choice size
    ax3.bar(np.where(matChoices[:,3]==0)[0], matChoices[np.where(matChoices[:,3]==0)[0],2], color=(31/255,119/255,180/255))
    ax3.bar(np.where(matChoices[:,3]==1)[0], matChoices[np.where(matChoices[:,3]==1)[0],2], color=(255/255,127/255,14/255))
    ax3.set_xlabel('Choice #')
    ax3.set_ylabel('Licks')
    














def FitModel(matChoices: np.array, priors: bool = True) -> 'prints best-fit parameter values':
    
    """Perform model fitting procedure
    
    Inputs:
        matChoices: choices of animal for sucrose and water, generated using PreProcess
        priors: boolean indicating whether or not to regulate parameter estimates
                using prior distributions (default  is True)
    
    Outputs:
        Best-fit parameter estimates will be printed in the console
    
    @author: Jeroen Verharen
    """
    
    def GetLogLikelihood(params, matChoices, priors):
        
        """Function to return the log likelihood of the model given a set of 
        parameters [rho, alpha, eta], given an animal's choices
        @author: Jeroen Verharen
        """
        
        rho = params[0]
        alpha = params[1]
        eta = params[2]
        beta = 1  # softmax' inverse temperature, set at 1
        
        # initialize values
        ct_unchosen_water = 0
        ct_unchosen_sucrose = 0
        valW = 0
        valS = 0
        logP = 0
        
        # loop through choices
        try:
            for i in range(np.size(matChoices, axis=0)):
                
                if eta >= 0:  # attraction of unchosen option
                    Psuc = math.exp(beta*(valS+math.tanh(ct_unchosen_sucrose*eta)))/( math.exp(beta*(valW+math.tanh(ct_unchosen_water*eta))) + math.exp(beta*(valS+math.tanh(ct_unchosen_sucrose*eta))) )
                else:  # unchosen option discounted
                    Psuc = math.exp(beta*(valS-math.tanh(ct_unchosen_sucrose*-eta)*valS))/( math.exp(beta*(valW-math.tanh(ct_unchosen_water*-eta)*valW)) + math.exp(beta*(valS-math.tanh(ct_unchosen_sucrose*-eta)*valS)) )
    
                Pwat = 1-Psuc  # chance of choosing water is 1 - P_sucrose
                
                if matChoices[i,3] == 1: #if sucrose has been chosen
                    rpe = rho - valS  # compute reward prediction error
                    valS = valS + alpha*math.tanh(matChoices[i,2]/10)*rpe  # compute new value
                    
                    logP = logP + math.log(Psuc) # add to log likelihood
                    
                    ct_unchosen_water = ct_unchosen_water + 1
                    ct_unchosen_sucrose = 0
                    
                else:  # if water has been chosen
                    rpe = 1 - valW  # compute reward prediction error
                    valW = valW + alpha*math.tanh(matChoices[i,2]/10)*rpe  # compute new value
    
                    logP = logP + math.log(Pwat)  # add to log likelihood
                    
                    ct_unchosen_sucrose = ct_unchosen_sucrose + 1
                    ct_unchosen_water = 0
    
            if priors == True:
                return -1 * ( logP + 
                             math.log(sp.beta.pdf(params[0]/10, 1.3, 3)/np.max(sp.beta.pdf(np.arange(-10, 10, 0.01), 1.3, 3))) + 
                             math.log(sp.beta.pdf(params[1], 1.1, 5)/np.max(sp.beta.pdf(np.arange(0, 1, 0.001), 1.1, 5))) +  
                             math.log(sp.norm.pdf(params[2], 0, 0.2)/np.max(sp.norm.pdf(np.arange(-2, 2, 0.01), 0, 0.2))) )
            else:
                return -1 * logP
        except:
            return np.Inf
        
    print("Starting model fitting procedure...\n")
    warnings.filterwarnings("ignore") # suppress warnings from scipy.optimize
    
    # perform model fiting, use different starting points to prevent getting stuck in local minimum
    matMin = np.empty((0,4))
    for a in [1, 2, 5]:  # different starting values of rho
        for b in [0.01, 0.1, 0.2]:  # different starting values of alpha
            for c in [-0.1, 0, 0.1]:  # different starting values of eta
                
                # find local minimum
                res = spo.minimize(GetLogLikelihood, [a, b, c], args=(matChoices, priors))
                
                # collect all local minima in matrix
                matMin = np.vstack((matMin, [res.fun, res.x[0], res.x[1], res.x[2]]))
    
    # return absolute minimum
    minVal = matMin[np.argmin(matMin[:,0]),:]
    
    print("Model fitting successful.\n\nRHO (hedonia): "+
          str(minVal[1])+ 
          "\nALPHA (learning rate): "+
          str(minVal[2])+
          "\nETA (discount/attraction): "+
          str(minVal[3])+
          "\n\nLog likelihood of fit: "+
          str(-1*minVal[0]))
    

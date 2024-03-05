# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:21:11 2024

@author: Marta MurielElduayen
"""

from sklearn import preprocessing   
import numpy as np

import time
import datetime

import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


##############################################################################
"""
------------------------------------------------------------------------------
---------------------------- VECTORIAL FUNCTIONS -----------------------------
------------------------------------------------------------------------------
"""
##############################################################################
    
# Normalize data, raning from 0 to 1
def normalization(data):   
    data = np.array(data)  
    min_max_scaler = preprocessing.MinMaxScaler()
    data_norm = min_max_scaler.fit_transform(data)
    return data_norm

# Calculate the Euclidean distance between compared vectors
def getED(vec1,vec2):  
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # modified to allow a matrix as input
    ED = np.sqrt(np.sum(np.square(vec1-vec2), axis=1))  
    # reshape as a column vector
    return ED.reshape(ED.size,1)

# Calculate the cosine similarity between compared vectors
def getCosine(vec1,vec2):
    vec1 = np.array(vec1)
    vec1 = np.mat(vec1)
    vec2 = np.array(vec2)
    vec2 = np.mat(vec2)
    # modified to allow a matrix as input
    Cosine = (vec1*vec2.T)/(np.linalg.norm(vec1)*(np.linalg.norm(vec2, axis=1))) 
    # reshape as a column vector
    return Cosine.reshape(Cosine.size,1)

# Calculate the magnitude of a vector
def calculateMag(data):
    data = np.array(data)
    Mag = np.linalg.norm(data)
    if len(data) > len(newCase):
        # modified to return an array if the input is a matrix
        Mag = np.linalg.norm(data, axis = 1)
        # reshape as a column vector
        Mag = Mag.reshape(len(Mag), 1)
    return Mag

# Calcualte the triangular similarity measure between compared vectors
def getTSM(newCase, pastCases):     
    # TSM = ((getED(vec1,vec2)+abs(calculateMag(vec1)-calculateMag(vec2))))*(1-(getCosine(vec1,vec2)**2)+0.001)*calculateMag(vec1)*calculateMag(vec2)*0.25
    # sim = math.exp(-TSM)  
    magPC = calculateMag(pastCases)
    magNC = calculateMag(newCase)
    # modified to operate with a matrix
    TSM = np.multiply(((getED(newCase,pastCases)+abs(magNC-magPC))), np.multiply((1-np.square(getCosine(newCase,pastCases))+0.001),magNC*magPC)*0.25)
    # modified to allow for exp of an array
    sim = np.exp(-TSM)    
    return sim


##############################################################################
"""
------------------------------------------------------------------------------
----------------------------- MODIFIED FUNCTIONS -----------------------------
------------------------------------------------------------------------------
"""
##############################################################################


# Retrieve the most similar case from the pastCases
def getMaxTSM():     
    # Get the array of TSMs of the new case against the whole case base; 
    #   Operating directly with the matrix of past cases
    vecTSMs = getTSM (newCase, pastCases)
    # Remove possible nan values
    vecTSMs = np.nan_to_num(vecTSMs)
    # Return the index of the most similar past case
    return np.argmax(vecTSMs), np.max(vecTSMs)

# Get a matrix of 3 similar cases for each past case
def getSim () :
    global simMatrix
    tsim0 = time.time()
    # matrix to save similar cases
    simMatrix = []
    for caseID, case in enumerate(pastCases):
        # Get the array of TSM of each case against the whole case base; 
        #   Operating directly with the matrix of past cases
        vecTSMs = getTSM (case, pastCases)
        # Remove from the list the TSM of the current case (whose value will be 1)
        vecTSMs[caseID] = 0
        # Remove possible nan values
        vecTSMs = np.nan_to_num(vecTSMs)
        
        sim = []
        for i in range (3) :
            # get the index of the most similar
            simID = np.argmax(vecTSMs)
            # make its value in the array to be 0
            vecTSMs[simID] = 0
            # save the index
            sim.append(simID)
        
        # add the 3 most similar cases to the current case to the matrix
        simMatrix.append(sim)  
        
    print ("Association matrix creation time: %2.3f" %((time.time()-tsim0)))

# Calculate the difference between two arrays as the adittion of the difference in their features
def calDiff (vec1, vec2, w) :
    return np.sum(abs(vec1-vec2))

# Compare the solution of the adaptation case to the solution of its similar cases
def reviseSolNew (newCase, adaptCase) :
    # temp media, temp max, temp min, hum media, hum max, hum min, precipitacion, et0 y prec efect
    # directa = temp = 
    #           0,1,2,3,4,5
    # inversa = hum, precipitacion, et0, prec efectiva = 
    #           6,7,8,9,10,11,12,13,14,15,16,17
    dir_idx = [0,1,2,3,4,5]
    inv_idx = [6,7,8,9,10,11,12,13,14,15,16,17]
    
    dir_nc = np.array(newCase)[dir_idx].astype(float)
    inv_nc = np.array(newCase)[inv_idx].astype(float)
    
    dir_ac = np.array(pastCases[adaptCase])[dir_idx].astype(float)
    inv_ac = np.array(pastCases[adaptCase])[inv_idx].astype(float)
    
    
    # The diference between the new case and the adaptation case
    difAdaptDir = calDiff(dir_nc, dir_ac, [])
    difAdaptInv = calDiff(inv_nc, inv_ac, [])
    # The solution of the adaptation case
    solAdapt = sol[adaptCase]
    # if both cases are equal, then the solution to the new case is that of the adaptation case
    if np.mean([difAdaptDir, difAdaptInv]) == 0.0 :
        return solAdapt, difAdaptDir, difAdaptInv
    
    # The past cases similar to the adaptation case
    simCases = pastCases[simMatrix[adaptCase]]
    # The solution of those similar cases
    solSimCases = sol[simMatrix[adaptCase]]
    solSimCases = solSimCases - solAdapt
    #solSimCases = np.concatenate ((solSimCases, [solAdapt]))
    # The mean of the solution of the similar cases
    solSimCasesMean = np.mean(solSimCases)
    
    # The difference between the adaptation case and its similar cases
    difsDir = [] 
    difsInv = [] 
    for case in simCases :
        difsDir.append(calDiff(dir_ac, case[dir_idx], []))
        difsInv.append(calDiff(inv_ac, case[inv_idx], []))
    # The mean of the difference between the adaptation case and its similar cases
    difsMeanDir = np.mean(difsDir)
    difsMeanInv = np.mean(difsInv)

    # Calculate the solution of the new case using a rule of 3
    solNewDir = abs(((solSimCasesMean * difAdaptDir)/difsMeanDir) + solAdapt)
    solNewInv = abs(((solSimCasesMean * difsMeanInv)/difAdaptInv) + solAdapt)
    solsNew = [solNewDir, solNewInv]
    solNew = np.mean(solsNew)
    
    return solNew, difAdaptDir, difAdaptInv


##############################################################################
"""
------------------------------------------------------------------------------
------------------------------- MAIN FUNCTION --------------------------------
------------------------------------------------------------------------------
"""
##############################################################################

def cbr_algorithm (origNewCase) :
    global origPastCases
    tt = time.time()
    print ("\n", "----------------------------------------------------------------------","\n")
    print (datetime.datetime.strftime(datetime.datetime.today(), "%H:%M:%S %d/%m/%Y"),"\n")
    setup(origNewCase) 
    print ()
    
    # Calculate the most similar case to the new one
    t0 = time.time()
    print ("NEW CASE:", newCase,"\n", np.round(origNewCase), "\n")
    adaptCaseID, maxTSM = getMaxTSM()
    print ("PAST CASE WITH MAX TSM:", pastCases[adaptCaseID],"\n", origPastCases[adaptCaseID],"\n")
    print (" -- Time to get the max TSM: %2.3f\n" %(time.time()-t0))
    
    # Calculate the solution for the new case
    t1 = time.time()
    newCaseSol, difAdaptDir, difAdaptInv = reviseSolNew(newCase, adaptCaseID)
    print (" -- Time to get the solution for the new case: %2.3f\n" %(time.time()-t1))
    print ("REVISED SOLUTION:", newCaseSol, "\n") 
    
    # If the new case is distant enough from the past ones, we add it to the case base
    if maxTSM < 0.9 and np.mean([difAdaptDir, difAdaptInv]) > (0.2*newCase.size) :
        origNewCase = origNewCase.reshape(origNewCase.size)
        origNewCase = np.concatenate ((origNewCase, [newCaseSol]))
        print ("The new case is included in the case base as: ", origNewCase,"\n")
        origPastCases = np.concatenate ((origPastCases, [origNewCase]))        
    
    print ("TOTAL TIME: %2.3f\n" %(time.time()-tt))
    print ("----------------------------------------------------------------------")
    
    return newCaseSol

def load_kb () :
    global origPastCases
    # get the original matrix with the knowledge base
    origPastCases = np.loadtxt(open(processingMonth + ".csv", "rb"), delimiter=",", dtype=str)[:,1:].astype(float)

def update_kb () :
    # save the knowledge base matrix
    np.savetxt(processingMonth+".csv", np.asarray(origPastCases), delimiter=",", fmt='%s')

def setup (origNewCase) :
    global allCases, pastCases, newCase, sol, processingMonth
    
    currentMonth = datetime.datetime.strftime(datetime.datetime.today(), '%m') 
    try : 
        processingMonth
    except : 
        processingMonth = currentMonth
        load_kb ()
    else :
        if processingMonth != currentMonth :
            #update_kb ()
            processingMonth = currentMonth
            load_kb ()        
     
    # place in allCases the normalization of the pastCases (without the solutions) and the array of the new case
    allCases = normalization(np.concatenate((origPastCases[:, :-1], [origNewCase])))
    # extract the pastCases as all the rows except the last one (new case) 
    pastCases = allCases[:-1]
    # extract the new case as the last row
    newCase = allCases[-1]
    
    # extract the solution vector as the last column of all tha past cases without normalization
    sol = origPastCases[:,-1]
    
    # get the matrix of similar cases
    getSim()
    
        
##############################################################################

def calculate_water (daily_readings) : 
    return cbr_algorithm(daily_readings)
    
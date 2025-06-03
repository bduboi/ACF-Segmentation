############################################################################################################################################################
# Function file for the Segment ACF package
############################################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os
from scipy.signal import get_window
import pyfftw as fftw
from numpy.lib.stride_tricks import sliding_window_view
fftw.config.NUM_THREADS = 4  # or os.cpu_count()

## PACKAGE LIBRARY FOR COMBCC Class
#############################################################################################################################################################
#
#
#
# Contains :
# 1. getCombs : Generates the 4 overlapping comb functions used in the segment ACF calculation.
# 2. getQ_P : Computes the number of comb configurations Q and the number of segments per comb configuration P.
# 3. CalculateLags : Calculates the lags for a given pair of comb functions (i,j = 1 to 4).
# 4. getCentralLagArray : Generates the central lag array which contains all the lags for each pair. It is used for vectorized processing.
# 5. InitialiseLagList : Generates the list of lags for the segment ACF. This array contains all the oredered possible time lags for the CCF
# 6. combCounting : Computes the cross-count between two comb configurations.
# 7. getCorrection : Computes the correction factor for each comb configuration and central lag, based on a selection list. The selection is made inside the class CombCC.
# 8. getSelectionArrays : Computes the selection list and count list for each comb configuration.
# 9. getPhi_ : Computes the phi_k and zeta_k for the segment ACF.

# 10. CombCC : Class implementation for the CombCC pre-processing worder.

#############################################################################################################################################################

####
# GenerateCombs FUNCTION : Generates the 4 overlapping comb functions used in the segment ACF calculation.
####

def getCombs(ell,L):
    """
    Generates a comb function for segment ACF calculation.
    Args:
        L (int): Length of the segment L.
        ell (int): Length of the sub-segments ell.
        For the comb function to be valid, the following conditions must be met:
        - ell and L must be positive integers.
        - ell must be at least L//4. The reason for this is that we use 4 overlapping comb functions, so we need at least 4 sub-segments of ell to cover L.
        
    Returns:
        np.ndarray: A 2D array representing the comb function.
    """
    if ell <= 0 or L <= 0:
        raise ValueError("Both ell and L must be positive integers.")
    if ell > L:
        raise ValueError("ell must be less than or equal to L.")
    if ell % 2 != 0:
        raise ValueError("ell must be an even integer.")
    if L % 2 != 0:
        raise ValueError("L must be an even integer.")
    if L//ell < 4:
        raise ValueError("L must be at least 4 times ell to ensure enough sub-segments for the comb function.")
    
    comb = np.zeros((4, L))
    swin = np.zeros(ell)
    for j in range(ell):
        swin[j] = get_window('hann', ell)[j]  # Using a Hann window for tapering
    for i in range(4):
        if i <= 2: # the last comb function will have a tooth less than the other, so it needs to be handled separately.
            jmax = L // ell // 2
        else:
            jmax = L // ell // 2 - 1
        for j in range(jmax):
            comb[i][ell * j * 2 + i * (ell // 2):ell * (2 * j + 1) + i * (ell // 2)] = swin
            
            
    return comb

def getQ_P(npts,ell,L):
    """Computes the number of comb configurations Q and the number of segments per comb configuration P.
    Q is not exactly the number of semgments of L available in npts because we need an overlap of ell//2 to ensure that the combs are not too short.
    

    Args:
        npts (int): total number of points in the trace.
        L (int): length of the comb segments.
        ell (int): length of the sub-segments for the auto-correlation.

    Returns:
        int: Number of comb configurations Q.
        int: Number of segments per comb configuration P.
    """
    if L>npts:
        print('L is larger than npts')
        return None,None
    
    nb_full_comb= npts/L
    
    Q = nb_full_comb*(L + ell//2) //L
    
    P = (L//ell)*2 -1
    return int(Q),P

def CalculateLags(i, j, ell, L):
    """
    Calculates the lags for the segment ACF.
    Args:
        i (int): Index of the comb function.
        j (int): Index of the sub-segment.
        ell (int): Length of the sub-segments ell.
        L (int): Length of the segment L.
    Returns:
        np.ndarray: An array of lags for the segment ACF for the configuration specified by i and j (comb number i correlated with comb number j).
    """
    
    if ell <= 0 or L <= 0:
        raise ValueError("Both ell and L must be positive integers.")
    if L//ell < 4:
        raise ValueError("L must be at least 4 times ell to ensure enough sub-segments for the comb function.")

    
    
    if i <= 2:
        overlap_adjust = 0
    else:
        overlap_adjust = -1
    causal_central_lags = np.arange(i * ell // 2 - j * ell // 2, L + overlap_adjust * 2*ell, 2 * ell)
    anti_causal_lags = np.flip(-np.arange(j * ell // 2 - i * ell // 2, L + overlap_adjust * ell - ell, 2 * ell)[1:])
    return np.concatenate((anti_causal_lags, causal_central_lags))


def getCentralLagArray(ell,L):
        
    central_lag_ij_array = np.empty((4,4), dtype=object)
    for i in range(4):
        lags = CalculateLags(i, i, ell, L)
        central_lag_ij_array[i, i] = lags
        for j in range(i):

            lags = CalculateLags(i, j, ell, L)

            central_lag_ij_array[i, j] = lags
            
    return central_lag_ij_array
    
          
def InitialiseLagList(ell,L):
    central_lag_list = []
    for i in range(4):
        central_lags = CalculateLags(i, i, ell, L)
        for central_lag in central_lags:
            if central_lag not in central_lag_list:
                central_lag_list.append(central_lag)
                
                if -central_lag not in central_lag_list:
                    central_lag_list.append(-central_lag)
                    
        for j in range(i):
            central_lags = CalculateLags(i, j, ell, L)
            for central_lag in central_lags:
                if central_lag not in central_lag_list:
                    central_lag_list.append(central_lag)
                    
                    if -central_lag not in central_lag_list:
                        central_lag_list.append(-central_lag)
    return central_lag_list


def combCounting(comb_selection_list1, comb_selection_list2,central_lag_dict,count_dict,q,type='cross'):
    """For a given pair of station, computes the cross-count beteen the two combs.
    
    Args:
        comb_selection_list1 (np array): selection list for the first comb configuration.
        comb_selection_list2 (np array): selection list for the second comb configuration.
        central_lag_dict (dict): Dictionary containing the central lags for each comb configuration.
        central_lag_dict (dict): Dictionary containing the central lags for each comb configuration.
        count_dict (dict): Dictionary to store the counts for each comb configuration and central lag.
        q (int): Index of the comb configuration.
        type (str, optional): Type of counting ('cross' or 'auto'). Defaults to 'cross'.
    

    Returns:
        _type_: count_dict which is modified on the spot.
    """
    cross_count = correlate(comb_selection_list1, comb_selection_list2, 'full')
    for i,central_lag in enumerate(central_lag_dict):
        if type=='cross':
                
            count_dict[f'{q}_{central_lag}'].append( cross_count[i])
            count_dict[f'{q}_{-central_lag}'].append( cross_count[i])
            
        else:
            count_dict[f'{q}_{central_lag}'].append( cross_count[i])
            
    return count_dict



def getCorrection(Q,P,selection_list1,selection_list2,central_lag_array_ij,central_lag_list):
    
    selection_dict1 = {}
    selection_dict2 = {}
    
    perfect_selection_dict1 = {}
    perfect_selection_dict2 = {}
    
    count_dict={f'{comb}_{lag}': [] for comb in range(Q) for lag in central_lag_list}
    
    expected_count_dict={f'{comb}_{lag}': [] for comb in range(Q) for lag in central_lag_list}
    
    #### Compute the selection list and count list for each comb
    for q in range(Q):
        selection_list_comb1= selection_list1[q*P:(q+1)*P]
        selection_list_comb2= selection_list2[q*P:(q+1)*P]
        
        for i in range(4):
            selection_dict1[f'{q}_{i}'] = selection_list_comb1[i::4]
            selection_dict2[f'{q}_{i}'] = selection_list_comb2[i::4]
            
            
            perfect_selection_dict1[f'{q}_{i}'] = np.ones_like(selection_dict1[f'{q}_{i}'])
            perfect_selection_dict2[f'{q}_{i}'] = np.ones_like(selection_dict2[f'{q}_{i}'])

            central_lags= central_lag_array_ij[i, i]
            count_dict = combCounting(selection_dict1[f'{q}_{i}'], selection_dict2[f'{q}_{i}'], central_lags,count_dict,q,type='auto')
            expected_count_dict = combCounting(perfect_selection_dict1[f'{q}_{i}'], perfect_selection_dict2[f'{q}_{i}'], central_lags,expected_count_dict,q,type='auto')

            for j in range(i):
                central_lags= central_lag_array_ij[i, j]
                count_dict = combCounting(selection_dict1[f'{q}_{i}'], selection_dict2[f'{q}_{j}'], central_lags,count_dict,q,type='cross')
                
                expected_count_dict = combCounting(perfect_selection_dict1[f'{q}_{i}'], perfect_selection_dict2[f'{q}_{j}'], central_lags,expected_count_dict,q,type='cross')
    
    correction_dict = {f'{q}_{lag}': [] for q in range(Q) for lag in central_lag_list}
    final_correction_dict={lag: [] for lag in central_lag_list}
    
    for q in range(Q):
        for central_lag in central_lag_list:

            correction_dict[f'{q}_{central_lag}'] = np.array((count_dict[f'{q}_{central_lag}']))/np.array((expected_count_dict[f'{q}_{central_lag}']))
            correction_dict[f'{q}_{central_lag}']=np.mean(correction_dict[f'{q}_{central_lag}'])
            correction= correction_dict[f'{q}_{central_lag}']
            final_correction_dict[central_lag].append(correction)
        
            
    Correction = np.zeros(len(central_lag_list))
    for i,central_lag in enumerate(central_lag_list):
    
        correction= np.mean(final_correction_dict[central_lag]) 
        if correction==0:
            correction=1
            
            
        Correction[i] = correction
        
    return Correction

##############################################################################################################################################################
# SECTION 2 : DATA SELECTION AND PROCESSING
##############################################################################################################################################################

def getSelectionArrays(Q,P,selection_list):
    
    """
    Computes the selection list and count list for each comb configuration.
    Args:
        Q (int): Number of comb configurations.
        P (int): Number of segments per comb configuration.
        selection_list (list): List of selections for each segment.
        central_lag_dict (dict): Dictionary containing the central lags for each comb configuration.
    Returns:
        selection_array (np.ndarray): A 2D array of shape (Q, 4) containing the selection lists for each comb configuration.
        perfect_selection_array (np.ndarray): A 2D array of shape (Q, 4) containing the perfect selection lists for each comb configuration.
    Raises:
    """
    
    P_single = (P+1)//4
    selection_array = np.zeros((Q,3,P_single)) # Q, P_single, 4
    
    
    
    selection_array_i3 = np.zeros((Q,1,P_single-1)) # Q, P_single-1, 4
    
    
    
    perfect_selection_array_i3 = np.zeros((Q,1,P_single-1)) # Q, P_single-1, 4
    
    
    #### Compute the selection list and count list for each comb
    for q in range(Q):
        selection_list_comb= selection_list[q*P:(q+1)*P]
        
        for i in range(3):
            if i== 3:
                selection_array_i3[q] = selection_list_comb[i::4]
                
                perfect_selection_array_i3[q] = np.ones_like(selection_array_i3[q])                
                break 
            selection_array[q,i] = selection_list_comb[i::4]

            
    return selection_array,selection_array_i3

###############################################################################################################################################################
# FUNCITONS FOR THE EARTHQUAKE CATALOG PARSING
################################################################################################################################################################

import pandas as pd
from datetime import datetime, timedelta
import math
def parse_date(parts):
    """Convert [year, doy, hour, min, sec] to datetime."""
    year = int(parts[0])
    year += 2000 if year < 76 else 1900
    doy = int(parts[1])
    hour, minute, second = int(parts[2]), int(parts[3]), float(parts[4])
    return datetime(year, 1, 1) + timedelta(days=doy-1, hours=hour, minutes=minute, seconds=second)

def read_earthquake_data(file_path):
    """Read earthquake catalog and return list of (datetime, (lat, lon), seismic moment)."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            date_time = parse_date(parts[:5])
            location = (float(parts[6]), float(parts[7]))
            moment = float(parts[-1])
            data.append((date_time, location, moment))
    return data

def dyne_cm_to_Mw(dyne_cm,):
    # Convert dyne/cm to N/m
    M0 = dyne_cm * 1e-7
    # Calculate moment magnitude (Mw) using the formula
    Mw = (2/3) * (math.log10(M0) - 9.1) 
    return Mw
def earthquakes_between_dates(file_path, date1, date2, max_mag):
    """Return list of ISO strings of earthquake datetimes within date range and above magnitude."""
    date1 = pd.to_datetime(date1.datetime if hasattr(date1, 'datetime') else date1)
    date2 = pd.to_datetime(date2.datetime if hasattr(date2, 'datetime') else date2)
    quakes = read_earthquake_data(file_path)
    results = [
        quake[0].strftime("%Y-%m-%dT%H:%M:%S")
        for quake in quakes
        if date1 <= quake[0] <= date2 and dyne_cm_to_Mw(quake[-1]) >= max_mag
    ]
    return results

def convert_dates_to_seconds(dates, date1):
    """Convert a list of datetime strings to seconds since date1."""
    dates = pd.to_datetime(dates)
    return (dates - date1).total_seconds()

def find_segment_indices_t0(t0s, segment_times):
    """Vectorized: find closest inferior segment index for each t0."""
    t0s = np.atleast_1d(t0s)
    segment_times = np.asarray(segment_times)
    indices = np.searchsorted(segment_times, t0s, side='right') - 1
    indices = np.clip(indices, 0, len(segment_times)-1)  # prevent out of bounds
    return indices

def A5t(t, t0, A5t0s, frequency=1e-3, Q=190,epsilon = 1e-8):
    """Vectorized calculation of A5t."""
    t, t0 = np.asarray(t), np.asarray(t0)
    
    diff_max = -(Q * np.log(epsilon)) / (2 * np.pi * frequency)
    diff = t[:, None] - t0[None, :]
    
    A5t0s = np.broadcast_to(A5t0s, (diff.shape[0], diff.shape[1]))  # Ensure A5t0s has the same shape as diff
    mask = (diff >= 0) #& (diff <= diff_max)
    result = np.zeros_like(diff)
    result[mask] = A5t0s[mask] * np.exp(-(2 * np.pi * frequency * diff[mask] / Q))
    return result

###############################################################################################################################################################
## getPhi_batch FUNCTION : Core function to compute the CCF
################################################################################################################################################################

def getPhi_(Correction,central_lag_list, central_lag_ij_array,
                 fft_segments1,fft_segments2,
                 ell, L, batch_size=50):
    
    """Batch-wise computation of phi_k and zeta_k"""
    nLagBins = len(central_lag_list)
    nfft = fft_segments1.shape[-1]
    n_segments = fft_segments1.shape[0]
    zeta_separate = np.zeros((nLagBins, 2*L-1))  # (nLagBins, 2*L-1)

    lags_acf = np.arange(-L + 1, L)
    lag_to_idx = {lag: k for k, lag in enumerate(central_lag_list)}
    lag_indices = {lag: np.where(lags_acf == lag)[0][0] for lag in np.unique(lags_acf)}  # faster lookup
    window_offsets = np.arange(-ell, ell+1)
    
    # Initialise the FFTW plan for the inverse FFT
    init = fftw.empty_aligned((batch_size, nfft), dtype='complex64')  # Allocate memory for the FFTW plan
    init[:,:] = 0  # Initialize the array to zeros
    process = fftw.builders.ifft(init)  # Create the FFTW plan for inverse FFT

    len_segments = fft_segments1.shape[0]  # Number of segments
    init = fftw.empty_aligned((len_segments, nfft), dtype='complex64')
    process = fftw.builders.irfft(init, axis=-1)
    
            # === Auto terms (i,i) ===
    for i in range(4):
        auto_spec = fft_segments1[:, i, :] * np.conj(fft_segments2[:, i, :])
        #inverse_fft = np.fft.ifft(auto_spec, axis=-1).real
        init[:, :] = auto_spec  # Fill the FFTW plan with the auto spectrum
        inverse_fft = process(init)  # Apply the FFTW plan to compute the inverse FFT
        inverse_fft = inverse_fft.real  # Convert to real part
        inverse_fft = np.fft.fftshift(inverse_fft, axes=-1)

        central_lags = central_lag_ij_array[i, i]
        acf_indices = np.array([lag_indices[lag] for lag in central_lags])
        central_indices = np.array([lag_to_idx[lag] for lag in central_lags])

        # Precompute all slices (vectorized extract)
        slices = inverse_fft[:, acf_indices[:, None] + window_offsets[None, :]]  # shape: (N_segments, N_lags, ell*2+1)
        extracts = np.mean(slices, axis=0)  # shape: (N_lags, window_size)

        for k in range(len(central_lags)):
            temp = np.zeros(2*L-1, dtype=extracts.dtype)
            start = acf_indices[k] - ell
            end = acf_indices[k] + ell + 1
            temp[start:end] = extracts[k]
            zeta_separate[central_indices[k]] += temp

    
        for j in range(i):
            cross_spec = fft_segments1[:, i, :] * np.conj(fft_segments2[:, j, :])
            #inverse_fft = np.fft.ifft(cross_spec, axis=-1).real
            init[:, :] = cross_spec  # Fill the FFTW plan with the cross spectrum
            inverse_fft = process(init)  # Apply the FFTW plan to compute the inverse FFT
            inverse_fft = inverse_fft.real  # Convert to real part
            # Shift the zero frequency component to the center
            inverse_fft = np.fft.fftshift(inverse_fft, axes=-1)

            central_lags = central_lag_ij_array[i, j]
            #acf_indices = np.array([lag_indices[lag] for lag in central_lags])
            acf_indices = np.array([np.where(lags_acf == lag)[0][0] for lag in central_lags])
            central_indices = np.array([lag_to_idx[lag] for lag in central_lags])
            central_indices_flip = np.array([lag_to_idx[-lag] for lag in central_lags])
            #slices = inverse_fft[:, acf_indices[:, None] + window_offsets[None, :]]  # shape: (N_segments, N_lags, 2*ell+1)


            valid_mask = (acf_indices - ell >= 0) & (acf_indices + ell < inverse_fft.shape[1])
            acf_indices = acf_indices[valid_mask]
            central_indices = central_indices[valid_mask]
            central_indices_flip = central_indices_flip[valid_mask]

            combined_indices = acf_indices[:, None] + window_offsets[None, :]
            slices = inverse_fft[:, combined_indices]  # shape: (N_segments, N_lags, window_size)
            
            extracts = np.mean(slices, axis=0)  # shape: (N_lags, window_size)

            for k in range(len(acf_indices)):
                temp = np.zeros(2*L-1, dtype=extracts.dtype)

                start_in_ifft = acf_indices[k] - ell
                end_in_ifft   = acf_indices[k] + ell + 1

                start_in_temp = max(0, start_in_ifft)
                end_in_temp   = min(2*L-1, end_in_ifft)

                start_in_extract = start_in_temp - start_in_ifft
                end_in_extract   = start_in_extract + (end_in_temp - start_in_temp)

                # Now safely assign
                temp[start_in_temp:end_in_temp] = extracts[k, start_in_extract:end_in_extract]
                zeta_separate[central_indices[k]] += temp
                zeta_separate[central_indices_flip[k]] += np.flip(temp)
        
    

    # Final assembly and phi_k
    phi_k  = np.zeros((2*L-1))

    for lag in central_lag_list:
        index = lag_to_idx[lag]
        phi_k  += zeta_separate[index] / Correction[index]
        


    return phi_k



############################################################################################################################################
#### CLASS IMPLEMENTATION FOR THE COMBCC PREPROCESSING WORDER
############################################################################################################################################


class CombCC: # Shamelessly copied from Tahn-Son Pham's GCC implementation for the coda cross-correlation.
    '''
    Pre-Processing Worder.
    '''
    def __init__ (self,npts, L , ell, fbands,thresholds):
        
        self.npts = npts  # Total number of points in the trace
        self.fft_npts = 2*L-1 
        self.ram_fband = fbands
        self.thresholds = thresholds
        self.ell = ell
        self.L = L
        
        self.Q,self.P = getQ_P(self.npts, ell,L)  # Get the number of segments Q and P based on the trace length and segment lengths.
        
        self.npts_used = (L-ell//2)*self.Q
        self.nsubseg  = 2*(self.npts_used // self.ell)-1 
        
        
        ## prepare FFTW for real input        
        self.re_arr = fftw.empty_aligned((self.Q-1,4,self.fft_npts), dtype='float32') # need the -1 because the last comb in the data might not be full and thus lead to error.
        self.rfft = fftw.builders.rfft(self.re_arr)
        # prepare inverse FFT for real input
        self.cp_arr = fftw.empty_aligned(self.get_spec_npts(), dtype='complex64')
        self.irfft = fftw.builders.irfft(self.cp_arr)
        
    def get_temp_npts(self):
        return self.fft_npts
    
    def get_spec_npts(self):
        return (int(self.fft_npts / 2) + 1)
    
    def init_subsegment_fft(self):
        """
        Initialize the FFTW plan for sub-segment processing.
        This method sets up the FFTW plan for sub-segments of length `ell` and allocates the necessary memory.

        """
        self.re_arr_sub = fftw.empty_aligned((self.Q*self.P -1,self.ell), dtype='float32') # Here we use all the Qs
        self.rfft_sub = fftw.builders.rfft(self.re_arr_sub)
            
    ### MAIN FUNCTION FOR THE PROCESSING.
    
    
    def pre_proc(self,trace):
        self.dt = trace.stats.delta  # Sampling interval of the trace
        fftw.interfaces.cache.enable()  # cache FFT plans
        
        ell,Q,L = self.ell, self.Q-1, self.L  # Unpack the segment length and number of segments
        combs= getCombs(ell,L)  # Generate the comb functions for Q segments, each of length ell and total length L
        data = trace.data  
        sum_taper = np.sum(np.sum(combs,axis=0)**2)
        self.effective_sum = sum_taper  # Store the effective sum of the taper for normalization later
        
        discard_indexes = self.MakeSelectionList(trace)
        selection_list_full1 = np.ones((self.nsubseg))
        for i in range(len(selection_list_full1)):
            if i in discard_indexes:
                selection_list_full1[i] = 0
                
        selection_array,selection_array_i3= getSelectionArrays(self.Q-1,self.P,selection_list_full1)
        
        mute_mask1 = np.ones((Q, 4, L))
        mute_mask2 = np.copy(mute_mask1)  # Initialize mute masks for both traces

        npts_used = (L-ell//2)*(Q+1)

        for i in range(4):
            if i<3:
                selection_list1 = selection_array[:, i]
                len_ = selection_list1.shape[1]  # Length of the selection list for the current comb
            else:
                selection_list1 = selection_array_i3[:,0]
                len_ = selection_list1.shape[1]
            for j in range(len_):
                mute_mask1[:, i, i*ell//2 + 2*j*ell:i*ell//2 + 2*(j+1)*ell] *= selection_list1[:,j][:, np.newaxis]
            
        #### VECTORIZED OPERATION FOR THE SEGMENT MUTING ####
        comb_array1 = np.tile(combs, (Q, 1, 1))  # (Q, 4, L)
        comb_array2 = np.copy(comb_array1)  # Copy for the second trace
        
        comb_array1 *= mute_mask1  # now all Q combs are pre-masked
        comb_array2 *= mute_mask2  # now all Q combs are pre-masked for the second trace
        # Roll the data segments according to the shifts
        

        step=(L - ell//2) # Take the overlap into account (half sub-segment length to ensure a flat equivalent windowing taper)
        
        data_rolled = np.zeros((Q, 4, L))
        
        segments = sliding_window_view(data[:npts_used], window_shape=L)[::step].copy()
        
        for q in range(Q):
            data_rolled[q] = np.tile(segments[q], (4, 1))  # shape (Q, 4, L)

        ### Vectorized calculation of the fourrier transforms
        segments = comb_array1 * data_rolled  # shape (Q, 4, L)
    

        self.re_arr[:,:,:]=0
        self.re_arr[:,:,0:L] = segments
        fft_segments = self.rfft()  # shape (Q, 4, 2*l-1)

        return fft_segments,selection_list_full1
    
    def MakeSelectionList(self, trace,eQ_Catalog=False):
        
        L = self.L
        ell = self.ell
        self.init_subsegment_fft()
        
        npts = trace.stats.npts
        dt = trace.stats.delta
        
        Q = self.Q
        npts_used = (L - ell // 2) * Q
        step = ell // 2
        n_subseg = 2 * (npts_used // ell) - 1

        segments = sliding_window_view(trace.data[:npts_used], window_shape=ell)[::step].copy()

        taper_function = get_window('hann', ell)
        sum_taper = np.sum(taper_function ** 2)
        
        segments *= taper_function #Broadcasting the taper function to each segment
    
        
        self.re_arr[:,:] = 0
        self.re_arr_sub[:,0:ell] = segments
        fourier_segments1 = self.rfft_sub()
        

        freqs = np.fft.rfftfreq(ell, dt)
        psd_segments= (np.abs(fourier_segments1) ** 2) * 2 * dt / sum_taper
        
        

        # Get the freq masks
        freq_ranges = self.ram_fband
        freq_masks = [(freqs > fmin) & (freqs < fmax) for (fmin, fmax) in freq_ranges]

        # Vectorized mean PSD per band and segment
        mean_psds = np.array([
            psd_segments[:, mask].mean(axis=1)
            for mask in freq_masks
        ])  # shape (n_ranges, n_subseg)

        ##############################################################################################################################
        # CREATE THE SELECTION LIST
        ###############################################################################################################################
 
        discarded_segments_dict = {'All': []} # Initialize the 'All' key in the dictionary, which uniquely cumulates all the discarded segments (cannot append twice the same segment)

        
        for i in range(len(self.thresholds)):
            discarded_segments_dict[f'N_{i}'] = []
        
            

        ### FIRST PART : Discard based on the EQ catalog
        if eQ_Catalog:
            ts = trace.stats.starttime  # Start time of the trace
            te = trace.stats.endtime  # End time of the trace    
            segment_times_seconds = np.arange(n_subseg) * step * dt
            discarded_segments_dict['EQ'] = []  # Initialize the 'EQ' key in the dictionary
            earthquake_catalog = "/Volumes/2024_Dubois/Project_ERI/Ambient_noise_Nishida_1999/moment_loc_76_20"

            ts_datetime = ts.datetime if hasattr(ts, 'datetime') else ts  # Ensure ts is a datetime object
            te_datetime = te.datetime if hasattr(te, 'datetime') else te  # Ensure te is a datetime object
            t0s = earthquakes_between_dates(earthquake_catalog,date1=ts_datetime,date2=te_datetime,max_mag=5) # (n earthquakes between ts and te,)

            t0_seconds = convert_dates_to_seconds(t0s, date1=ts_datetime) # (nt0 : number of earthquake beyond magnitude threshold, shape unknown)

            segment_indices = find_segment_indices_t0(t0_seconds, segment_times_seconds) # (shape : (nt0,))
            t= np.asarray(segment_times_seconds)

            A5t0s = mean_psds[0,segment_indices] # (shape : (nt0,)). 0 stands for the first frequency band, which is the one used to compute the A5t0s.
            
            t0_seconds = np.asarray(t0_seconds)


            A5t_values = A5t(t, t0_seconds, A5t0s=A5t0s) # (shape : (n_subseg, nt0))
            n_discard_eq =0
            
            for eq_index, t0_sec in enumerate(t0_seconds):
                
                # Find the segment where t0 falls:
                t0_segment_index = find_segment_indices_t0(t0_sec, segment_times_seconds)[0]
                # Get PSD decay values for this earthquake across all segments:
                psd_decay = A5t_values[:, eq_index]
                
                
                # Consider only segments at or after t0_segment_index:
                for seg_idx in range(t0_segment_index, len(segment_times_seconds)):
                    if psd_decay[seg_idx] > 0.5e-19: # Threshold for discarding segments based on A5t decay
                        discarded_segments_dict['All'].append(seg_idx)
                        n_discard_eq+= 1
                    else:
                        # PSD below threshold — no need to check further segments for this earthquake
                        break
                    
        else :
            n_discard_eq=0
            
        
        n_discard_threshold = 0
        for i in range(len(self.thresholds)):
            mask_low  = mean_psds[i] <= self.thresholds[i][0]
            mask_high = mean_psds[i] >= self.thresholds[i][1]

            mask_discard = mask_high | mask_low

            discarded_idxs = np.where(mask_discard)[0].tolist()
            
            
            discarded_segments_dict[f'N_{i}'].extend(discarded_idxs)
            
            
            discarded_segments_dict['All'].extend(discarded_idxs)  # Ensure unique indices in the 'all' key
            
            
        n_discard_threshold1 = len(list(set(discarded_segments_dict['All']))) - n_discard_eq  # Unique indices in 'All' minus those already counted from EQ catalog
        
        
        print(f'Proportion of discarded segments for: {trace.stats.station}:', np.round((n_discard_eq + n_discard_threshold1) / len(mean_psds[0]),2)*100, '%')
        
        
        # Remove duplicates in the 'all' key
        discarded_segments_dict['All'] = list(set(discarded_segments_dict['All']))
        
        return discarded_segments_dict['All']
    
    
    
    def ProcessData(self, spec1,spec2,selection_list_full1,selection_list_full2,central_lag_array_ij,central_lag_list):
            """
            Preprocessing for a single station. Here are the following steps:
            1. Get the selection list segment and the correction list.                           # Size is P.
            2. Get the comb function and appply it to the trace (Inside GetPhi).                #Size is Q,4
            3. Get the spectrum of the comb functions and recover Phi (Inside GetPhi)           # Size is Q,4, 2*L-1    
            """
            
            #phi = getPhi_(Correction,central_lag_list, central_lag_ij_array,fft_segments1,fft_segments2, ell, L)
            
            
            Correction = getCorrection(self.Q-1,self.P,selection_list_full1,selection_list_full2,central_lag_array_ij,central_lag_list)
            Phi = getPhi_(Correction,central_lag_list, central_lag_array_ij,spec1,spec2, self.ell, self.L)

            Phi*= 2*self.dt / self.effective_sum  # Normalize by the effective sum of the taper
            
            return Phi
    

####PLOTS


def plotCombs(comb, ell, L, central_lags_dict):
    """
    Plots the comb functions and their cross-correlations.
    Args:
        comb (np.ndarray): A 2D array representing the comb function.
        ell (int): Length of the sub-segments ell.
        L (int): Length of the segment L.
        central_lags_dict (dict): A dictionary containing the central lags for each comb configuration.
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 7))
    time = np.arange(L)

    # First column plots
    # 1st row: Only comb[0]
    axes[0, 0].plot(time, comb[0],color='black',linewidth=1)
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('A single comb')
    axes[0, 0].set_yticks([])

    # xticks and labels for the first comb alone
    xticks = np.arange(ell//2, L, 2*ell)
    axes[0, 0].set_xticks(xticks)
    axes[0, 0].set_xticklabels([rf'$p_{{{i}}}^{{0}}$' for i in range(len(xticks))])

    # 2nd row: All combs and their sum
    colors=['tab:blue','tab:red','tab:green','tab:gray']
    for i in range(4):
        axes[0, 1].plot(time, comb[i], label=f'Comb {i}',linewidth=1,color=colors[i])
    axes[0, 1].plot(time, np.sum(comb, axis=0), label='Sum', color='black', linewidth=1, linestyle='--')
    axes[0, 1].legend()
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('All combs and their sum')
    axes[0, 1].set_yticks([])


    # xticks and labels for the plot with all combs
    xticks_comb0 = np.arange(ell//2, L, 2*ell)
    xticks_comb2 = np.arange(3*ell//2, L, 2*ell)

    # Combine the labels for comb[0] and comb[2]
    xticks = np.concatenate((xticks_comb0, xticks_comb2))
    xtick_labels = [rf'$p_{{{i}}}^{{0}}$' for i in range(len(xticks_comb0))] + \
                [rf'$p_{{{i}}}^{{2}}$' for i in range(len(xticks_comb2))]

    # Sort the xticks and labels by their positions
    xticks_sorted_indices = np.argsort(xticks)
    xticks = xticks[xticks_sorted_indices]
    xtick_labels = np.array(xtick_labels)[xticks_sorted_indices]

    axes[0, 1].set_xticks(xticks)
    axes[0, 1].set_xticklabels(xtick_labels)

    # 3rd row: Auto-correlation of comb[0] with lag lines
    auto_corr = correlate(comb[0], comb[0], mode='full')
    lags_auto_corr = np.arange(-L + 1, L)

    # xticks and labels for the auto-correlation plot
    lag_ticks = np.arange(-len(xticks_comb0) + 1, len(xticks_comb0), 1)
    xticks_auto = np.arange(-L + 2*ell, L, 2*ell)
    axes[1, 0].set_xticks(xticks_auto)
    axes[1, 0].set_xticklabels([f'{i}ℓ' for i in lag_ticks])

    # Second column plots: Cross-correlations
    cross_indices = [(0,0),(1, 0), (2, 0), (3, 0)]
    rows = [1,1,2,2]
    cols=[0,1,0,1]
    for f,(i, j) in enumerate(cross_indices):
        row_idx= rows[f]
        col_idx=cols[f]
        correlation = correlate(comb[i], comb[j], mode='full')
        lags_correlation = np.arange(-L + 1, L)


        axes[row_idx, col_idx].plot(lags_correlation, correlation, label=f'Correlation {i}{j}',color='black',linewidth=1)
        axes[row_idx, col_idx].plot(lags_correlation, np.flip(correlation), label=f'Correlation {j}{i}',color='tab:red',linewidth=1) if f!=0 else ''
        
        for lag in central_lags_dict[f'{i}{j}']:
            axes[row_idx, col_idx].axvline(lag, linestyle='--', color='black', alpha=1,linewidth=0.7)
            #axes[idx, 1].axvline(lag, linestyle='--', color='black', alpha=1, label=f'Lag line {i}{j}' if lag == central_lags_dict[f'{i}{j}'][0] else "",linewidth=0.7)
            axes[row_idx, col_idx].axvline(-lag, linestyle='--', color='black', alpha=1,linewidth=0.7)
            #axes[idx, 1].axvline(-lag, linestyle='--', color='black', alpha=1, label=f'Lag line {j}{i}' if lag == central_lags_dict[f'{i}{j}'][0] else "",linewidth=0.7)
        
        axes[row_idx, col_idx].set_yticks([])
        axes[row_idx, col_idx].legend()
        axes[row_idx, col_idx].set_ylabel('Correlation')
        axes[row_idx, col_idx].set_title(f'Cross-correlation between comb {i} and comb {j}')

        # xticks and labels for the cross-correlation plots
        xticks_cross = np.arange(-L + 2*ell, L, 2*ell)
        lag_ticks_cross = np.arange(-len(xticks_comb0) + 1, len(xticks_comb0), 1)
        axes[row_idx, col_idx].set_xticks(xticks_cross)
        axes[row_idx, col_idx].set_xticklabels([f'{i}ℓ' for i in lag_ticks_cross])


    # Common x-label for the entire figure
    for ax in axes[2, :]:
        ax.set_xlabel('Lag [s]')

    letters = ['(a)', '(c)', '(e)']
    for i,ax in enumerate(axes[:, 0]):
        ax.text(0.05, 1.15, letters[i], transform=ax.transAxes, ha='right', va='top', fontsize=16)
        ax.tick_params(labelsize=13)
    letters = ['(b)', '(d)', '(f)']    
    for i,ax in enumerate(axes[:, 1]):
        ax.text(0.05, 1.15, letters[i], transform=ax.transAxes, ha='right', va='top', fontsize=16)
        ax.tick_params(labelsize=13)



    plt.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    
    plt.savefig('Figures/CombFunctions.png')
    plt.show()
    
    



def getWelchACF(data, len2, big_window):
    """
    Calculate the Welch ACF using the given data and parameters.
    
    Args:
        data (np.ndarray): The input data signal.
        len2 (int): Length of the segment L.
        big_window (np.ndarray): The tapering window.
        
    Returns:
        np.ndarray: The calculated ACF.
    """
    welch_acf = np.zeros(len2)
    sample_per_segment = len2
    start_idx = 0
    psd_segments = []
    effective_sum = np.sum(big_window**2) 
    while start_idx + sample_per_segment <= len(data):
        end_idx = start_idx + sample_per_segment
        segment = data[start_idx:end_idx]
        segment = segment * big_window
        segment_pad = np.pad(segment, (0, len2-1))
        fft_segment = np.fft.fft(segment_pad)
        psd_segment = np.abs(fft_segment)**2
        psd_segments.append(psd_segment)
        start_idx += len2 // 2
    
    mean_psd = np.mean(psd_segments, axis=0) / effective_sum
    welch_acf = np.fft.ifft(mean_psd).real
    welch_acf = np.fft.fftshift(welch_acf)  # Shift the zero frequency component to the center
    return welch_acf

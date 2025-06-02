############################################################################################################################################################
# Function file for the Segment ACF package
############################################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os
from scipy.signal import get_window
import pyfftw

#############################################################################################################################################################
# SECTION 1 : COMB FUNCTIONS
#
#
#
# Contains :
# - GenerateCombs: Generates the 4 overlapping comb functions used in the segment ACF calculation.
# - CalculateLags: Calculates the center of the lag bins expected for the comb function.
# - PlotCombs: Plots the comb functions.
#############################################################################################################################################################

####
# GenerateCombs FUNCTION : Generates the 4 overlapping comb functions used in the segment ACF calculation.
####

def GenerateCombs(len1, len2):
    """
    Generates a comb function for segment ACF calculation.
    Args:
        len2 (int): Length of the segment L.
        len1 (int): Length of the sub-segments ell.
        For the comb function to be valid, the following conditions must be met:
        - len1 and len2 must be positive integers.
        - len1 must be at least len2//4. The reason for this is that we use 4 overlapping comb functions, so we need at least 4 sub-segments of len1 to cover len2.
        
    Returns:
        np.ndarray: A 2D array representing the comb function.
    """
    if len1 <= 0 or len2 <= 0:
        raise ValueError("Both len1 and len2 must be positive integers.")
    if len1 > len2:
        raise ValueError("len1 must be less than or equal to len2.")
    if len1 % 2 != 0:
        raise ValueError("len1 must be an even integer.")
    if len2 % 2 != 0:
        raise ValueError("len2 must be an even integer.")
    if len2//len1 < 4:
        raise ValueError("len2 must be at least 4 times len1 to ensure enough sub-segments for the comb function.")
    
    comb = np.zeros((4, len2))
    swin = np.zeros(len1)
    for j in range(len1):
        swin[j] = get_window('hann', len1)[j]  # Using a Hann window for tapering
    for i in range(4):
        if i <= 2: # the last comb function will have a tooth less than the other, so it needs to be handled separately.
            jmax = len2 // len1 // 2
        else:
            jmax = len2 // len1 // 2 - 1
        for j in range(jmax):
            comb[i][len1 * j * 2 + i * (len1 // 2):len1 * (2 * j + 1) + i * (len1 // 2)] = swin
    return comb


#####
# CalculateLags FUNCTION : Calculates the center of the lag bins expected for the comb function
######

def CalculateLags(i, j, len1, len2):
    """
    Calculates the lags for the segment ACF.
    Args:
        i (int): Index of the comb function.
        j (int): Index of the sub-segment.
        len1 (int): Length of the sub-segments ell.
        len2 (int): Length of the segment L.
    Returns:
        np.ndarray: An array of lags for the segment ACF for the configuration specified by i and j (comb number i correlated with comb number j).
    """
    
    if len1 <= 0 or len2 <= 0:
        raise ValueError("Both len1 and len2 must be positive integers.")
    if len2//len1 < 4:
        raise ValueError("len2 must be at least 4 times len1 to ensure enough sub-segments for the comb function.")

    
    
    if i <= 2:
        overlap_adjust = 0
    else:
        overlap_adjust = -1
    causal_central_lags = np.arange(i * len1 // 2 - j * len1 // 2, len2 + overlap_adjust * 2*len1, 2 * len1)
    anti_causal_lags = np.flip(-np.arange(j * len1 // 2 - i * len1 // 2, len2 + overlap_adjust * len1 - len1, 2 * len1)[1:])
    return np.concatenate((anti_causal_lags, causal_central_lags))

####
# getQ_P FUNCTION : Calculates the number of combs and segments for a given number of points.

def getQ_P(npts,len2,len1):
    if len2>npts:
        print('len2 is larger than npts')
        return None,None
    
    nb_full_comb= npts/len2
    
    Q = nb_full_comb*(len2 + len1//2) //len2
    
    P = (len2//len1)*2 -1
    return int(Q),P

#####
# combCounting FUNCTION : Counts the number of accepted segments for each comb configuration.
#####
def InitialiseLagList(len1,len2):
    central_lag_list = []
    for i in range(4):
        central_lags = CalculateLags(i, i, len1, len2)
        for central_lag in central_lags:
            if central_lag not in central_lag_list:
                central_lag_list.append(central_lag)
                
                if -central_lag not in central_lag_list:
                    central_lag_list.append(-central_lag)
                    
        for j in range(i):
            central_lags = CalculateLags(i, j, len1, len2)
            for central_lag in central_lags:
                if central_lag not in central_lag_list:
                    central_lag_list.append(central_lag)
                    
                    if -central_lag not in central_lag_list:
                        central_lag_list.append(-central_lag)
    return central_lag_list


######
# computeSelectionCount FUNCTION : Computes the selection list and count list for each comb configuration.
######
def getSelectionArray(Q,P,selection_list):
    
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
    perfect_selection_array = np.zeros((Q,3,P_single)) # Q, P_single, 4
    
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

            perfect_selection_array[q,i] = np.ones_like(selection_array[q,i])
            
    return selection_array,selection_array_i3



def combCounting(comb_selection_list1, comb_selection_list2,central_lag_dict,count_dict,q,type='cross'):
    cross_count = correlate(comb_selection_list1, comb_selection_list2, 'full')
    for i,central_lag in enumerate(central_lag_dict):
        if type=='cross':
                
            count_dict[f'{q}_{central_lag}'].append( cross_count[i])
            count_dict[f'{q}_{-central_lag}'].append( cross_count[i])
            
        else:
            count_dict[f'{q}_{central_lag}'].append( cross_count[i])
            
    return count_dict


def getCorrection(Q,P,selection_list,central_lag_array_ij,central_lag_list):
    selection_dict = {}
    perfect_selection_dict = {}
    
    count_dict={f'{comb}_{lag}': [] for comb in range(Q) for lag in central_lag_list}
    
    expected_count_dict={f'{comb}_{lag}': [] for comb in range(Q) for lag in central_lag_list}
    
    #### Compute the selection list and count list for each comb
    for q in range(Q):
        selection_list_comb= selection_list[q*P:(q+1)*P]
        
        for i in range(4):
            selection_dict[f'{q}_{i}'] = selection_list_comb[i::4]

            perfect_selection_dict[f'{q}_{i}'] = np.ones_like(selection_dict[f'{q}_{i}'])
            central_lags= central_lag_array_ij[i, i]
            count_dict = combCounting(selection_dict[f'{q}_{i}'], selection_dict[f'{q}_{i}'], central_lags,count_dict,q,type='auto')
            expected_count_dict = combCounting(perfect_selection_dict[f'{q}_{i}'], perfect_selection_dict[f'{q}_{i}'], central_lags,expected_count_dict,q,type='auto')

            for j in range(i):
                central_lags= central_lag_array_ij[i, j]
                count_dict = combCounting(selection_dict[f'{q}_{i}'], selection_dict[f'{q}_{j}'], central_lags,count_dict,q,type='cross')
                
                expected_count_dict = combCounting(perfect_selection_dict[f'{q}_{i}'], perfect_selection_dict[f'{q}_{j}'], central_lags,expected_count_dict,q,type='cross')
    
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


######
# PlotCombs FUNCTION : Plots the comb functions.
######


def plotCombs(comb, len1, len2, central_lags_dict):
    """
    Plots the comb functions and their cross-correlations.
    Args:
        comb (np.ndarray): A 2D array representing the comb function.
        len1 (int): Length of the sub-segments ell.
        len2 (int): Length of the segment L.
        central_lags_dict (dict): A dictionary containing the central lags for each comb configuration.
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 7))
    time = np.arange(len2)

    # First column plots
    # 1st row: Only comb[0]
    axes[0, 0].plot(time, comb[0],color='black',linewidth=1)
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('A single comb')
    axes[0, 0].set_yticks([])

    # xticks and labels for the first comb alone
    xticks = np.arange(len1//2, len2, 2*len1)
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
    xticks_comb0 = np.arange(len1//2, len2, 2*len1)
    xticks_comb2 = np.arange(3*len1//2, len2, 2*len1)

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
    lags_auto_corr = np.arange(-len2 + 1, len2)

    # xticks and labels for the auto-correlation plot
    lag_ticks = np.arange(-len(xticks_comb0) + 1, len(xticks_comb0), 1)
    xticks_auto = np.arange(-len2 + 2*len1, len2, 2*len1)
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
        lags_correlation = np.arange(-len2 + 1, len2)


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
        xticks_cross = np.arange(-len2 + 2*len1, len2, 2*len1)
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
    
    
    
def getCentralLagArray(len1,len2):
        
    central_lag_ij_array = np.empty((4,4), dtype=object)
    for i in range(4):
        lags = CalculateLags(i, i, len1, len2)
        central_lag_ij_array[i, i] = lags
        for j in range(i):

            lags = CalculateLags(i, j, len1, len2)

            central_lag_ij_array[i, j] = lags
            
    return central_lag_ij_array
    
          
def getPhi(Correction,central_lag_list, central_lag_ij_array,fft_segments, len1, len2):
    
    zeta_separate= np.zeros((len(central_lag_list), 2*len2-1))  # shape (nLagBins, 2*len2-1)
    lags_acf = np.arange(-len2 + 1, len2)  # lags for the ACF, from -L+1 to L-1
    lag_to_idx = {lag: k for k, lag in enumerate(central_lag_list)}
    
    for i in range(4):
        auto_spec = fft_segments[:,i,:] * np.conj(fft_segments[:,i,:])
        inverse_fft = np.fft.ifft(auto_spec, axis=-1).real
        inverse_fft = np.fft.fftshift(inverse_fft, axes=-1)  # center lags
        
        central_lags = central_lag_ij_array[i, i]
        for lag in central_lags:
            central_idx = lag_to_idx[lag]
            acf_index = np.where(lags_acf == lag)[0][0]


            extract = np.mean(inverse_fft[:, acf_index-len1 : acf_index+len1+1], axis=0)
            temp = np.zeros((2*len2-1))
            temp[acf_index-len1 : acf_index+len1+1] = extract
            zeta_separate[central_idx] += temp  # shape (nLagBins, 2*len2-1)
        

        for j in range(i):
            cross_spec = fft_segments[:,i,:] * np.conj(fft_segments[:,j,:])
            inverse_fft = np.fft.ifft(cross_spec, axis=-1).real
            inverse_fft = np.fft.fftshift(inverse_fft, axes=-1)  # center lags
        
            

            central_lags = central_lag_ij_array[i, j]
            for lag in central_lags:
                central_idx = lag_to_idx[lag]
                acf_index = np.where(lags_acf == lag)[0][0]

                #print(inverse_fft[:, central_idx-len1 : central_idx+len1+1].shape)
                extract = np.mean(inverse_fft[:, acf_index-len1 : acf_index+len1+1], axis=0)
                temp = np.zeros((2*len2-1))
                temp[acf_index-len1 : acf_index+len1+1] = extract
                zeta_separate[central_idx] += temp
                
                #if i != j:
                central_idx_flip = lag_to_idx[-lag]
                extracts_flipped = np.flip(temp)
                zeta_separate[central_idx_flip] += extracts_flipped
            

    ## SUM ACF IS NOW SHAPE (n_terms, 2*len2-1 ) --> Assemble the final ACF for each central lag
    zeta_k = np.zeros((2*len2-1))  # Initialize the biased ACF
    phi_k = np.zeros((2*len2-1))  # Initialize the unbiased ACF
    
    for lag in central_lag_list:
        index = lag_to_idx[lag]
        zeta_k += zeta_separate[index]
        phi_k += zeta_separate[index] / Correction[index]
    
    
    
    return phi_k, zeta_k


def getPhi_batch(Correction, central_lag_list, central_lag_ij_array, fft_segments, len1, len2, batch_size=50):
    """Batch-wise computation of phi_k and zeta_k"""
    nLagBins = len(central_lag_list)
    nfft = fft_segments.shape[-1]
    n_segments = fft_segments.shape[0]
    zeta_separate = np.zeros((nLagBins, 2*len2-1))  # (nLagBins, 2*len2-1)

    lags_acf = np.arange(-len2 + 1, len2)
    lag_to_idx = {lag: k for k, lag in enumerate(central_lag_list)}

    # Loop over batches
    norm_factor= 0 
    for i_start in range(0, n_segments, batch_size):
        i_end = min(i_start + batch_size, n_segments)
        batch_fft = fft_segments[i_start:i_end]  # (batch_size, 4, nfft)

        for i in range(4):
            auto_spec = batch_fft[:, i, :] * np.conj(batch_fft[:, i, :])
            inverse_fft = np.fft.ifft(auto_spec, axis=-1).real
            inverse_fft = np.fft.fftshift(inverse_fft, axes=-1)

            central_lags = central_lag_ij_array[i, i]
            for lag in central_lags:
                central_idx = lag_to_idx[lag]
                acf_index = np.where(lags_acf == lag)[0][0]
                extract = np.mean(inverse_fft[:, acf_index-len1 : acf_index+len1+1], axis=0)
                temp = np.zeros((2*len2-1))
                temp[acf_index-len1 : acf_index+len1+1] = extract
                zeta_separate[central_idx] += temp

            for j in range(i):
                cross_spec = batch_fft[:, i, :] * np.conj(batch_fft[:, j, :])
                inverse_fft = np.fft.ifft(cross_spec, axis=-1).real
                inverse_fft = np.fft.fftshift(inverse_fft, axes=-1)

                central_lags = central_lag_ij_array[i, j]
                for lag in central_lags:
                    central_idx = lag_to_idx[lag]
                    acf_index = np.where(lags_acf == lag)[0][0]
                    extract = np.mean(inverse_fft[:, acf_index-len1 : acf_index+len1+1], axis=0)
                    temp = np.zeros((2*len2-1))
                    temp[acf_index-len1 : acf_index+len1+1] = extract
                    zeta_separate[central_idx] += temp

                    # add flipped version for (j, i)
                    central_idx_flip = lag_to_idx[-lag]
                    zeta_separate[central_idx_flip] += np.flip(temp)
        norm_factor+=1

    # Final assembly of zeta_k and phi_k
    zeta_k = np.zeros((2*len2-1))
    phi_k  = np.zeros((2*len2-1))

    for lag in central_lag_list:
        index = lag_to_idx[lag]
        zeta_k += zeta_separate[index]
        phi_k  += zeta_separate[index] / Correction[index]
        
   
    ## Normalize phi_k and zeta_k by the number of batches
    phi_k /= norm_factor
    zeta_k /= norm_factor

    return phi_k, zeta_k


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


## WAS REPLACED BY THE sliding_window_view FUNCTION WHICH IS FEW TIMES FASTER BECAUSE IT USES LESS MEMORY
def getFourrierSubSegmentsOLD(st, L,ell,no_pad=True):
    """
    Function to compute the Fourier transform of each sub-segment of length ell for the data selection.
    The data selection is independent of the comb functions, but first we must evaluate the range of the data that we will use, because the comb functions are defined on the whole data set.
    Since the comb function need an offset of ell//2, we need to ensure that the data is long enough to accommodate this offset.
    
    Parameters:
    st : obspy.Stream
        The stream object containing the waveform data.
    L : int
        The length of the data segment to be used for Fourier transform.
    ell : int
        The length of each sub-segment for Fourier transform.
    no_pad : bool, optional
        If True, no padding is applied to the segments before Fourier transform. If False, padding is applied. Default is True.
        Removing padding will result in a less accuracy in the FFT but will be around 10 times faster.
    Returns:
    fourier_segments : np.ndarray
        An array containing the Fourier transforms of each sub-segment.
    """

    npts= st[0].stats.npts  # Number of points in the trace
    dt = st[0].stats.delta  # Sampling interval
    

    #We have to take the following steps because the comb function do not overlap perfectly and hence 
    # The number of segments Q is not exactly npts/L but slightly larger.
    Q,P = getQ_P(npts, L, ell)  # Get the number of segments Q and P based on the trace length and segment lengths.
    
    npts_used = (L-ell//2)*Q
  
    
    n_subseg = 2*(npts_used // ell)-1  # Number of sub-segments in the used data
    
    print(f"Total numbeer of points : {npts}, Points used: {npts_used}")
    print('Number of sub-segment in the used data:', n_subseg, 
          '\nNumber of sub-segments in the whole data:', 2*(npts//ell)-1,)
    
    
    roll_shifts = -np.arange(n_subseg)*(ell//2) # Take the overlap into account (half sub-segment length to ensure a flat equivalent windowing taper)

    data = st[0].data[:npts_used]  # Extract the data from the first trace in the stream. Truncate the data to the number that can effectively be achieved by the combs.
    data_rolled = np.zeros((n_subseg, ell)) # Initialize an array to hold the all the p sub-segments 
    
    for p in range(P):
        data_rolled[p] = np.roll(data, roll_shifts[p], axis=-1)[:ell]  # shape (P, ell)
            
    ### Vectorized calculation of the fourrier transforms
    taper_function = get_window('hann', ell)  # Get a Hann window for tapering the data
    sum_taper = np.sum(taper_function**2)  # Sum of the taper function values
    segments = taper_function * data_rolled  # shape 
    
    if no_pad:
        fourier_segments = np.fft.rfft(segments, axis=-1)  # Compute the Fourier transform of each sub-segment    
    else:
        fourier_segments = np.fft.rfft(np.pad(segments, ((0,0),(0,ell-1))), axis=-1)
    psd_segments = (np.abs(fourier_segments)**2)*2*dt   /   (ell * sum_taper)  # Compute the Power Spectral Density (PSD) of each sub-segment. The factor of 2 accounts for the two-sided spectrum
    
    freqs= np.fft.rfftfreq(ell, dt)  # Get the frequencies corresponding to the Fourier transform
    return psd_segments,freqs




#############
# FUNCTIONS FOR THE DATA SELECTION
import pandas as pd
from datetime import datetime, timedelta
import math
from numpy.lib.stride_tricks import sliding_window_view

def getFourierSubSegments(trace, L,ell, no_pad=True):
    npts = trace.stats.npts
    dt = trace.stats.delta
    starttime = trace.stats.starttime

    Q,P = getQ_P(npts, L, ell)  # Get the number of segments Q and P based on the trace length and segment lengths.
    
    npts_used = (L-ell//2)*Q
    step = ell // 2  # Step size for sliding window view, half the segment length to ensure overlap
    
    n_subseg = 2*(npts_used // ell)-1  # Number of sub-segments in the used data

    print(f"Total number of points : {npts}, Points used: {npts_used}")
    print('Number of sub-segment in the used data:', n_subseg, 
          '\nNumber of sub-segments in the whole data:', 2*(npts//ell)-1,)
    
    
    # Create sliding window view
    segments = sliding_window_view(trace.data[:npts_used], window_shape=ell)[::step].copy()

    # Apply taper
    taper_function = get_window('hann', ell)
    sum_taper = np.sum(taper_function**2)
    segments *= taper_function

    # Fourier transform
    if no_pad:
        fourier_segments = np.fft.rfft(segments, axis=-1)
        freqs = np.fft.rfftfreq(ell, dt)
    else:
        fourier_segments = np.fft.rfft(np.pad(segments, ((0,0),(0,ell-1))), axis=-1)
        freqs = np.fft.rfftfreq(2*ell-1, dt)

    # PSD calculation
    psd_segments = (np.abs(fourier_segments)**2) * 2 * dt / ( sum_taper)

    # Frequencies
    

    # Vectorized computation of segment start times
    segment_times_seconds = np.arange(n_subseg) * step * dt
    #segment_times = [starttime + t for t in segment_times_seconds]

    return psd_segments, freqs, segment_times_seconds

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


def getSelectionListFull(ts,te,segment_times_seconds,mean_psds, thresholds,eqCatalog=True, max_mag=4.5):
    """Get selection list based on mean PSDs and thresholds
    args :
        ts : obspy.UTCDateTime or datetime.datetime
            Start time of the selection period.
        te : obspy.UTCDateTime or datetime.datetime
            End time of the selection period.
        segment_times_seconds : np.ndarray
            Array of segment start times in seconds.
        mean_psds : np.ndarray
            Mean Power Spectral Densities for each frequency band.
        thresholds : list of tuples
            List of (low, high) thresholds for each frequency band.
        eqCatalog : bool, optional
            Whether to use earthquake catalog for additional filtering. Default is True.
        max_mag : float, optional
            Maximum magnitude threshold for earthquakes. Default is 4.5.
    returns :
        discarded_segments_dict : dict
            Dictionary containing indices of discarded segments based on thresholds and earthquake catalog.
    """
        
    # Initialize the dictionary to follow the discarded segments. Contain the indices of the discarded segments.

    discarded_segments_dict = {'All': []} # Initialize the 'All' key in the dictionary, which uniquely cumulates all the discarded segments (cannot append twice the same segment)
    
    for i in range(len(thresholds)):
        discarded_segments_dict[f'N_{i}'] = []
        
        
    ### FIRST PART : Discard based on the EQ catalog
    if eqCatalog:
        discarded_segments_dict['EQ'] = []  # Initialize the 'EQ' key in the dictionary
        earthquake_catalog = "/Volumes/2024_Dubois/Project_ERI/Ambient_noise_Nishida_1999/moment_loc_76_20"

        ts_datetime = ts.datetime if hasattr(ts, 'datetime') else ts  # Ensure ts is a datetime object
        te_datetime = te.datetime if hasattr(te, 'datetime') else te  # Ensure te is a datetime object
        t0s = earthquakes_between_dates(earthquake_catalog,date1=ts_datetime,date2=te_datetime,max_mag=max_mag) # (n earthquakes between ts and te,)

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
                    n_discard_eq += 1
                else:
                    # PSD below threshold — no need to check further segments for this earthquake
                    break
                
    else :
        n_discard_eq=0
        
    
    n_discard_threshold = 0
    for i in range(len(thresholds)):
        mask_low  = mean_psds[i] <= thresholds[i][0]
        mask_high = mean_psds[i] >= thresholds[i][1]

        mask_discard = mask_high | mask_low

        discarded_idxs = np.where(mask_discard)[0].tolist()
        discarded_segments_dict[f'N_{i}'].extend(discarded_idxs)
        discarded_segments_dict['All'].extend(discarded_idxs)  # Ensure unique indices in the 'all' key
        
    n_discard_threshold = len(list(set(discarded_segments_dict['All']))) - n_discard_eq  # Unique indices in 'All' minus those already counted from EQ catalog
    print(f"Number of sub-segments discarded based on earthquake catalog: {n_discard_eq}")
    print(f"Number of sub-segments discarded based on thresholds: {n_discard_threshold}")
    print(f"Total number of sub-segments discarded: {n_discard_eq + n_discard_threshold}")
    print('Proportion of discarded segments:', np.round((n_discard_eq + n_discard_threshold) / len(segment_times_seconds),2)*100, '%')
    # Remove duplicates in the 'all' key
    discarded_segments_dict['All'] = list(set(discarded_segments_dict['All']))
    
    
    return discarded_segments_dict['All']


def ProcessData(data,Q,ell,L,selection_array,selection_array_i3,Correction,central_lag_list,central_lag_ij_array):
    pyfftw.interfaces.cache.enable()  # cache FFT plans
    combs= GenerateCombs(ell,L)  # Generate the comb functions for Q segments, each of length ell and total length L
    
    
    mute_mask = np.ones((Q, 4, L))

    npts_used = (L-ell//2)*(Q+1)

    for i in range(4):
        if i<3:
            selection_list = selection_array[:, i]
            len_ = selection_list.shape[1]  # Length of the selection list for the current comb
        else:
            selection_list = selection_array_i3[:,0]
            len_ = selection_list.shape[1]
        for j in range(len_):
            mute_mask[:, i, i*ell//2 + 2*j*ell:i*ell//2 + 2*(j+1)*ell] *= selection_list[:,j][:, np.newaxis]
        
    #### VECTORIZED OPERATION FOR THE SEGMENT MUTING ####
    comb_array = np.tile(combs, (Q, 1, 1))  # (Q, 4, L)
    comb_array *= mute_mask  # now all Q combs are pre-masked

    # Roll the data segments according to the shifts

    step=(L - ell//2) # Take the overlap into account (half sub-segment length to ensure a flat equivalent windowing taper)
    data_rolled = np.zeros((Q, 4, L))
    segments = sliding_window_view(data[:npts_used], window_shape=L)[::step].copy()
    for q in range(Q):
        data_rolled[q] = np.tile(segments[q], (4, 1))  # shape (Q, 4, L)

    ### Vectorized calculation of the fourrier transforms
    segments = comb_array * data_rolled  # shape (Q, 4, L)

    fft_segments = pyfftw.interfaces.numpy_fft.fft(np.pad(segments, ((0,0),(0,0),(0,L-1))), axis=-1)
    phi,zeta_k = getPhi_batch(Correction,central_lag_list, central_lag_ij_array,fft_segments, ell, L,batch_size=100)
    
    return phi,zeta_k
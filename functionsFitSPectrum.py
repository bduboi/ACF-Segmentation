# This file contains the functions used to fit the observed spectrum with the synthetic spectrum
# and to optimize the PSD of the radial pressure and shear traction sources.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import multiprocessing
from matplotlib.gridspec import GridSpec
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import itertools
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings


warnings.filterwarnings("ignore")

def GetSpectrum1(args, params, d=10, R=6371 * 1e3):
    f0 = 1e-3
    alpha,Beta,len2, freqs, freq_mask, psd_stacked = params
    ref_trac = (freqs / f0) ** (-2.3) * (2 * 1e9 / (4 * np.pi * R ** 2))
    
    freq_reduced = freqs[(freqs > 0.91 * 1e-3) & (freqs < 5 * 1e-3)]
    freq1= freqs[freq_mask][0]
    freq2= freqs[freq_mask][-1]
    freq_mask_reduced= (freq_reduced >= freq1) & (freq_reduced <= freq2)
    alpha=alpha[freq_mask_reduced]
    Beta=Beta[freq_mask_reduced]
    
    n1, n2, n3, n4 = args
    noiseModel = NoiseModel(freqs, n1, n2, n3, n4)[freq_mask]
    E_atm = np.load(f'InversionData/Earth_response_function/atmosphere/E_{len2}.npy')
    E_oc = np.load(f'InversionData/Earth_response_function/ocean/E_{len2}.npy')
    
    Csi = Csi_func(E_atm, ref_trac)[freq_mask]
    Csi_oc = Csi_func(E_oc, ref_trac)[freq_mask]
    SyntheticPSD = Csi*np.abs(alpha) + Csi_oc*np.abs(Beta) + noiseModel
    detrend_psd_stacked = psd_stacked[freq_mask] - noiseModel
    
    Csi*=np.abs(alpha)
    Csi_oc*=np.abs(Beta)
        
    return SyntheticPSD, detrend_psd_stacked, Csi, Csi_oc

def NoiseModel(freqs,n1,n2,n3,n4):
    return n1 + n2*freqs + n3*freqs**(-n4)
def Csi_func(E,ref_trac):
    R=6371*1e3
    Csi = R**4 * E * ref_trac

    return Csi
def IntegralMisfit(args, params,d=10,R = 6371 * 1e3):
    "Fits the observed data over a 1mHz frequency band and "
    "inverts for the contribution of the radial pressure and shear traction."
    "Args are the length of the CCF and the frequency band"
    "Params are the parameters of the inversion for the fit"
    "f1 and f2 in Hz, len2 is a length, and freqs is the array of frequencies"
    f0 = 1e-3
    p0 = [1, 1, 1E-19, 1E-17]

    alpha,Beta,n1,n2 = args[0],args[1],args[2],args[3]
    len2, freqs, DeltaFreq, freq_mask, psd_stacked = params[0], params[1], params[2], params[3], params[4]
    ref_trac = (freqs / f0) ** (-2.3) * (2 * 1e9 / (4 * np.pi * R ** 2))

    # Load the pre-computed excitation functions
    E_atm = np.load(f'InversionData/Earth_response_function/atmosphere/E_{len2}.npy')
    E_oc = np.load(f'InversionData/Earth_response_function/ocean/E_{len2}.npy')
    
    # Calculate the radial and shear pressure PSD
    
    Csi =Csi_func(E_atm, ref_trac)[freq_mask] 
    Csi_oc =Csi_func(E_oc, ref_trac)[freq_mask]

    noiseModel = (n1*p0[2] + n2*p0[3]*np.abs(freqs))[freq_mask] #rev KN
    syntheticPSD = np.zeros(len(freqs))
    #Mute 0S0
    
    syntheticPSD[freq_mask] =  p0[0]*alpha*Csi + p0[1]*Beta*Csi_oc + noiseModel
    syntheticPSD[(freqs<1e-3)] = 0 #Mute 0S0
    # Integrate over the frequency band
    Integral = np.sum((syntheticPSD[freq_mask]*1E20 - psd_stacked*1E20)**2 ) * DeltaFreq #KN Add freq_mask and 1E-20

    return Integral

def getPeakFrequencySeveralModes(file_path, ranks, freq_start, freq_end): # Helper function to extract peak frequencies from the mode file
    """Extract peak frequencies from the mode file for the given ranks."""
    freqs = []
    for rank in ranks:
        modes = read_modes(file_path, rank)
        freqs.extend(value for mode, (value, Q) in modes.items() if freq_start < value * 1e3 < freq_end)
    return np.sort(np.array(freqs))

def optimize_psd(freqBand, freqs, psd_stacked, len2): # Function to optimize PSD, used for parallel execution.
    """Function to optimize PSD, used for parallel execution."""
    f1, f2 = freqBand
    freq_mask = (freqs > f1) & (freqs < f2)
    DeltaFreq = f2 - f1

    options = {'pgtol': 1e-12, 'maxiter': 1000}#, 'disp': True }
    args = [len2, freqs, DeltaFreq, freq_mask, psd_stacked[freq_mask]]
    p0 = [1, 1, 2E-19, 1E-17]
    bounds = [(0, 5), (0, 5), (-1e-17/p0[2], 1e-17/p0[2]), (-2e-15/p0[3], 2e-15/p0[3])] # Normalise for stability
    result = minimize(fun=IntegralMisfit, x0=p0, args=(args,), method='L-BFGS-B', bounds=bounds,options=options)

    #print(result)
    if result.success:
        return 1, result.x[0], result.x[1], result.x[2]*p0[2], result.x[3]*p0[3]  # Freq, alpha, beta. Return the normalisation reversed
    return 1, None, None, None, None


def Misfit(args, params):
    syntheticPSD, _, _,_ = GetSpectrum1(args, params)
    psd_stacked = params[5]
    return np.sum((syntheticPSD - psd_stacked) ** 2)


def GetFreqBands(freqs,Freqs,band):
    "freqs : The full frequency range"
    "Freqs : The discrete frequency centers"
    "band : The width of the frequency band"
    
    freqBands = []
    for Freq in Freqs:
        threshold = 1e-3
        DiscreteFreq = freqs[(freqs >= Freq) & (freqs <= Freq + threshold * 1e-3)]
        
        while len(DiscreteFreq) > 1:
            threshold -= 1e-4
            DiscreteFreq = freqs[(freqs >= Freq) & (freqs <= Freq + threshold * 1e-3)]

        while len(DiscreteFreq) < 1:
            threshold += 1e-4

        freqBands.append((float(DiscreteFreq) - band / 2, float(DiscreteFreq) + band / 2))
            
    return freqBands
    
def fit_frequency_range_parallel(args):
    freq_range, params, p0 = args
    freqMask = (params[3] >= freq_range[0] * 1e-3) & (params[3] <= freq_range[1] * 1e-3)
    params[4] = freqMask
    original_psd = params[5]
    psd_masked = params[5][freqMask]
    params[5] = psd_masked
    
    result = minimize(fun=Misfit, x0=p0, args=(params,), method='Nelder-Mead')
    
    params[5] = original_psd  # Reset original PSD
    return result.x



######################################################
# PLOTTING FUNCTIONS
######################################################

def SimplePlot(axes,freqs,psd,label=None):
    
    axes.plot(freqs*1e3,psd*1e18,label=label)
    axes.set_xlim(0.6,5)
    axes.set_xlabel('Frequency (mHz)')
    axes.set_ylabel('PSD (m^2/s^2/Hz)')
    axes.set_title('Observed PSD')
    
    return

def plotResults(freqs, total_spectrum_atm, total_synt_atm, total_synt_oc, CenterFreqs, meanAlpha, meanBeta, stdAlpha, stdBeta):
    len2=2**16
    freqStart, freqEnd = 0.91, 5
    freqMask = (freqs>=freqStart*1e-3) & (freqs<=freqEnd*1e-3)
    fig = plt.figure(figsize=(18, 8),constrained_layout=True)  # Increase figure size
    gs = GridSpec(12, 20, figure=fig)

    ax1 = fig.add_subplot(gs[0:6, 0:10])
    #ax2 = fig.add_subplot(gs[0:2, 13:18])
    ax2 = fig.add_subplot(gs[0:6, 10:20])
    ax3 = fig.add_subplot(gs[6:12, 0:20])

    #ax3.text(1, 1.7,  'Acceleration PSD ' r'$m^2/s^{3}$', transform=ax5.transAxes, 
    #            ha='right', va='top', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', which='major', labelsize=10, width=0.5, right=True, direction='in',labelright=False, labelleft=False,left=False)

    letters= ['(a)','(b)']
    for i,ax in enumerate([ax1,ax2]):
        ax.text(-0.05, 1.09, letters[i], transform=ax.transAxes, fontsize=16, va='top', ha='right')
    ax3.text(0.05, 1.09, '(c)', transform=ax3.transAxes, fontsize=16, va='top', ha='right')


    x1,x2 = 0.8,5
    for axes in [ax1,ax2]:
        axes.text(0.03, 1.05,  r'$\times 1e^{-18}$' , transform=axes.transAxes, 
                ha='center',va='center', fontsize=13)    
        #plot_mode_lines(axes,'./Smode.out','0','r',[x1,x2])
        #plot_mode_lines(axes,'./Smode.out','0','tab:green',[x1,x2])
        plot_mode_lines(axes,'InversionData/Smode.out','0','slategrey',[x1,x2])
        #plot_mode_lines(axes,'./Smode.out','1','tab:green',[x1,x2])
        axes.plot(freqs[freqMask]*1e3,total_spectrum_atm[0:len(freqs[freqMask])]*1e18,label='Observed Spectrum',linewidth=0.8,color='k')
        
        #axes.set_xlim([x1,x2])
        #axes.set_xlim([0.8,2])
        axes.set_ylim(-0.11,0.8)
        #axes.set_ylim(-0.05,1.5)
        axes.set_xlabel('Frequency (mHz)',fontsize=14)
        axes.set_ylabel('acceleration PSD ' r'$m^.s^{-3}$',fontsize=14)
        axes.tick_params(axis='x', which='major', labelsize=14, width=1,length=5, direction='out',top=True)
        axes.tick_params(axis='x', which='minor', labelsize=14, width=1,length=3, direction='out',top=True)
        axes.set_yticks([0,0.2,0.4,0.6])
        #axes.grid()
        
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter('{x:.1f}')
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    #ax1.set_xticks([1,1.5,2,2.5])
    ax1.plot(freqs[freqMask]*1e3,total_synt_oc[0:len(freqs[freqMask])]*1e18,label='Synthetic Spectrum - Shear Source ',linewidth=1,color='tab:blue',linestyle='--')
    ax1.plot(freqs[freqMask]*1e3,total_synt_atm[0:len(freqs[freqMask])]*1e18,label='Synthetic Spectrum - Pressure Source ',linewidth=1,color='tab:red',linestyle='--')
    
    ax2.plot(freqs[freqMask]*1e3,total_synt_oc[0:len(freqs[freqMask])]*1e18,label='Synthetic Spectrum - Shear Source ',linewidth=1,color='tab:blue',linestyle='--')
    ax2.plot(freqs[freqMask]*1e3,total_synt_atm[0:len(freqs[freqMask])]*1e18,label='Synthetic Spectrum - Pressure Source ',linewidth=1,color='tab:red',linestyle='--')
    ax1.set_xlim([1.2,2.2])
    ax2.set_xlim([2.2,3.3])
    ax1.set_title('Synthetic reference spectra : 1.2 - 2.2 mHz',fontsize=18)
    ax2.set_title('Synthetic reference spectra : 2.2 - 3 mHz',fontsize=18)
    
    ax1.legend(loc='upper right', fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)






    # Modify ax3 to plot meanAlpha and meanBeta with error bars
    ax3.errorbar(CenterFreqs*1e3, meanAlpha, yerr=stdAlpha, ecolor='tab:red', markersize=8,
                elinewidth=1, capsize=5, label='random pressure source', color='tab:red', linestyle='-',linewidth=2)

    ax3.errorbar(CenterFreqs*1e3, meanBeta, yerr=stdBeta, ecolor='tab:blue', markersize=8,
                elinewidth=1, capsize=5, label='random shear source', color='tab:blue', linestyle='-',linewidth=2)

    ax3.grid()
    ax3.set_xlim([1.2, freqEnd])  # Main figure shows 1.5 - 5 mHz
    ax3.set_ylim([-0.1, 1.5])
    ax3.set_xlabel('Frequency (mHz)', fontsize=14)
    ax3.set_ylabel('Normalized amplitude', fontsize=14)
    ax3.set_title("Normalized source spectra of Pressure and Shear traction sources", fontsize=14)
    ax3.legend()


    # Ticks formatting
    ax3.xaxis.set_major_locator(MultipleLocator(1))
    ax3.xaxis.set_major_formatter('{x:.1f}')
    ax3.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax3.set_yticks([0, 0.5, 1, 1.5, 2])
    ax3.tick_params(axis='both', which='major', labelsize=12, direction='out', bottom=True, length=6, width=1)
    ax3.tick_params(axis='both', which='minor', labelsize=12, direction='out', bottom=True, length=4, width=1)

    # Annotations with arrows
    #ax3.annotate(r'$_0S_{29}$', xy=(3.7254, 0.1), xytext=(3.7254, 0.5),
    #             fontsize=15, arrowprops=dict(facecolor='black', arrowstyle='->'))
    #ax3.annotate(r'', xy=(3.7254, 0.8), xytext=(3.78, 0.3),
                #fontsize=15, arrowprops=dict(facecolor='black', arrowstyle='->'))
    #ax3.annotate(r'$_0S_{37}$', xy=(4.4419, 0.1), xytext=(4.4419, 0.5),
    #            fontsize=15, arrowprops=dict(facecolor='black', arrowstyle='->'))
    #ax3.annotate(r'', xy=(4.4419, 0.8), xytext=(4.5, 0.4),
                #fontsize=15, arrowprops=dict(facecolor='black', arrowstyle='->'))


    for ax in [ax1,ax2]:
        ax.annotate(r'$_1S_{0}$', xy=(1.6313, 0.05), xytext=(1.6313, 0.2),
                fontsize=15, arrowprops=dict(facecolor='black', arrowstyle='->'),color='tab:red')
        ax.annotate(r'$_2S_{0}$', xy=(2.5105, 0.05), xytext=(2.48, 0.2),
                fontsize=15, arrowprops=dict(facecolor='black', arrowstyle='->'),color='tab:red')
        ax.annotate(r'$_0S_{0}$', xy=(0.81431, 0.2), xytext=(0.92, 0.5),
                fontsize=15, arrowprops=dict(facecolor='black', arrowstyle='->'),color='tab:red')
    #1.6313
    #2.5105

    # Display the plot
    return ax1,ax2,ax3


def read_modes(file_path,n):
        # Initialize an empty dictionary to store modes and values
        mode_value_dict = {}

        # Open the file
        with open(file_path, 'r') as file:
            # Read each line in the file
            lines = file.readlines()

            # Iterate over the lines
            for line in lines:
                # Check if the line starts with '0' in the first column
                if line.split()[0] == n:
                    # Split the line into columns
                    columns = line.split()
                    # Extract the mode and value
                    mode = ''.join(columns[:3])
                    freq = float(columns[3])
                    Q = float(columns[7])
                    # Add mode and value to the dictionary
                    mode_value_dict[mode] = freq,Q

        return mode_value_dict
    
def plot_mode_lines(ax, file_path,rank,color,xlim ,mode_range_start=0, mode_range_end=100):
    modes = read_modes(file_path,rank)
    modes_to_plot = {mode: value for mode, (value,Q) in modes.items() if
                     int(mode.split('S')[-1]) >= mode_range_start and int(mode.split('S')[-1]) <= mode_range_end}
    #modes_to_plot_overtone = {overtone_mode: overtone_value for overtone_mode, (overtone_value,overtone_Q) in modes_overtone.items() if
      #               int(overtone_mode.split('S')[-1]) >= mode_range_start and int(overtone_mode.split('S')[-1]) <= mode_range_end}
    # Iterate over the filtered modes
    first_iteration = True
    for index,(mode, mode_value) in enumerate(modes_to_plot.items()):
        # Plot the modes
        mode_value_vline = mode_value * 1e+3  # Use default frequency for vline
        mode_value_text = mode_value * 1e+3   # Adjust the frequency slightly for text
        if first_iteration:
            ax.vlines(mode_value_vline, ymin=-0.5, ymax=20, linewidth=0.5, linestyle='-',alpha=1,colors=color,label='PREM fundamentals')
            first_iteration = False
        else:
            ax.vlines(mode_value_vline, ymin=-0.5, ymax=20, linewidth=0.5 ,linestyle='-',alpha=1,colors=color)
    return


def meanAndInterpResults(alpha_dict, beta_dict, freq_dict, bandWidths,freqs):
    meanAlpha = np.mean([alpha_dict[f"{band}"] for band in bandWidths], axis=0)
    meanBeta = np.mean([beta_dict[f"{band}"] for band in bandWidths], axis=0)
    stdAlpha = np.std([alpha_dict[f"{band}"] for band in bandWidths], axis=0)
    stdBeta = np.std([beta_dict[f"{band}"] for band in bandWidths], axis=0)
    CenterFreqs = freq_dict[f"0.001"]
    
    # Interpolate the alpha and beta values

    freqStart, freqEnd = 0.91e-3, 5e-3
    freqMask = (freqs > freqStart) & (freqs < freqEnd)

    interp_func = interp1d(CenterFreqs, meanAlpha, kind='cubic', fill_value="extrapolate")
    
    alpha_interp = interp_func(freqs[freqMask])
    interp_func = interp1d(CenterFreqs, meanBeta, kind='cubic', fill_value="extrapolate")
    Beta_interp = interp_func(freqs[freqMask])
    
    return alpha_interp, Beta_interp, freqs[freqMask], CenterFreqs,meanAlpha,meanBeta,stdAlpha,stdBeta


def add_zebra_frame(ax, lw=2, crs="pcarree", zorder=None):

    ax.spines["geo"].set_visible(False)
    left, right, bot, top = ax.get_extent()
    
    # Alternate black and white line segments
    bws = itertools.cycle(["k", "white"])

    xticks = sorted([left, *ax.get_xticks(), right])
    xticks = np.unique(np.array(xticks))
    yticks = sorted([bot, *ax.get_yticks(), top])
    yticks = np.unique(np.array(yticks))
    for ticks, which in zip([xticks, yticks], ["lon", "lat"]):
        for idx, (start, end) in enumerate(zip(ticks, ticks[1:])):
            bw = next(bws)
            if which == "lon":
                xs = [[start, end], [start, end]]
                ys = [[bot, bot], [top, top]]
            else:
                xs = [[left, left], [right, right]]
                ys = [[start, end], [start, end]]

            # For first and lastlines, used the "projecting" effect
            capstyle = "butt" if idx not in (0, len(ticks) - 2) else "projecting"
            for (xx, yy) in zip(xs, ys):
                ax.plot(
                    xx,
                    yy,
                    color=bw,
                    linewidth=lw,
                    clip_on=False,
                    transform=crs,
                    zorder=zorder,
                    solid_capstyle=capstyle,
                    # Add a black border to accentuate white segments
                    path_effects=[
                        pe.Stroke(linewidth=lw + 1, foreground="black"),
                        pe.Normal(),
                    ],
                )
                
               
from matplotlib.ticker import ScalarFormatter, FuncFormatter, MultipleLocator    
custom_format = lambda x, pos: f'{x:.0f}' if x != 0 else '0'  # Custom format function for y-axis

def plotMapSpectrum(coordinates_used, station_used, psd_stacked, freqs):
        
    fig = plt.figure(figsize=(16, 6))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

    # Left: Map
    crs = ccrs.PlateCarree()
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    ax_map.set_global()
    ax_map.coastlines(resolution='110m')
    ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
    #ax_map.gridlines()
    ax_map.set_aspect('auto')
    #ax_map.set_xticks([0,60,120,180,240,300,360])
    ax_map.set_title('Localisation of the stations used in this study')
    # Plot each station as a red dot
    for station_name, coordinate in zip(station_used, coordinates_used):
        ax_map.scatter(coordinate[1], coordinate[0], transform=ccrs.PlateCarree(), color='tab:red', s=20, zorder=5)
        #ax_map.text(coordinate[1], coordinate[0], station_name, transform=ccrs.PlateCarree(), color='black', fontsize=8)
    ax_map.gridlines(draw_labels=True, dms=True, linestyle='--',
                #   x_inline=True, y_inline=True,
                xlocs=np.arange(-150, 180, 50), ylocs=np.arange(-80, 81, 40))

    add_zebra_frame(ax_map, crs=crs)
    # Right: PSD plot
    ax_psd = fig.add_subplot(gs[0, 1])
    ax_psd.semilogy(freqs * 1e3, 2*psd_stacked*1e18, color='k', alpha=0.9,linewidth=0.5,label='Stacked spectrum')
    #ax_psd.semilogy(freqs * 1e3, std, label=f'STD {nb_station_used} stations', color='red', alpha=0.7)
    ax_psd.set_xlim([0.6, 7])
    ax_psd.set_ylim([2.5e-1, 1.5e1])
    plot_mode_lines(ax_psd,'InversionData/Smode.out','0','tab:blue',[0.6,7])

    ax_psd.legend()

    # Set labels and titles
    ax_psd.set_xlabel('Frequency (mHz)')
    ax_psd.set_ylabel('Acceleration PSD  ' r'$m^{2}. s^{-3}$')
    ax_psd.set_title('Stacked PSD at the 75 stations')
    ax_psd.tick_params(axis='y',which='both',right=True)
    ax_psd.tick_params(axis='x',which='major',top=True,length=5)
    ax_psd.tick_params(axis='x',which='minor',top=True,length=3)
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)  # Disable scientific notation
    formatter.set_useOffset(False)  # Disable offset notation if present
    ax_psd.yaxis.set_major_formatter(FuncFormatter(custom_format))
    #ax_psd.set_facecolor('lightblue')
    ax_psd.text(0.04, 1.04,  r'$\times 10^{-18}$' , transform=ax_psd.transAxes, 
                ha='center',va='center', fontsize=10)
    ax_psd.xaxis.set_major_locator(MultipleLocator(1))
    ax_psd.xaxis.set_major_formatter('{x:.0f}')
    ax_psd.xaxis.set_minor_locator(MultipleLocator(0.25))
    plt.tight_layout()

    letters=['(a)','(b)']
    for i,ax in enumerate([ax_map,ax_psd]):
        ax.text(0.05, 1.12, letters[i], transform=ax.transAxes, fontsize=12, va='top', ha='right')
        plt.savefig('Figures/StackedSpectrumMap.png', dpi=300, bbox_inches='tight')
    plt.show()

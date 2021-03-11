#!/usr/bin/python
# Robert Lindner
# Autonomous Gaussian Decomposition

# Standard Libs
import time

# Standard Third Party
import numpy as np
from scipy.interpolate import interp1d

# from scipy.optimize import leastsq, minimize
from lmfit import minimize as lmfit_minimize
from lmfit import Parameters

# import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.ndimage.filters import median_filter, convolve

# Python Regularized derivatives
from . import tvdiff


def vals_vec_from_lmfit(lmfit_params):
    """ Return Python list of parameter values from LMFIT Parameters object"""
    vals = [value.value for value in lmfit_params.values()]
    return vals


def errs_vec_from_lmfit(lmfit_params):
    """ Return Python list of parameter uncertainties from LMFIT Parameters object"""
    errs = [value.stderr for value in lmfit_params.values()]
    return errs


def paramvec_to_lmfit(paramvec):
    """ Transform a Python iterable of parameters into a LMFIT Parameters object"""
    ncomps = len(paramvec) // 3
    params = Parameters()
    
    absorption_amps = paramvec[:ncomps]
    absorption_widths = paramvec[ncomps:2*ncomps]
    absorption_means = paramvec[2*ncomps:3*ncomps]

    sigma_level_tau=3
    sigma_tau=0.0005897952
    sigma_level_tb=3
    sigma_tb=0.055
    
    for i in range(ncomps):#for the amplitudes in opacity 
        params.add(f'a{i}', value=absorption_amps[i], min=sigma_level_tau*sigma_tau)
    for i in range(ncomps):#for the widths in opacity
        params.add(f'width{i}', value=absorption_widths[i],
        min=(np.sqrt((sigma_level_tb*sigma_tb)/(21.866*(1-np.exp(-sigma_level_tau*sigma_tau))))))
    for i in range(ncomps): #for the positions in opacity
        params.add(f'position{i}', value=absorption_means[i])

    #print(f'printing {len(params)} abs comps')
    #print(params)
    #print('done with abs comps')
    return params



def paramvec_p3_to_lmfit(paramvec, max_tb, p_width, d_mean, min_dv, abs_widths=None, abs_pos=None):
    """ Transform a Python iterable of parameters into a LMFIT Parameters object"""
    ncomps = len(paramvec) // 5
    params = Parameters()

    emission_amps = paramvec[:ncomps]
    emission_widths = paramvec[ncomps:2*ncomps]
    emission_means = paramvec[2*ncomps:3*ncomps]
    labels = np.array(paramvec[3*ncomps:4*ncomps]).astype(int)
    tau = paramvec[4*ncomps:5*ncomps] 


    min_ts=15 #below this temp atomic H is unlikely to exist
    sigma_level_tau=3 #3 sigma min
    sigma_tau=0.0005897952 #0.0005897952 is the measured tau noise
    sigma_level_tb=3 #3 sigma min
    sigma_tb=0.055 #0.055mK is the estimate of the Tb noise from the GASS bonn server, 

    print(f'emission_amps = {emission_amps}')
    print(f'emission_widths = {emission_widths}')
    print(f'emission_means = {emission_means}')

    #Tb AMPLITUDES
    for i in range(len(labels)): 
        #ABS-MATCHED
        if labels[i] == 1: 
            if max_tb is not None:
                if max_tb == "max":
                    #set max amplitude based on the absorption fit amplitude and width to not be beyond what is possible when fully thermalised 
                    max_tb_value = (
                        21.866
                        * np.float(emission_widths[i]) ** 2
                        * (1.0 - np.exp(-1.0 * tau[i]))
                    )
                else:
                    #set arbitrary max temperature
                    max_tb_value = max_tb
                #add parametr with max bound
                params.add(f'a{i}', value=emission_amps[i], 
                #min=0,
                min=min_ts*(1-np.exp(-tau[i])), 
                max=max_tb_value) #possibly where the emission only comps and maybe some abs comps are being set? #3 sigma min and max set by the measured tau comp
            else:
                #add parameter without max bound
                params.add(f'a{i}', value=emission_amps[i], min=min_ts*(1-np.exp(-tau[i])))  


        #EMISSION ONLY
        else:
            if max_tb is not None:
                if max_tb == "max":
                    #set the max Tb to be based on the absorption width and a sigma_tau
                    max_tb_value = (
                        21.866
                        * np.float(emission_widths[i]) ** 2 #check where the emission width intitial guess comes from and keep it below 30km/s
                        * (1.0 - np.exp(-sigma_level_tau * sigma_tau))
                    )
                else:
                    max_tb_value = max_tb
                params.add(f'a{i}', value=emission_amps[i], min=(sigma_level_tb*sigma_tb), max=max_tb_value)
            else:
                params.add(f'a{i}', value=emission_amps[i], min=(sigma_level_tb*sigma_tb)) 
        print(f'amp {i} = {params[f"a{i}"]}')


    #WIDTHS (FWHM)
    for i in range(len(labels)): #delete this redundant loop, just for confirming teh same order of exectuion for rewriting block    
        #ABS-MATCHED
        #print(emission_widths[i])
        if labels[i] == 1:  
            if p_width < 0.001:
                p_width = 0.001
            #uncomment below line to implement the expr functionality
            #params.add(f'd{i}', value=0.00001,min=0,max=21.866*(1-np.exp(-tau[i])),vary=True)
            #params.add(
            #    f'delta{i}',
            #    value=0.0000000001,#tau[i], #maybe a bad initial guess that won't always work. not certain.
            #    min=0,
            #    max=21.866*(1-np.exp(-tau[i])),
            #    vary=True
            #    )

            tenpercent=emission_widths[i] - np.abs(p_width * emission_widths[i])
            ampbased=np.sqrt((params[f'a{i}'].max)/(21.866*(1-np.exp(-tau[i]))))
            print(f'10% = {tenpercent}')
            print(f'amp based bound = {ampbased}')

            if tenpercent > ampbased:
                print('10% IS HIGHER')
            else:
                print('AMP BOUND IS HIGHER')
            #comment out below block to implement the expr functionality
            #need to find a way to reincorporate the bounds for +-10% here
            params.add(
                f'w{i}',
                #value=emission_widths[i],
                value=ampbased+0.1, #revert to old line and delete, test only
                #min=emission_widths[i] - np.abs(p_width * emission_widths[i]),
                min=np.max([
                    (abs_widths[i] - np.abs(p_width * abs_widths[i])),
                (np.sqrt((params[f'a{i}'].max)/(21.866*(1-np.exp(-tau[i]))))) #using .max is a bit of a brute force solution and is excessive since the actual value may not come that high, will prohibit the solution of fully thermalised lines
                ]),
                max=abs_widths[i] + np.abs(p_width * abs_widths[i]))

            #not sure how or if i can also incorporate the p_width bounds into this parameter
            #uncomment below line to implement the expr functionality
            #params.add(f'w{i}', expr=f'sqrt(a{i}/d{i})')
        #EMISSION ONLY
        else:
            #print(f'em width {i-ncomps}')

            #uncomment below block to implement the expr functionality
            #params.add(f'd{i}', value=0.00001,min=0,max=21.866*(1-np.exp(-tau[i])),vary=True)
            #params.add(f'w{i}', expr=f'sqrt(a{i}/d{i})')

            #comment out below block to implement the expr functionality
            #need to reincorporate the max of min_dv or the eqn here so it doesn't get too narrow
            params.add(f'w{i}', 
            value=emission_widths[i], 
            min=np.max([
                min_dv,
            np.sqrt((params[f'a{i}'].max)/(21.866*(1-np.exp(-sigma_level_tau * sigma_tau)))) #needs to be based on previous amp calculated
            ])) #using .max is a bit of a brute force solution and is excessive since the actual value may not come that high, will prohibit the solution of fully thermalised lines
        print(f'fwhm {i} = {params[f"w{i}"]}')


    #POSITIONS
    for i in range(len(labels)): #delete this redundant loop, just for confirming teh same order of exectuion for rewriting block           
        #ABS-MATCHED
        if labels[i] == 1:
            if d_mean < 0.001:
                d_mean = 0.001
            params.add(
                f'p{i}',
                value=emission_means[i],
                min=abs_pos[i] - d_mean,
                max=abs_pos[i] + d_mean,
            )
        #EMISSION ONLY
        else:
            params.add(f'p{i}', value=emission_means[i])
        print(f'position {i} = {params[f"p{i}"]}')
    #print(labels)
    #print(params)
    #print('iteration')
    return params


def create_fitmask(size, offsets_i, di):
    """Return valid domain for intermediate fit in d2/dx2 space

    fitmask = (0,1)
    fitmaskw = (True, False)
    """
    fitmask = np.zeros(size)
    for i in range(len(offsets_i)):
        fitmask[int(offsets_i[i] - di[i]) : int(offsets_i[i] + di[i])] = 1.0
    fitmaskw = fitmask == 1.0
    return fitmask, fitmaskw


def say(message, verbose=False):
    """Diagnostic messages"""
    if verbose:
        print(message)


def gaussian(peak, FWHM, mean):
    """Return a Gaussian function"""
    sigma = FWHM / 2.354820045  # (2 * sqrt( 2 * ln(2)))
    return lambda x: peak * np.exp(-((x - mean) ** 2) / 2.0 / sigma ** 2)


def func(x, *args):
    """Return multi-component Gaussian model F(x).

    Parameter vector kargs = [amp1, ..., ampN, width1, ..., widthN, mean1, ..., meanN],
    and therefore has len(args) = 3 x N_components.
    """
    ncomps = len(args) // 3
    yout = np.zeros(len(x))
    for i in range(ncomps):
        yout = yout + gaussian(args[i], args[i + ncomps], args[i + 2 * ncomps])(x)
    return yout

def param_extract(dictionary, unique_identifier):
    """
    inputs:
    dictionary - any dictionary (usually a lmfit_minimise result such as result.params)
    unique_identifier - any string that occurs only in the target keys from the dictionary i.e.'a','w','p' or 'd'
    could work with floats or longer string etc but i don't guarantee that as its untested.

    Pulls out key,value pairs from a dictionary based on idenitifiers in the key. 
    In most cases used to withdraw the amplitude,width,position and delta parameters that 
    were output from lmfit_minimize.
    e.g. param_extract(result_em.params,'a') will pull from the result_em.params dictionary 
    all the keys with a in their name that should denote an amplitude value
    """
    return {key: value for key, value in dictionary.items() if unique_identifier in key}

def objective_leastsq(paramslm, vel, data, errors):

    amp_dict=param_extract(paramslm,'a')
    width_dict=param_extract(paramslm,'w')
    pos_dict=param_extract(paramslm,'p')
    delta_dict=param_extract(paramslm,'d') #not passed further down the line but needs to be input into lmfit minimise

    params = np.concatenate([
        vals_vec_from_lmfit(amp_dict),
        vals_vec_from_lmfit(width_dict),
        vals_vec_from_lmfit(pos_dict)])

    resids = (func(vel, *params).ravel() - data.ravel()) / errors
    return resids

def exprfunc(x, *args):
    """Return multi-component Gaussian model F(x).

    Parameter vector kargs = [amp1, ..., ampN, width1, ..., widthN, mean1, ..., meanN],
    and therefore has len(args) = 3 x N_components.
    """
    ncomps = len(args) // 3
    yout = np.zeros(len(x))

    amps=args[0:ncomps]
    widths=args[ncomps:2*ncomps]
    pos=args[2*ncomps:3*ncomps]
    deltas=args[3*ncomp:4*ncomps]

    for i in range(ncomps):
        yout = yout + gaussian(amps[i], widths[i], pos[i])(x)
    return yout


def initialGuess(
    vel,
    data,
    errors=None,
    alpha=None,
    # plot=False,
    mode="python",
    verbose=False,
    SNR_thresh=5.0,
    BLFrac=0.1,
    SNR2_thresh=5.0,
    deblend=True,
):
    """Find initial parameter guesses (AGD algorithm)

    data,             Input data
    dv,             x-spacing absolute units
    alpha = No Default,     regularization parameter
    plot = False,     Show diagnostic plots?
    verbose = True    Diagnostic messages
    SNR_thresh = 5.0  Initial Spectrum S/N threshold
    BLFrac =          Edge fraction of data used for S/N threshold computation
    SNR2_thresh =   S/N threshold for Second derivative
    mode = Method for taking derivatives
    """

    errors = None  # Until error

    say("\n\n  --> initialGuess() \n", verbose)
    say("Algorithm parameters: ", verbose)
    say("alpha = {0}".format(alpha), verbose)
    say("SNR_thesh = {0}".format(SNR_thresh), verbose)
    say("SNR2_thesh = {0}".format(SNR2_thresh), verbose)
    say("BLFrac = {0}".format(BLFrac), verbose)

    if not alpha:
        print("Must choose value for alpha, no default.")
        return

    if np.any(np.isnan(data)):
        print("NaN-values in data, cannot continue.")
        return

    # Data inspection
    vel = np.array(vel)
    data = np.array(data)
    dv = np.abs(vel[1] - vel[0])
    fvel = interp1d(np.arange(len(vel)), vel)  # Converts from index -> x domain
    data_size = len(data)

    # Take regularized derivatives
    t0 = time.time()
    if mode == "python":
        say("Taking python derivatives...", verbose)
        u = tvdiff.TVdiff(data, dx=dv, alph=alpha)
        u2 = tvdiff.TVdiff(u, dx=dv, alph=alpha)
        u3 = tvdiff.TVdiff(u2, dx=dv, alph=alpha)
        u4 = tvdiff.TVdiff(u3, dx=dv, alph=alpha)
    elif mode == "conv":
        say("Convolution sigma [pixels]: {0}".format(alpha), verbose)
        gauss_sigma = alpha
        gauss_sigma_int = np.max([np.fix(gauss_sigma), 5])
        gauss_dn = gauss_sigma_int * 6

        xx = np.arange(2 * gauss_dn + 2) - (gauss_dn) - 0.5
        gauss = np.exp(-(xx ** 2) / 2.0 / gauss_sigma ** 2)
        gauss = gauss / np.sum(gauss)
        gauss1 = np.diff(gauss) / dv
        gauss3 = np.diff(np.diff(gauss1)) / dv ** 2

        xx2 = np.arange(2 * gauss_dn + 1) - (gauss_dn)
        gauss2 = np.exp(-(xx2 ** 2) / 2.0 / gauss_sigma ** 2)
        gauss2 = gauss2 / np.sum(gauss2)
        gauss2 = np.diff(gauss2) / dv
        gauss2 = np.diff(gauss2) / dv
        gauss4 = np.diff(np.diff(gauss2)) / dv ** 2

        u = convolve(data, gauss1, mode="wrap")
        u2 = convolve(data, gauss2, mode="wrap")
        u3 = convolve(data, gauss3, mode="wrap")
        u4 = convolve(data, gauss4, mode="wrap")

    say(
        "...took {0:4.2f} seconds per derivative.".format((time.time() - t0) / 4.0),
        verbose,
    )

    # Decide on signal threshold
    if not errors:
        errors = np.std(data[0 : int(BLFrac * data_size)])

    thresh = SNR_thresh * errors
    mask1 = np.array(data > thresh, dtype="int")[1:]  # Raw Data S/N
    mask3 = np.array(u4.copy()[1:] > 0.0, dtype="int")  # Positive 4nd derivative

    if SNR2_thresh > 0.0:
        wsort = np.argsort(np.abs(u2))
        RMSD2 = (
            np.std(u2[wsort[0 : int(0.5 * len(u2))]]) / 0.377
        )  # RMS based in +-1 sigma fluctuations
        say("Second derivative noise: {0}".format(RMSD2), verbose)
        thresh2 = -RMSD2 * SNR2_thresh
        say("Second derivative threshold: {0}".format(thresh2), verbose)
    else:
        thresh2 = 0.0
    mask4 = np.array(u2.copy()[1:] < thresh2, dtype="int")  # Negative second derivative

    # Find optima of second derivative
    # --------------------------------
    zeros = np.abs(np.diff(np.sign(u3)))
    zeros = zeros * mask1 * mask3 * mask4
    offsets_data_i = np.array(np.where(zeros)).ravel()  # Index offsets
    offsets = fvel(offsets_data_i + 0.5)  # Velocity offsets (Added 0.5 July 23)
    N_components = len(offsets)
    say(
        "Components found for alpha={1}: {0}".format(N_components, alpha),
        verbose=verbose,
    )

    # Check if nothing was found, if so, return null
    # ----------------------------------------------
    if N_components == 0:
        odict = {
            "means": [],
            "FWHMs": [],
            "amps": [],
            "u2": u2,
            "errors": errors,
            "thresh2": thresh2,
            "thresh": thresh,
            "N_components": N_components,
        }

        return odict

    #        say('AGD2.initialGuess: No components found for alpha={0}! Returning ([] [] [] [] [])'.format(alpha))
    #        return [], [], [], u2

    # Find Relative widths, then measure
    # peak-to-inflection distance for sharpest peak
    widths = np.sqrt(np.abs(data / u2)[offsets_data_i])
    FWHMs = widths * 2.355
    # print("u2", u2, "ncomps", N_components, "offsets", offsets, "hmm fwhms", FWHMs)

    # Attempt deblending.
    # If Deblending results in all non-negative answers, keep.
    amps = np.array(data[offsets_data_i])
    keep = FWHMs > 0.0
    offsets = offsets[keep]
    FWHMs = FWHMs[keep]
    amps = amps[keep]
    N_components = len(amps)
    # print("new ncomps", N_components)

    if deblend:
        FF_matrix = np.zeros([len(amps), len(amps)])
        for i in range(FF_matrix.shape[0]):
            for j in range(FF_matrix.shape[1]):
                FF_matrix[i, j] = np.exp(
                    -((offsets[i] - offsets[j]) ** 2) / 2.0 / (FWHMs[j] / 2.355) ** 2
                )
        amps_new = lstsq(FF_matrix, amps, rcond=None)[0]
        if np.all(amps_new > 0):
            amps = amps_new

    odict = {
        "means": offsets,
        "FWHMs": FWHMs,
        "amps": amps,
        "u2": u2,
        "errors": errors,
        "thresh2": thresh2,
        "thresh": thresh,
        "N_components": N_components,
    }

    return odict


def AGD(
    vel,
    data,
    errors,
    alpha1=None,
    alpha2=None,
    # plot=False,
    mode="python",
    verbose=False,
    SNR_thresh=5.0,
    BLFrac=0.1,
    SNR2_thresh=5.0,
    deblend=True,
    perform_final_fit=True,
    phase="one",
):
    """Autonomous Gaussian Decomposition"""

    if type(SNR2_thresh) != type([]):
        SNR2_thresh = [SNR2_thresh, SNR2_thresh]
    if type(SNR_thresh) != type([]):
        SNR_thresh = [SNR_thresh, SNR_thresh]

    say("\n  --> AGD() \n", verbose)

    if (not alpha2) and (phase == "two"):
        print("alpha2 value required")
        return

    dv = np.abs(vel[1] - vel[0])
    v_to_i = interp1d(vel, np.arange(len(vel)))

    # --------------------------------------#
    # Find phase-one guesses               #
    # --------------------------------------#
    agd1 = initialGuess(
        vel,
        data,
        errors=None,
        alpha=alpha1,
        # plot=plot,
        mode=mode,
        verbose=verbose,
        SNR_thresh=SNR_thresh[0],
        BLFrac=BLFrac,
        SNR2_thresh=SNR2_thresh[0],
        deblend=deblend,
    )

    amps_g1, widths_g1, offsets_g1, u2 = (
        agd1["amps"],
        agd1["FWHMs"],
        agd1["means"],
        agd1["u2"],
    )
    params_g1 = np.append(np.append(amps_g1, widths_g1), offsets_g1)
    ncomps_g1 = len(params_g1) // 3
    ncomps_g2 = 0  # Default
    ncomps_f1 = 0  # Default

    # ----------------------------#
    # Find phase-two guesses #
    # ----------------------------#
    if phase == "two":
        say("Beginning phase-two AGD... ", verbose)
        ncomps_g2 = 0

        # ----------------------------------------------------------#
        # Produce the residual signal                               #
        #  -- Either the original data, or intermediate subtraction #
        # ----------------------------------------------------------#
        if ncomps_g1 == 0:
            say(
                "Phase 2 with no narrow comps -> No intermediate subtration... ",
                verbose,
            )
            residuals = data
        else:
            # "Else" Narrow components were found, and Phase == 2, so perform intermediate subtraction...

            # The "fitmask" is a collection of windows around the a list of phase-one components
            fitmask, fitmaskw = create_fitmask(
                len(vel), v_to_i(offsets_g1), widths_g1 / dv / 2.355 * 0.9
            )
            notfitmask = 1 - fitmask

            # Error function for intermediate optimization
            def objectiveD2_leastsq(paramslm):
                params = vals_vec_from_lmfit(paramslm)
                model0 = func(vel, *params)
                model2 = np.diff(np.diff(model0.ravel())) / dv / dv
                resids1 = fitmask[1:-1] * (model2 - u2[1:-1]) / errors[1:-1]
                resids2 = notfitmask * (model0 - data) / errors / 10.0
                return np.append(resids1, resids2)

            # Perform the intermediate fit using LMFIT
            t0 = time.time()
            say("Running LMFIT on initial narrow components...", verbose)
            lmfit_params = paramvec_to_lmfit(params_g1)
            result = lmfit_minimize(objectiveD2_leastsq, lmfit_params, method="leastsq")
            params_f1 = vals_vec_from_lmfit(result.params)
            ncomps_f1 = len(params_f1) // 3

            # # Make "FWHMS" positive
            # params_f1[0:ncomps_f1][params_f1[0:ncomps_f1] < 0.0] = (
            #     -1 * params_f1[0:ncomps_f1][params_f1[0:ncomps_f1] < 0.0]
            # )

            del lmfit_params
            say("LMFIT fit took {0} seconds.".format(time.time() - t0))

            if result.success:
                # Compute intermediate residuals
                # Median filter on 2x effective scale to remove poor subtractions of strong components
                intermediate_model = func(
                    vel, *params_f1
                ).ravel()  # Explicit final (narrow) model
                median_window = 2.0 * 10 ** ((np.log10(alpha1) + 2.187) / 3.859)
                residuals = median_filter(
                    data - intermediate_model, np.int(median_window)
                )
            else:
                residuals = data
            # Finished producing residual signal # ---------------------------

        # Search for phase-two guesses
        agd2 = initialGuess(
            vel,
            residuals,
            errors=None,
            alpha=alpha2,
            mode=mode,
            verbose=verbose,
            SNR_thresh=SNR_thresh[1],
            BLFrac=BLFrac,
            SNR2_thresh=SNR2_thresh[1],  # June 9 2014, change
            deblend=deblend,
            # plot=plot,
        )
        ncomps_g2 = agd2["N_components"]
        if ncomps_g2 > 0:
            params_g2 = np.concatenate([agd2["amps"], agd2["FWHMs"], agd2["means"]])
        else:
            params_g2 = []
        u22 = agd2["u2"]

        # END PHASE 2 <<<

    # Check for phase two components, make final guess list
    # ------------------------------------------------------
    if phase == "two" and (ncomps_g2 > 0):
        amps_gf = np.append(params_g1[0:ncomps_g1], params_g2[0:ncomps_g2])
        widths_gf = np.append(
            params_g1[ncomps_g1 : 2 * ncomps_g1], params_g2[ncomps_g2 : 2 * ncomps_g2]
        )
        offsets_gf = np.append(
            params_g1[2 * ncomps_g1 : 3 * ncomps_g1],
            params_g2[2 * ncomps_g2 : 3 * ncomps_g2],
        )
        params_gf = np.concatenate([amps_gf, widths_gf, offsets_gf])
        ncomps_gf = len(params_gf) // 3
    else:
        params_gf = params_g1
        ncomps_gf = len(params_gf) // 3

    # Sort final guess list by amplitude
    # ----------------------------------
    say("N final parameter guesses: " + str(ncomps_gf))
    amps_temp = params_gf[0:ncomps_gf]
    widths_temp = params_gf[ncomps_gf : 2 * ncomps_gf]
    offsets_temp = params_gf[2 * ncomps_gf : 3 * ncomps_gf]
    w_sort_amp = np.argsort(amps_temp)[::-1]
    params_gf = np.concatenate(
        [amps_temp[w_sort_amp], widths_temp[w_sort_amp], offsets_temp[w_sort_amp]]
    )

    if perform_final_fit:
        say("\n\n  --> Final Fitting... \n", verbose)

        if ncomps_gf > 0:

            # Final fit using unconstrained parameters
            t0 = time.time()
            lmfit_params = paramvec_to_lmfit(params_gf)
            result2 = lmfit_minimize(objective_leastsq, lmfit_params, args=(vel,data,errors), method="leastsq")
            params_fit = vals_vec_from_lmfit(result2.params)
            params_errs = errs_vec_from_lmfit(result2.params)
            ncomps_fit = len(params_fit) // 3

            del lmfit_params
            say("Final fit took {0} seconds.".format(time.time() - t0), verbose)

            # # Make "FWHMS" positive
            # params_fit[0:ncomps_fit][params_fit[0:ncomps_fit] < 0.0] = (
            #     -1 * params_fit[0:ncomps_fit][params_fit[0:ncomps_fit] < 0.0]
            # )

            best_fit_final = func(vel, *params_fit).ravel()
            rchi2 = np.sum((data - best_fit_final) ** 2 / errors ** 2) / len(data)

            # Check if any amplitudes are identically zero, if so, remove them.
            if np.any(params_fit[0:ncomps_gf] == 0):
                amps_fit = params_fit[0:ncomps_gf]
                fwhms_fit = params_fit[ncomps_gf : 2 * ncomps_gf]
                offsets_fit = params_fit[2 * ncomps_gf : 3 * ncomps_gf]
                w_keep = amps_fit > 0.0
                params_fit = np.concatenate(
                    [amps_fit[w_keep], fwhms_fit[w_keep], offsets_fit[w_keep]]
                )
                ncomps_fit = len(params_fit) // 3
        else:
            best_fit_final = np.zeros(len(vel))
            rchi2 = -999.0
            ncomps_fit = ncomps_gf

    # if plot:
    #     #                       P L O T T I N G
    #     datamax = np.max(data)
    #
    #     # Set up figure
    #     fig = plt.figure("AGD results", [12, 12])
    #     ax1 = fig.add_axes([0.1, 0.5, 0.4, 0.4])  # Initial guesses (alpha1)
    #     ax2 = fig.add_axes([0.5, 0.5, 0.4, 0.4])  # D2 fit to peaks(alpha2)
    #     ax3 = fig.add_axes([0.1, 0.1, 0.4, 0.4])  # Initial guesses (alpha2)
    #     ax4 = fig.add_axes([0.5, 0.1, 0.4, 0.4])  # Final fit
    #
    #     # Decorations
    #     plt.figtext(0.52, 0.47, "Final fit")
    #     # if perform_final_fit:
    #     #     plt.figtext(0.52, 0.45, "Reduced Chi2: {0:3.1f}".format(rchi2))
    #     #     plt.figtext(0.52, 0.43, "N components: {0}".format(ncomps_fit))
    #
    #     plt.figtext(0.12, 0.47, "Phase-two initial guess")
    #     plt.figtext(0.12, 0.45, "N components: {0}".format(ncomps_g2))
    #
    #     plt.figtext(0.12, 0.87, "Phase-one initial guess")
    #     plt.figtext(0.12, 0.85, "N components: {0}".format(ncomps_g1))
    #
    #     plt.figtext(0.52, 0.87, "Intermediate fit")
    #
    #     # Initial Guesses (Panel 1)
    #     # -------------------------
    #     ax1.xaxis.tick_top()
    #     u2_scale = 1.0 / np.max(np.abs(u2)) * datamax * 0.5
    #     ax1.plot(vel, data, "-k")
    #     ax1.plot(vel, u2 * u2_scale, "-r")
    #     ax1.plot(vel, vel / vel * agd1["thresh"], "-k")
    #     ax1.plot(vel, vel / vel * agd1["thresh2"] * u2_scale, "--r")
    #
    #     for i in range(ncomps_g1):
    #         one_component = gaussian(
    #             params_g1[i], params_g1[i + ncomps_g1], params_g1[i + 2 * ncomps_g1]
    #         )(vel)
    #         ax1.plot(vel, one_component, "-g")
    #
    #     # Plot intermediate fit components (Panel 2)
    #     # ------------------------------------------
    #     ax2.xaxis.tick_top()
    #     ax2.plot(vel, data, "-k")
    #     ax2.yaxis.tick_right()
    #     for i in range(ncomps_f1):
    #         one_component = gaussian(
    #             params_f1[i], params_f1[i + ncomps_f1], params_f1[i + 2 * ncomps_f1]
    #         )(vel)
    #         ax2.plot(vel, one_component, "-", color="blue")
    #
    #     # Residual spectrum (Panel 3)
    #     # -----------------------------
    #     if phase == "two":
    #         u22_scale = 1.0 / np.abs(u22).max() * np.max(residuals) * 0.5
    #         ax3.plot(vel, residuals, "-k")
    #         ax3.plot(vel, vel / vel * agd2["thresh"], "--k")
    #         ax3.plot(vel, vel / vel * agd2["thresh2"] * u22_scale, "--r")
    #         ax3.plot(vel, u22 * u22_scale, "-r")
    #         for i in range(ncomps_g2):
    #             one_component = gaussian(
    #                 params_g2[i], params_g2[i + ncomps_g2], params_g2[i + 2 * ncomps_g2]
    #             )(vel)
    #             ax3.plot(vel, one_component, "-g")
    #
    #     # Plot best-fit model (Panel 4)
    #     # -----------------------------
    #     if perform_final_fit:
    #         ax4.yaxis.tick_right()
    #         ax4.plot(vel, best_fit_final, label="final model", color="purple")
    #         ax4.plot(vel, data, label="data", color="black")
    #         for i in range(ncomps_fit):
    #             one_component = gaussian(
    #                 params_fit[i],
    #                 params_fit[i + ncomps_fit],
    #                 params_fit[i + 2 * ncomps_fit],
    #             )(vel)
    #             ax4.plot(vel, one_component, "-", color="purple")
    #         ax4.plot(vel, best_fit_final, "-", color="purple")
    #
    #     plt.show()
    #     plt.close()

    # Construct output dictionary (odict)
    # -----------------------------------
    odict = {}
    odict["initial_parameters"] = params_gf
    odict["N_components"] = ncomps_gf

    if (perform_final_fit) and (ncomps_gf > 0):
        odict["best_fit_parameters"] = params_fit
        odict["best_fit_errors"] = params_errs
        odict["rchi2"] = rchi2

    return (1, odict)


def AGD_double(
    vel,
    data,
    vel_em,
    data_em,
    errors,
    errors_em,
    alpha1=None,
    alpha2=None,
    alpha_em=None,
    max_tb=None,
    p_width=0.1,
    d_mean=2,
    drop_width=3,
    min_dv=0,
    mode="python",
    verbose=False,
    SNR_thresh=5.0,
    BLFrac=0.1,
    SNR2_thresh=5.0,
    SNR_em=5.0,
    deblend=True,
    perform_final_fit=True,
    phase="two",
):
    """Autonomous Gaussian Decomposition "hybrid" Method
    Allows for simultaneous decomposition of 21cm absorption and emission
    New free parameters, in addition to the alpha parameters and
    signal to noise ratios for standard one- or two-phase AGD fit, include:
        alpha_em:     regularization parameter for the fit to emission (either
                        the residuals of the absorption ,or the full emission
                        spectrum in the absence of detected absorption components)
                        Default: None
        max_tb:       If 'max', the maximum brightness temperature of absorption
                        components will be fixed to be no larger than Tb computed
                        from the maximum kinetic temperature. If a value, the maximum Tb will be
                        set to this value. If None, no limit imposed.
                        Default: None
        p_width:      +/- percentage by which the width fitted to absorption fit is
                        allowed to vary in the fit to emission. i.e., if set to 0.1,
                        the widths may vary by +/-10%. If set to zero, a
                        minimum percentage is applied (0.1%).
                        Default: 10%
        d_mean:       +/- absolute number of channels by which the mean positions of
                        the absorption fits are allowed to vary in the fit
                        to emission. i.e., if set to 2, the mean positions
                        are allowed to vary by +/-2 channels. If set to zero,
                        a minimum value is applied (0.001).
                        Default: 2
        drop_width:   if a component is found in the fit to emission and its mean
                        position is within +/- drop_width (defind in channels)
                        from the mean position of any absorption component, the
                        emission component is dropped from the fit.
                        Default: 3 channels
        min_dv:       minimum width of components fit in the emission-only fit.
                        Limits AGD from fitting unrealistically narrow components
                        in emission which, if real, shoud have been detected in
                        absorption.
                        Default: 0 (i.e., no constraint)
    """
    if type(SNR2_thresh) != type([]):
        SNR2_thresh = [SNR2_thresh, SNR2_thresh]
    if type(SNR_thresh) != type([]):
        SNR_thresh = [SNR_thresh, SNR_thresh]

    say("\n  --> AGD() \n", verbose)

    if (not alpha2) and (phase == "two"):
        print("alpha2 value required")
        return

    dv = np.abs(vel[1] - vel[0])
    v_to_i = interp1d(vel, np.arange(len(vel)))

    # --------------------------------------#
    # Find phase-one guesses               #
    # --------------------------------------#
    agd1 = initialGuess(
        vel,
        data,
        errors=errors,
        alpha=alpha1,
        # plot=plot,
        mode=mode,
        verbose=verbose,
        SNR_thresh=SNR_thresh[0],
        BLFrac=BLFrac,
        SNR2_thresh=SNR2_thresh[0],
        deblend=deblend,
    )
    #width opacity guesses are put in here
    amps_g1, widths_g1, offsets_g1, u2 = (
        agd1["amps"],
        agd1["FWHMs"],
        agd1["means"],
        agd1["u2"],
    )
    params_g1 = np.append(np.append(amps_g1, widths_g1), offsets_g1)
    ncomps_g1 = len(params_g1) // 3
    ncomps_g2 = 0  # Default
    ncomps_f1 = 0  # Default

    # ----------------------------#
    # Find phase-two guesses #
    # ----------------------------#
    if phase == "two":
        say("Beginning phase-two AGD... ", verbose)
        ncomps_g2 = 0

        # ----------------------------------------------------------#
        # Produce the residual signal                               #
        #  -- Either the original data, or intermediate subtraction #
        # ----------------------------------------------------------#
        if ncomps_g1 == 0:
            say(
                "Phase 2 with no narrow comps -> No intermediate subtration... ",
                verbose,
            )
            residuals = data
        else:
            # "Else" Narrow components were found, and Phase == 2, so perform intermediate subtraction...

            # The "fitmask" is a collection of windows around the a list of phase-one components
            fitmask, fitmaskw = create_fitmask(
                len(vel), v_to_i(offsets_g1), widths_g1 / dv / 2.355 * 0.9
            )
            notfitmask = 1 - fitmask

            # Error function for intermediate optimization
            def objectiveD2_leastsq(paramslm):
                params = vals_vec_from_lmfit(paramslm)
                model0 = func(vel, *params)
                model2 = np.diff(np.diff(model0.ravel())) / dv / dv
                resids1 = fitmask[1:-1] * (model2 - u2[1:-1]) / errors[1:-1]
                resids2 = notfitmask * (model0 - data) / errors / 10.0
                return np.append(resids1, resids2)

            # Perform the intermediate fit using LMFIT
            t0 = time.time()
            say("Running LMFIT on initial narrow components...", verbose)
            lmfit_params = paramvec_to_lmfit(params_g1)
            result = lmfit_minimize(objectiveD2_leastsq, lmfit_params, method="leastsq")
            params_f1 = vals_vec_from_lmfit(result.params)
            ncomps_f1 = len(params_f1) // 3

            # # Make "FWHMS" positive
            # params_f1[0:ncomps_f1][params_f1[0:ncomps_f1] < 0.0] = (
            #     -1 * params_f1[0:ncomps_f1][params_f1[0:ncomps_f1] < 0.0]
            # )

            del lmfit_params
            say("LMFIT fit took {0} seconds.".format(time.time() - t0))

            if result.success:
                # Compute intermediate residuals
                # Median filter on 2x effective scale to remove poor subtractions of strong components
                intermediate_model = func(
                    vel, *params_f1
                ).ravel()  # Explicit final (narrow) model
                median_window = 2.0 * 10 ** ((np.log10(alpha1) + 2.187) / 3.859)
                residuals = median_filter(
                    data - intermediate_model, np.int(median_window)
                )
            else:
                residuals = data
            # Finished producing residual signal # ---------------------------
        #print(f'residuals = {residuals}')
        # Search for phase-two guesses
        agd2 = initialGuess(
            vel,
            residuals,
            errors=errors,
            alpha=alpha2,
            mode=mode,
            verbose=verbose,
            SNR_thresh=SNR_thresh[1],
            BLFrac=BLFrac,
            SNR2_thresh=SNR2_thresh[1],  # June 9 2014, change
            deblend=deblend,
            # plot=plot,
        )
        #print(f'AGD2 = {agd2}')
        ncomps_g2 = agd2["N_components"]
        if ncomps_g2 > 0:
            params_g2 = np.concatenate([agd2["amps"], agd2["FWHMs"], agd2["means"]])
        else:
            params_g2 = []
        u22 = agd2["u2"]

        # END PHASE 2 <<<

    # Check for phase two components, make final guess list
    # ------------------------------------------------------
    if phase == "two" and (ncomps_g2 > 0):
        amps_gf = np.append(params_g1[0:ncomps_g1], params_g2[0:ncomps_g2])
        widths_gf = np.append(
            params_g1[ncomps_g1 : 2 * ncomps_g1], params_g2[ncomps_g2 : 2 * ncomps_g2]
        )
        offsets_gf = np.append(
            params_g1[2 * ncomps_g1 : 3 * ncomps_g1],
            params_g2[2 * ncomps_g2 : 3 * ncomps_g2],
        )
        params_gf = np.concatenate([amps_gf, widths_gf, offsets_gf])
        ncomps_gf = len(params_gf) // 3
    else:
        params_gf = params_g1
        ncomps_gf = len(params_gf) // 3

    # Sort final guess list by amplitude
    # ----------------------------------
    say("N final parameter guesses: " + str(ncomps_gf))
    amps_temp = params_gf[0:ncomps_gf]
    widths_temp = params_gf[ncomps_gf : 2 * ncomps_gf]
    offsets_temp = params_gf[2 * ncomps_gf : 3 * ncomps_gf]
    w_sort_amp = np.argsort(amps_temp)[::-1]
    params_gf = np.concatenate(
        [amps_temp[w_sort_amp], widths_temp[w_sort_amp], offsets_temp[w_sort_amp]]
    )

    ncomps_fit = ncomps_gf
    params_fit = params_gf

    if perform_final_fit:
        say("\n\n  --> Final Fitting... \n", verbose)

        if ncomps_gf > 0:

            # Final fit using unconstrained parameters
            t0 = time.time()
            lmfit_params = paramvec_to_lmfit(params_gf)
            result2 = lmfit_minimize(objective_leastsq, lmfit_params, args=(vel,data,errors), method="leastsq")
            params_fit = vals_vec_from_lmfit(result2.params)
            params_errs = errs_vec_from_lmfit(result2.params)
            ncomps_fit = len(params_fit) // 3

            del lmfit_params
            say("Final fit took {0} seconds.".format(time.time() - t0), verbose)

            # # Make "FWHMS" positive
            # params_fit[0:ncomps_fit][params_fit[0:ncomps_fit] < 0.0] = (
            #     -1 * params_fit[0:ncomps_fit][params_fit[0:ncomps_fit] < 0.0]
            # )

            best_fit_final = func(vel, *params_fit).ravel()
            rchi2 = np.sum((data - best_fit_final) ** 2 / errors ** 2) / len(data)

            # Check if any amplitudes are identically zero, if so, remove them.
            print(f'original value = {params_fit[0:ncomps_fit]}')
            print(f'as np array = {np.array(params_fit[0:ncomps_fit], dtype=float)}')
            amps_fit = np.array(params_fit[0:ncomps_fit], dtype=float)
            fwhms_fit = np.array(params_fit[ncomps_fit : 2 * ncomps_fit], dtype=float)
            offsets_fit = np.array(
                params_fit[2 * ncomps_fit : 3 * ncomps_fit], dtype=float
            )
            w_keep = amps_fit > 0.0
            params_fit = np.concatenate(
                [amps_fit[w_keep], fwhms_fit[w_keep], offsets_fit[w_keep]]
            )
            ncomps_fit = len(params_fit) // 3

            # Check if any mean velocities are >/< min and max velocities, if so, drop them.
            amps_fit = np.array(params_fit[0:ncomps_fit], dtype=float)
            fwhms_fit = np.array(params_fit[ncomps_fit : 2 * ncomps_fit], dtype=float)
            offsets_fit = np.array(
                params_fit[2 * ncomps_fit : 3 * ncomps_fit], dtype=float
            )
            w_keep = offsets_fit < np.max(vel)
            params_fit = np.concatenate(
                [amps_fit[w_keep], fwhms_fit[w_keep], offsets_fit[w_keep]]
            )
            fwhms_fit = np.abs(fwhms_fit[w_keep])
            ncomps_fit = len(params_fit) // 3

            amps_fit = np.array(params_fit[0:ncomps_fit], dtype=float)
            fwhms_fit = np.array(params_fit[ncomps_fit : 2 * ncomps_fit], dtype=float)
            offsets_fit = np.array(
                params_fit[2 * ncomps_fit : 3 * ncomps_fit], dtype=float
            )
            w_keep = offsets_fit > np.min(vel)
            params_fit = np.concatenate(
                [amps_fit[w_keep], fwhms_fit[w_keep], offsets_fit[w_keep]]
            )
            fwhms_fit = np.abs(fwhms_fit[w_keep])
            ncomps_fit = len(params_fit) // 3

            # Check if any widths are less than 3 channels in width, if so, drop them
            amps_fit = np.array(params_fit[0:ncomps_fit], dtype=float)
            fwhms_fit = np.array(params_fit[ncomps_fit : 2 * ncomps_fit], dtype=float)
            offsets_fit = np.array(
                params_fit[2 * ncomps_fit : 3 * ncomps_fit], dtype=float
            )
            w_keep = fwhms_fit > dv
            params_fit = np.concatenate(
                [amps_fit[w_keep], fwhms_fit[w_keep], offsets_fit[w_keep]]
            )
            ncomps_fit = len(params_fit) // 3
        else:
            best_fit_final = np.zeros(len(vel))
            rchi2 = -999.0
            ncomps_fit = ncomps_gf

    # Construct output dictionary (odict)
    # -----------------------------------
    odict = {}
    odict["initial_parameters"] = params_gf
    odict["N_components"] = ncomps_fit

    if (perform_final_fit) and (ncomps_fit > 0):
        odict["best_fit_parameters"] = params_fit
        odict["best_fit_errors"] = params_errs
        odict["rchi2"] = rchi2
    else:
        odict["best_fit_parameters"] = []
        odict["best_fit_errors"] = []
        odict["rchi2"] = []

    # ------ ADDING SUBSEQUENT FIT FOR EMISSION-ONLY COMPONENTS ---------
    # -- Find initial guesses for fit of absorption components to emission
    # -- Based on simple least-squares fit of absorption lines to emission
    # -- **Holding width and mean velocity constant for absorption lines
    # -- **Fitting only for amplitude (i.e., spin temperature)

    vel = vel_em
    data = data_em
    errors = errors_em
    dv = np.abs(vel[1] - vel[0])
    v_to_i = interp1d(vel, np.arange(len(vel)))

    params_em = []
    params_em_errs = []
    ncomps_em = 0
    if ncomps_fit > 0:

        params_full = np.concatenate( #i think this is em_amps(same as tau),em_widths,em_pos,labels,tau
            [params_fit, np.ones(ncomps_fit), params_fit[0:ncomps_fit]]
        )
        print("abs amps going into emission fit", params_fit[0:ncomps_fit])

        # Initial fit using constrained parameters
        t0 = time.time()

        #pass in abs pos and means to stop drift of values when allowing deviation from the emission values
        lmfit_params = paramvec_p3_to_lmfit(
            params_full, max_tb, p_width, d_mean, min_dv, abs_widths=fwhms_fit[w_keep], abs_pos=offsets_fit[w_keep]
        )
        result_em = lmfit_minimize(objective_leastsq, lmfit_params, args=(vel,data,errors), method="leastsq")
        
        amp_dict=param_extract(result_em.params,"a")
        width_dict=param_extract(result_em.params,"w")
        pos_dict=param_extract(result_em.params,"p")
        delta_dict=param_extract(result_em.params,"d")

        #print(f'result_em.params = {result_em.params}')
        #print(amp_dict)
        #print(width_dict)
        #print(pos_dict)

        params_em = vals_vec_from_lmfit(result_em.params)
        params_em_amp = vals_vec_from_lmfit(amp_dict)
        params_em_width = vals_vec_from_lmfit(width_dict)
        params_em_pos = vals_vec_from_lmfit(pos_dict)
        params_em_delta = vals_vec_from_lmfit(delta_dict)

        print(params_em)
        print(params_em_pos)
        params_em_nodelta = np.concatenate([ #not sure if this will be useful as delta is not a superfluous parameter
            params_em_amp,
            params_em_width,
            params_em_pos])

        print(f'params_em = {params_em}')
        params_em_errs = errs_vec_from_lmfit(result_em.params)
        ncomps_em = len(amp_dict.keys()) #explicitly defines the length instead of inferring from length of params var which will change with new params
        

        # The "fitmask" is a collection of windows around the a list of two-phase absorption components
        fitmask, fitmaskw = create_fitmask(
            len(vel),
            v_to_i(params_em_pos),
            params_em_width / dv / 2.355 * 0.9,
        )#
        notfitmask = 1 - fitmask
        # notfitmaskw = np.logical_not(fitmaskw)

        # Compute intermediate residuals
        # Median filter on 2 x effective scale to remove poor subtractions of strong components
        intermediate_model = func(
            vel, *params_em_nodelta
        ).ravel()  # Explicit final (narrow) model
        median_window = 2.0 * 10 ** ((np.log10(alpha_em) + 2.187) / 3.859)
        residuals = median_filter(data - intermediate_model, np.int(median_window))

    else:
        residuals = data
        # Finished producing residual emission signal # ---------------------------

    #initial guesses won't have the delta and expression bounds that paramvec_p3_to_lmfit has but that probably isn't an issue for the guesses
    # Search for phase-three guesses in residual emission spectrum
    agd3 = initialGuess(
        vel,
        residuals,
        errors_em,
        alpha=alpha_em,
        mode=mode,
        verbose=verbose,
        SNR_thresh=SNR_em,
        BLFrac=BLFrac,
        SNR2_thresh=SNR_thresh[0],
        deblend=deblend,
    )

    ncomps_g3 = agd3["N_components"]
    # Check for phase three components, make final guess list
    # ------------------------------------------------------
    if ncomps_g3 > 0:
        abs_offsets = np.array(params_em_pos, dtype=float)
        em_amps = np.array(agd3["amps"], dtype=float)
        em_widths = np.array(agd3["FWHMs"], dtype=float)
        em_offsets = np.array(agd3["means"], dtype=float)

        indices = []
        # Check if any emission components are within 3 channels of an absorption component
        if ncomps_fit > 0:
            for i, offset in enumerate(em_offsets):
                drop_comp = False
                for abs_offset in abs_offsets:
                    if np.abs(abs_offset - offset) < drop_width * dv:
                        # print(abs_offset, offset, 3.0 * dv)
                        drop_comp = True
                if not drop_comp:
                    indices.append(i)

            em_offsets = em_offsets[indices]
            em_amps = em_amps[indices]
            em_widths = em_widths[indices]
            ncomps_g3 = len(em_amps)

            # print("ncomps_em", ncomps_em)
            amps_emf = np.append(params_em_amp, em_amps)
            widths_emf = np.append(params_em_width, em_widths)
            offsets_emf = np.append(
                params_em_pos, em_offsets
            )
            tau_emf = np.append(params_fit[0:ncomps_em], np.zeros(ncomps_g3))
            labels_emf = np.append(np.ones(ncomps_em), np.zeros(ncomps_g3))
            # print("labels", labels_emf)
        else:
            amps_emf = em_amps
            widths_emf = em_widths
            offsets_emf = em_offsets
            tau_emf = np.zeros(ncomps_g3)
            labels_emf = np.zeros(ncomps_g3)
        params_emf = np.concatenate([amps_emf, widths_emf, offsets_emf])
        ncomps_emf = len(amps_emf)
    else:
        params_emf = params_em
        ncomps_emf = ncomps_em
        if ncomps_em > 0:
            tau_emf = params_fit[0:ncomps_em]
            labels_emf = np.ones(ncomps_em)
        else:
            tau_emf = []
            labels_emf = []

    params_emfit = []
    params_emfit_errs = []
    if ncomps_emf > 0:
        say("\n\n  --> Final Fitting... \n", verbose)

        # Compile parameters, labels, and original optical depths for final fit:
        params_full = np.concatenate([params_emf, labels_emf, tau_emf])

        # Final fit using constrained parameters
        t0 = time.time()
        lmfit_params = paramvec_p3_to_lmfit(
            params_full, max_tb, p_width, d_mean, min_dv, abs_widths=fwhms_fit[w_keep], abs_pos=offsets_fit[w_keep]
        )
        result3 = lmfit_minimize(objective_leastsq, lmfit_params, args=(vel,data,errors), method="leastsq")
        print(result3.params)
        params_emfit = vals_vec_from_lmfit(result3.params)
        params_emfit_amp = vals_vec_from_lmfit(param_extract(result3.params,'a'))
        params_emfit_width = vals_vec_from_lmfit(param_extract(result3.params,'w'))
        params_emfit_pos = vals_vec_from_lmfit(param_extract(result3.params,'p'))
        params_emfit_delta = vals_vec_from_lmfit(param_extract(result3.params,'d'))

        print(f'params_emfit = {params_emfit}')
        params_emfit_errs = errs_vec_from_lmfit(result3.params)
        ncomps_emfit = len(params_emfit_amp)
        print(f'ncomps_emfit = {ncomps_emfit}')
        del lmfit_params
        say("Final fit took {0} seconds.".format(time.time() - t0), verbose)

        params_emfit_nodelta = np.concatenate([ #not sure if this will be useful as delta is not a superfluous parameter
            params_emfit_amp,
            params_emfit_width,
            params_emfit_pos])

        best_fit_final = func(vel, *params_emfit_nodelta).ravel()
        rchi2 = np.sum((data - best_fit_final) ** 2 / errors ** 2) / len(data)
    else:
        ncomps_emfit = ncomps_emf

    # Construct output dictionary (odict)
    # -----------------------------------
    # print(
    #     "NUMBER ABS",
    #     ncomps_fit,
    #     " NUMBER EM",
    #     len(labels_emf[labels_emf == 1]),
    #     len(params_emfit) // 3,
    #     ncomps_emfit,
    # )
    # print(params_emfit)
    odict["N_components_em"] = ncomps_emfit
    if ncomps_emfit >= ncomps_fit:
        odict["best_fit_parameters_em"] = params_emfit
        odict["best_fit_errors_em"] = params_emfit_errs
        odict["fit_labels"] = labels_emf
    else:
        odict["best_fit_parameters_em"] = []
        odict["best_fit_errors_em"] = []
        odict["fit_labels"] = []
    print(odict.keys())
    return (1, odict)

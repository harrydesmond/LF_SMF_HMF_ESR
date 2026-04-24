"""HMF analysis pipeline: identify, refit, and visualise top ESR functions.

This script has three stages, selected via a command-line argument:

  step1  Identify the top 500 unique ESR functions from comprehensive fits
         on 10 Quijote simulations (0, 10, 20, ..., 90), ranked by average
         position across simulations. Output: top_500_all.txt.

  step2  Refit the top 200 of those functions to all 100 Quijote simulations
         (parallelised over simulations with MPI). Output: per-simulation
         hmf_<sim>_data/final_all.txt files.

  step3  Generate publication plots from the step2 results:
         stacked bar chart, DL scatter, and box plots.

Usage:
    python3 sample_top_200.py step1
    mpirun -np <N> python3 sample_top_200.py step2
    python3 sample_top_200.py step3

Inputs:
    - step1: hmf_data/hmf_<sim>_data/final_<comp>_new.dat (from fit_all.py)
    - step2: top_500_all.txt, hmf_files/hmf_<sim>.dat
    - step3: hmf_data/hmf_<sim>_data/final_all.txt

Outputs:
    - top_500_all.txt
    - hmf_data/hmf_<sim>_data/final_all.txt (per simulation)
    - Final_Plots/stacked_rank.pdf, DL_scatter.pdf, box_plot.pdf

Dependencies:
    numpy, matplotlib, mpi4py, prettytable, pytexit, sympy, scipy,
    esr (https://github.com/DeaglanBartlett/ESR)
"""

import sys
# Update this path to your local ESR installation
# sys.path.insert(0, '/path/to/ESR')

import numpy as np
import os
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import string
from pytexit import py2tex
from sympy import N
#import seaborn as sns
from scipy.stats import gaussian_kde


from esr.fitting.likelihood import PoissonLikelihood
from esr.fitting.fit_single import fit_from_string, single_function

# Update this to your ESR data directory (where hmf_files/ etc. are located)
DATA_DIR = '.'

from mpi4py import MPI
comm = MPI.COMM_WORLD
global rank
rank = comm.Get_rank()


# --- For colours on graph function lines --- #
global cmap
cmap = cm.winter


# --- Fit individual equations --- #
def fit_str(string,hmf_sim):
    basis_functions = [["x", "a"],  # type0
                ["inv","exp","log","abs"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2

    nparam = 0
    for i in range(0,4):
        if 'a{}'.format(i) in string:
            nparam += 1
            continue

    Niter_params=[3040,3060]
    Nconv_params=[-5,20]
    Niter_new = int(np.sum(nparam ** np.arange(len(Niter_params)) * np.array(Niter_params)))
    Nconv_new = int(np.sum(nparam ** np.arange(len(Nconv_params)) * np.array(Nconv_params)))

    if (Nconv_new <= 0) or (Niter_new <= 0) or (Nconv_new > Niter_new):
        raise ValueError("Nconv and/or Niter have unacceptable values")

    likelihood = PoissonLikelihood('data/hmf_files/hmf_{}.dat'.format(hmf_sim), 'poisson_example', data_dir=DATA_DIR, fn_set='base_e_maths')

    # --- Allows multiple attempts at fitting in case it fails to converge (max 20 attempts) --- #
    count = 0
    done = False
    while count < 20 and done == False:
        count += 1
        
        logl_lcdm, dl_lcdm, tree, params = fit_from_string(string,
                                                basis_functions,
                                                likelihood,
                                                verbose=False,
                                                Niter=Niter_new,
                                                Nconv=Nconv_new,
                                                return_params=True)

        try:
            a = float(logl_lcdm)
            b = float(dl_lcdm)
            done = True
        except:
            pass

    if done == False:
        return 'naan', 'naan', params


    return logl_lcdm, dl_lcdm, params



# --- A function to check the lengths of the comp 10 output files --- #
def check_final_length(hmf_sims):

    lengths = []
    hmf_sims = []
    for hmf_sim in range(0,100):
        try:
            # Extract data
            comp = '10'
            ranking, function, DL, prel, nll, codelen, aifeyn, a0, a1, a2, a3 = np.loadtxt(os.path.join(DATA_DIR, "hmf_{0}_data/final_{1}_sample.dat".format(hmf_sim,comp)), unpack=True, dtype=str, delimiter=";")
            # ranking, function, DL, prel, nll, codelen, aifeyn, a0, a1, a2, a3 = np.loadtxt("hmf_{0}_data/output/output_poisson_example/final_{1}.dat".format(hmf_sim,comp), unpack=True, dtype=str, delimiter=";")
            
            lengths.append(len(ranking))
            hmf_sims.append(hmf_sim)

        except:
            pass

    plt.plot(hmf_sims, lengths)
    plt.xlabel("HMF simulation number")
    plt.ylabel("Length of output file")
    plt.savefig("length.png",bbox_inches='tight',dpi=400)
    plt.clf()


# --- Formats functions to have a generic format, such to better identify any duplicates --- #
def format_function(fcn):
    fcn = fcn.replace('a0','C').replace('a1','C').replace('a2','C').replace('a3','C')
    fcn = fcn.replace('exp(C)','C').replace('1/C','C')
    fcn = fcn.replace('Abs(C + x)', 'Abs(C - x)') # These are identical, and sometimes duplicates slip through... Typically the fcns using C-have C=+ve
    fcn = fcn.replace('log(Abs(C))', 'C')
    fcn = fcn.replace(',(C)',',C')
    fcn = fcn.replace('Abs(1/C)','Abs(C)')

    # Ensures all constants are in the order a0, a1, a2, a3, in case they were generated in different orders
    const = 0
    new_fcn = ''
    for char in fcn:
        if char == 'C':
            new_fcn+='a{}'.format(const)
            const += 1
        else:
            new_fcn+=char

    # Removes duplicates, specifically negative constants
    fcn = new_fcn.replace('-a','a')
    
    return fcn


# --- A function to ensure all functions being assessed were always successfully fitted --- #
# I don't think this was used in the end... but useful to keep just in case.
def remove_non_generalised(zipped, hmf_sims):
    format_zipped = []

    for row in zipped:

        if len(row[1].split(';')) != len(hmf_sims): # If this function has not come up in each simulation top 200...
            if rank == 1:
                print('Not in every simulation: ', row[0])
                print(row[1:])
            pass # Do nothing
        else:
            format_zipped.append(row)

    return format_zipped


# --- Averages the values corresponding to one function across all simulations --- #
def average_values(zipped, val_idx):
    format_zipped = []

    for row in zipped:
        # Isolate values and convert to a float list of values, then average
        values = row[val_idx].split(';')
#        values = list(map(float,values))
        values = np.array(values).astype(float)

        if val_idx == 1: # DL...
            values -= min(values)

        av_val = np.average(values)

        # Make new row (amending old doesn't work: row[val_idx] = av_val)
        format_row = []
        for i in range(0,len(row)):
            format_row.append(row[i])
        format_row.append(av_val)

        if len(values) > 5:
            # Add to formatted zipped array
            format_zipped.append(format_row)

    return format_zipped



def run_fits(hmf_sim):

    print()
    print('Rank {}, HMF {}:'.format(rank, hmf_sim))

    # --- Fitting individually --- #

    # Selects functions to fit
    with open('top_500_all.txt', 'r') as file:
        function = [line.rstrip('\n') for line in file]

    function = function[:200] # Let's just handle the top 200 for now - you can fit more later if needed

    data, all_funcs = [], [] # all_funcs double checks that the function is unique
    for idx, func in enumerate(function):

        # Fits each function from the top 200
        logl_lcdm, dl_lcdm, params = fit_str(func,hmf_sim)

        # If fitted correctly and not yet fitted, add to data
        if (not np.isnan(dl_lcdm)) and (func not in all_funcs):
            data.append([func, dl_lcdm, logl_lcdm, str(params)])
            all_funcs.append(func)
        else:
            print(f"Rank {rank}, skipping due to nan or not unique: {func}")

        # Displays progress
        if idx%10 == 0:
            print('Rank {}, HMF {}: {}/200'.format(rank, hmf_sim, idx))
    # --- End of fitting individually --- #

    # --- Saves these fits into hmf_{}_data/ --- #
    sorted_combined = sorted(data, key=lambda x: x[1], reverse=False)
    fcn_sorted, DL_sorted, NLL_sorted, params_sorted = zip(*sorted_combined)

    # Convert to lists (zip returns tuples)
    function = np.array(list(fcn_sorted))
    DL = np.array(list(DL_sorted))
    NLL = np.array(list(NLL_sorted))
    params = np.array(list(params_sorted))
    ranking = np.arange(0,len(function),1)

    with open('hmf_{}_data/final_all.txt'.format(hmf_sim), 'w') as file:
        for idx in ranking:
            line = str(idx) + ';' + function[idx] + ';' + str(DL[idx]) + ';' + str(NLL[idx]) + ';' + str(params[idx])
            file.write(line)
            file.write('\n')

    print('Rank {}, HMF {} saved!'.format(rank, hmf_sim))


# --- Identifies the top 200 functions across all comprehensive fits --- #
def get_uniq_top_200(hmf_sims):

    # --- Identifying duplicates --- #
    flagged = {}

    for hmf_sim in hmf_sims:

        try:
            unique_NLL = {}
            function, DL, NLL = np.array([]), np.array([]), np.array([])

            # --- Compile data from complexities --- #
            for comp in range(10,3,-1):

                comp_str = str(comp) + '_new' # Accesses most recent fits

                ranking_comp, function_comp, DL_comp, NLL_comp = np.loadtxt(os.path.join(DATA_DIR, "hmf_{0}_data/final_{1}.dat".format(hmf_sim,comp_str)), unpack=True, dtype=str, delimiter=";", usecols=(0,1,2,4), max_rows=1500)

                function = np.concatenate((function, function_comp))
                DL = np.concatenate((DL, DL_comp))
                NLL = np.concatenate((NLL, NLL_comp))

            # --- Groups function by NLL --- #
            for idx, NLL_i in enumerate(NLL):
                fcn = format_function(function[idx])

                if (np.isfinite(float(DL[idx])) == False) or (np.isfinite(float(NLL[idx])) == False) or ('re' in function[idx]) or (float(DL[idx]) > 0) or (float(NLL[idx]) > 0):
                    # Not optimised or inacceptable function.
                    continue

                # - Ensures there's a record for each function - #
                # (This handles functions not flagged at all)    #
                if fcn not in flagged:
                    flagged[fcn] = False
                # ---------------------------------------------- #

                # - Uses NLL to group fcns --------------------- #
                if NLL_i not in unique_NLL:
                    unique_NLL[NLL_i] = [fcn]
                else:
                    unique_NLL[NLL_i].append(fcn)
                # ---------------------------------------------- #

            # --- Flags functions so that there is max one function in each not flagged --- #
            for idx in unique_NLL: # This grouped functions...
                found = False # This will contain the not flagged function! (The identifying one)
                flagged_fcns = []
                for fcn in unique_NLL[idx]:
                    if flagged[fcn] == False and found == False:
                        found = fcn # Unflagged and not yet found = identifier
                    else: # Either unflagged duplicates, or already flagged functions = duplicate
                        flagged_fcns.append(fcn)

                # --- Ensures all flagged functions are given the same function format --- #
                if found != False:
                    for fcn in flagged_fcns:
                        flagged[fcn] = found
        except:
            pass

#    # ------------- Test ------------- #
#    if rank == 1:
#        for key in list(flagged.keys())[:10]:
#            print(flagged[key])
#    # -------------------------------- #


    unique_best = {}

    for hmf_sim in hmf_sims:
        if rank == 1:
            print(hmf_sim)
        try:
            function, DL, NLL, comp_list = np.array([]), np.array([]), np.array([]), np.array([])

            # --- Compile the complexities --- #
            # Unnecessary to go below comp 4
            for comp in range(10,3,-1):

                comp_str = str(comp) + '_new' # Accesses most recent fits

                ranking, DL_comp, NLL_comp = np.loadtxt(os.path.join(DATA_DIR, "hmf_{0}_data/final_{1}.dat".format(hmf_sim,comp_str)), unpack=True, dtype=float, delimiter=";", usecols=(0,2,4), max_rows=1500)
                function_comp = np.loadtxt(os.path.join(DATA_DIR, "hmf_{0}_data/final_{1}.dat".format(hmf_sim,comp_str)), unpack=True, dtype=str, delimiter=";", usecols=(1), max_rows=1500)

                function = np.concatenate((function, function_comp))
                DL = np.concatenate((DL, DL_comp))
                NLL = np.concatenate((NLL, NLL_comp))
                comp_list = np.concatenate((comp_list, comp * np.ones(len(function_comp))))

            # --- Add the data using the flagged info --- #
            data = []
            for idx, NLL_i in enumerate(NLL):
                fcn = format_function(function[idx])
                
                if (np.isfinite(float(DL[idx])) == False) or (np.isfinite(float(NLL[idx])) == False) or ('re' in function[idx]) or (float(DL[idx]) > 0) or (float(NLL[idx]) > 0):
                    continue

                if flagged[fcn] == False:
                    data.append([fcn, DL[idx], NLL[idx], comp_list[idx]])

                else:
                    data.append([flagged[fcn], DL[idx], NLL[idx], comp_list[idx]])

            # --- Remove duplicates --- #
            # Order by DL
            sorted_combined = sorted(data, key=lambda x: x[1])

            seen = [] # Needed to select the highest (earliest) ranked appearance of the function.
            cleaned_data = []
            for idx, row in enumerate(sorted_combined):
                if row[0] not in seen:
                    cleaned_data.append(row)
                    seen.append(row[0])               

            fcn_sorted, DL_sorted, NLL_sorted, comp_sorted = list(zip(*cleaned_data))

            # Convert to lists (zip returns tuples)
            function = np.array(list(fcn_sorted))
            DL = np.array(list(DL_sorted)).astype(float)
            NLL = np.array(list(NLL_sorted)).astype(float)
            comp_list = np.array(list(comp_sorted)).astype(int)
            ranking = np.arange(0,len(function),1)
            

            # --- Now to compare across sims... --- #

            for idx, fcn in enumerate(function):
                if fcn in unique_best.keys():
                    unique_best[fcn][0].append(ranking[idx])
                    unique_best[fcn][1].append(DL[idx])
                    unique_best[fcn][2].append(NLL[idx])
                    unique_best[fcn][3].append(hmf_sim)
                else:
                    unique_best[fcn] = [[ranking[idx]],[DL[idx]],[NLL[idx]], [hmf_sim]]

        except:
            pass


    # --- Averages the ranking, DL and NLL values --- #
    # Not needed to average DL and NLL but a way to check if any crazy outliers have gotten involved.

    ordered_unique_best = []
    keys = list(unique_best.keys())

    for key in unique_best.keys():
        # Don't include a length condition here! Some good functions might not have fitted correctly, and such will not always be present!
        ordered_unique_best.append([np.average(unique_best[key][0]), key, np.average(unique_best[key][1]), np.average(unique_best[key][2])])


    # --- Order by ranking --- #
    sorted_combined = sorted(ordered_unique_best, key=lambda x: x[0], reverse=False)
    ranking_sorted, fcn_sorted, DL_sorted, NLL_sorted = zip(*sorted_combined)
    
    ranking = np.array(list(ranking_sorted)).astype(str)
    function = np.array(list(fcn_sorted)).astype(str)
    DL = np.array(list(DL_sorted)).astype(str)
    NLL = np.array(list(NLL_sorted)).astype(str)


    # --- Writes the top 500 functions for fitting --- #
    if rank == 1:
        with open('top_500_all.txt', 'w') as file:
            for idx, fcn in enumerate(function[:500]):
                #line = ranking[idx] + ';' + function[idx] + ';' + DL[idx] + ';' + NLL[idx]
                line = function[idx]
                file.write(line)
                file.write('\n')

                if idx < 10:
                    print(ranking[idx] + ';' + function[idx] + ';' + DL[idx] + ';' + NLL[idx])



# --- Gathers and sorts the fitted data --- #
def gather_barchart_data(hmf_sims):
    with open('top_500_all.txt', 'r') as file:
        function_all = [line.rstrip('\n') for line in file]

    function_all = function_all[:200]

    unique_best_DL_str = []
    unique_best_rank_str = []

    # Begin empty:
    for item in function_all:
        unique_best_DL_str.append('')
        unique_best_rank_str.append('')

    # Tally DL and rank to each function for each hmf sim data
    
    hmf_sims_actual = [] # In case any sims were not used for whatever reason... sometimes it was useful to only use a select few for convenience

    for hmf_sim in hmf_sims:
        try:
#        if True:
            
            ranking, function, DL, nll =  np.loadtxt("hmf_{0}_data/final_all.txt".format(hmf_sim), unpack=True, dtype=str, delimiter=";", usecols=(0,1,2,3))
            hmf_sims_actual.append(hmf_sim)

        except:
            continue


        if rank == 1:
            print()
            print("HMF {}".format(hmf_sim))

        # Checks local uniqueness of functions (removes duplicates to find correct ranking order)
        # This is done now, and not earlier, as the data would be lost from having to loop through again once the comprehensive unique list was compiled
        local_unique_fcns = []
        local_unique_DL = []
        for idx, fcn in enumerate(function[:500]):
            fcn = format_function(fcn)
            if fcn not in local_unique_fcns:
                local_unique_fcns.append(fcn)
                local_unique_DL.append(DL[idx])
        local_unique_ranking = np.arange(0,len(local_unique_fcns),1)

        if rank == 1:
            print('500 --> {}'.format(len(local_unique_fcns)))

        for idx_i, fcn_i in enumerate(function_all):
            found = False
            for idx_j, fcn_j in enumerate(local_unique_fcns):
                fcn_j = format_function(fcn_j)
                fcn_i = format_function(fcn_i)
                if fcn_i == fcn_j:
                    found = True
                    
                    if unique_best_DL_str[idx_i] == '': # Never been found before
                        unique_best_DL_str[idx_i] = str(local_unique_DL[idx_j])
                        unique_best_rank_str[idx_i] = str(local_unique_ranking[idx_j])
                    else: # Has been found before
                        unique_best_DL_str[idx_i] = unique_best_DL_str[idx_i] + ';' + str(local_unique_DL[idx_j])
                        unique_best_rank_str[idx_i] = unique_best_rank_str[idx_i] + ';' + str(local_unique_ranking[idx_j])
                    break
            if found == False:
                if rank == 1:
                    print("Not found: {}".format(unique_best_rank_str[idx_i]))
                    unique_best_DL_str[idx_i] = unique_best_DL_str[idx_i] + ';' + 'nan'
                    unique_best_rank_str[idx_i] = unique_best_rank_str[idx_i] + ';' + 'nan'

    zipped = zip(function_all, unique_best_DL_str, unique_best_rank_str)


    # zipped = remove_non_generalised(zipped, hmf_sims_actual)
    # This ^ was decided not to do...
   
    # Converts to a list... more useful later on
    zipped_list = []
    for row in zipped:
        zipped_list.append(row)

    return zipped_list, hmf_sims_actual




def create_all_plots(zipped, hmf_sims):
    data = []

    for row in zipped:
        gold, silver, bronze, fourth, fifth, sixth = 0, 0, 0, 0, 0, 0
        total = 0

        #for rank in row[1].replace(' [','').replace(']','').split(', '):
        for idx_rank, ranking in enumerate(row[2].split(';')):
            if ranking == '0':
                gold+=1
            elif ranking == '1':
                silver+=1
            elif ranking == '2':
                bronze+=1
            elif ranking == '3':
                fourth+=1
            elif ranking == '4':
                fifth+=1
            elif ranking == '5':
                sixth+=1
            else:
                pass

#        weighted = 6*gold + 5*silver + 4*bronze + 3*fourth + 2*fifth + 1*sixth
        total = gold + silver + bronze + fourth + fifth #+ sixth

        if gold==0 and silver==0 and bronze==0 and fourth==0 and fifth==0:# and sixth==0:
            pass
        else: 
#            data.append([gold, silver, bronze, fourth, fifth, sixth, total, row[1], row[0]])
            data.append([gold, silver, bronze, fourth, fifth, total, row[1], row[0]])

    ordered_gold = sorted(data, key = lambda x: x[-3])
    
    # Flip for appropriate order
    ordered_gold = list(reversed(ordered_gold))

    sub_categories = ['1st', '2nd', '3rd', '4th', '5th']#, '6th']

    # Define colors for each segment
    colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#6E7B8B', '#445577']#, '#7D1108']


    # --- Assessment of the functions --- #
    import sympy as sp
    x = sp.symbols('x', positive=True)
    a0, a1, a2, a3 = sp.symbols('a0 a1 a2 a3', real=True)
    
    ranking, function, DL, nll, params =  np.loadtxt("hmf_0_data/final_all.txt", unpack=True, dtype=str, delimiter=";", usecols=(0,1,2,3,4))
    
    for x_val, row in enumerate(ordered_gold):
        f = row[-1]
        idx = list(function).index(f)
        param = params[idx][1:-1].split(' ')
        param = [a for a in param if a != '']
        for p_idx, a in enumerate(param):
            f = f.replace('a{}'.format(p_idx), a)

        print(f"\nFunction: {f}")

    
    # --- Plot 1 --- #
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Create bottom array for stacking
    categories = len(ordered_gold)
    
    if categories > 26:
        ordered_gold = ordered_gold[:26]
        categories = len(ordered_gold)

    bottom = np.zeros(categories)
    x_ticks = list(string.ascii_uppercase)[:categories]
    x_plot = np.arange(0,categories,1)

    # Plot each segment
    for i, (color, sub_category) in enumerate(zip(colors, sub_categories)):
        y_plot = np.array([])
        for j in ordered_gold:
            y_plot = np.append(y_plot,j[i])

        ax.bar(x_plot, y_plot, bottom=bottom, label=sub_category, color=color)
        bottom += y_plot
    
    ax.set_xlabel('Functions')
    ax.set_xticks(x_plot, x_ticks)
    ax.set_ylabel('Number of Ranks Earned (in {} simulations)'.format(len(hmf_sims)))
    ax.legend(sub_categories)
    
    labels = []

    # Add function list to ax2
    for idx, row in enumerate(ordered_gold):
        text_fcn = py2tex(row[-1].replace('Abs','abs')).replace('$','').replace(' ','')
        text_fcn = text_fcn.replace('\\log', '\\ln')

        if r'e^{a0-{\frac{|a1|}{x}}^{a2-x}}' in text_fcn:
            text_fcn = text_fcn.replace(r'e^{a0-{\frac{|a1|}{x}}^{a2-x}}', r'e^{a0-\left({\frac{|a1|}{x}}\right)^{\left(a2-x\right)}}')
        elif r'{|a0|}^{e^{\frac{e^{x^{a1}}}{x}}}' in text_fcn:
            text_fcn = text_fcn.replace(r'{|a0|}^{e^{\frac{e^{x^{a1}}}{x}}}', r'{|a0|}^{\eXp{\left[\frac{e^{x^{a1}}}{x}\right]}}')

        text_fcn = text_fcn.replace('a0',r'\theta_0').replace('a1',r'\theta_1').replace('a2',r'\theta_2').replace('a3',r'\theta_3').replace('x',r'\sigma').replace('eXp','exp')

        text_fcn = text_fcn.replace('e^', ' e^')
        
        text = '{}: '.format(x_ticks[idx]) + r'${}$'.format(text_fcn)

        labels.append(text_fcn)

        ax2.text(0.01, (categories-idx)/categories, x_ticks[idx] + ': ', fontsize=10, ha='left', va='center', wrap=True)
        ax2.text(0.2, (categories-idx)/categories, r'${}$'.format(text_fcn), fontsize=10, ha='left', va='center', wrap=True)


    # Remove the axes
    ax2.axis('off')

    # Get functions to print, and how many times they were in top 6
    for idx, row in enumerate(ordered_gold):
        print('{0}: {1}'.format(x_ticks[idx], row[:6]))

    # Save
    plt.tight_layout()
    plt.savefig('stacked_rank.pdf',bbox_inches='tight')
    plt.clf()

    if rank == 1:
        print('--------------------------------------------')
        print('Plot 1 made')
        print('--------------------------------------------')

    # --- Plot 2 --- #
    cm = plt.get_cmap('viridis_r')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel('Simulations, ordered by the difference between best and second best')
    ax.set_xticks([])
    ax.set_ylabel('Scaled Description Length')
    ax.set_yticks([0])
    ax.set_yticklabels([0])

    all_DL = []

    for idx, row in enumerate(ordered_gold):
        DL = np.array(row[-2].split(';')).astype(float)
        all_DL.append(DL)

    missed = []
    all_y_data = []


    transposed_DL = np.transpose(all_DL)
    for idx, DL in enumerate(transposed_DL): # Now grouped by sim

        for val_idx, val in enumerate(DL):
            if val - min(DL) > 250: #150:
                if rank == 1:
                    missed_fcn = ordered_gold[val_idx][-1]
                    missed.append([missed_fcn, idx])
#                DL[val_idx] = np.nan
    
        scaled_DL = DL - min(DL)
        all_y_data.append(list(scaled_DL))

    max_heights = []
    all_y_data = np.transpose(all_y_data) # Now grouped by functions' DL

    hmf_sims = np.array(hmf_sims, dtype=np.float64)  # Convert to float

    # -- Ordering the data -- #
    data = all_y_data[:5]
    data = np.transpose(data) # Top 5 functions data, grouped by hmf_sim
    
    differences = []
    for i in data:
        differences.append(min(i) - min(np.delete( i, list(i).index(min(i)) )) )

    sorted_indices = np.argsort(differences)
    
    sorted_data = data[sorted_indices]
    data = np.transpose(sorted_data)

    for idx, fcn_DL in enumerate(data):
        ax.plot(hmf_sims, fcn_DL, c=cm(idx/5), zorder=5-idx, label= r'${}$'.format(labels[idx]))
        max_heights.append(max(fcn_DL))

    ax.set_xlim([hmf_sims[0],hmf_sims[-1]])
    ax.set_ylim(top=max(max_heights)*1.1, bottom=0.001)
    
    ax.legend(ncols=5)
    plt.tight_layout()
    plt.savefig('DL_scatter.pdf',bbox_inches='tight')
    plt.clf()


    if rank == 1:
        print('--------------------------------------------')
        print('Plot 2 made')
        print('--------------------------------------------')


## If you want to save the histogram data, run this:
#    np.savetxt('histogram_data.dat', np.transpose(all_y_data))


#    # --- Plot 3 --- #
#    fig, ax = plt.subplots(figsize=(10, 5))
#    for idx, row in enumerate(ordered_gold[:5]):
#        data = all_y_data[idx]
#        bin_width = 2
#        x = np.linspace(0, 35 + bin_width, bin_width)
#        print(r'${}$'.format(labels[idx]))
#        ax.hist(data, bins=binwidth, range=(0,35), alpha=0.3, density=True, histtype='bar', color=cm(idx/6), label=r'${}$'.format(labels[idx]), zorder=6-idx)
#    
#    #    kde = gaussian_kde(data, bw_method=0.1)
#    #    if rank == 1:
#    #        bw = kde.factor
#    #        print(bw)
#    #    x = np.linspace(0, data.max() + 2, 1000)
#    #    y = kde(x)
#
#    #    ax.fill_between(x,y,color=cm(idx/5), label=r'${}$'.format(labels[idx]), zorder=5-idx, alpha=0.3)
#    #    ax.plot(x,y,color=cm(idx/5), zorder=5-idx)
#
#        #sns.kdeplot(x, ax=ax, fill=True, color=cm(idx/5), label=r'${}$'.format(labels[idx]), zorder=5-idx, cut=0, bw_adjust=.75)
#        
#        if idx == 0 and max(data) > 0.6:
#            # Surpasses y maximum
#            ax.text(2, 0.55, s='Peak height: {}'.format(round(max(data),2)))
#
#    ax.legend()
#    ax.minorticks_on()
#    ax.set_xlim(left=0, right=35)
#    ax.set_ylim(bottom=0, top=0.6)
#    ax.set_xlabel(r'$\delta$' + 'DL')
#    ax.set_ylabel('Frequency density')
#
#    plt.tight_layout()
#    plt.savefig('ranked_histogram.png', dpi=200)
#    plt.clf()
#
#    if rank == 1:
#        print('--------------------------------------------')
#        print('Plot 3 made')
#        print('--------------------------------------------')

    # --- Plot 4 --- #
    import random

    fig, ax = plt.subplots(figsize=(10, 5))
    data, fcns = [], []
    for idx, row in enumerate(ordered_gold[:5]):
        data.append(all_y_data[idx])
        fcns.append(labels[idx])


    box = ax.boxplot(data, positions=np.arange(1, 6), widths=0.6,
                 patch_artist=True, showfliers=False,
                 boxprops=dict(facecolor='lightgray', color='black'),
                 medianprops=dict(color='black'),
                 zorder = 1)
    
#    for i in range(len(data)):
#        # Box
#        box['boxes'][i].set_facecolor(cm(i/5))
#        box['boxes'][i].set_alpha(0.5)
#        box['boxes'][i].set_edgecolor(cm(i/5))
#
#        # Median
#        box['medians'][i].set_color('black')
#        box['medians'][i].set_linewidth(2.5)
#
#        # Whiskers (2 per box)
#        box['whiskers'][2*i].set_color(cm(i/5))
#        box['whiskers'][2*i + 1].set_color(cm(i/5))
#        box['whiskers'][2*i].set_linewidth(1.5)
#        box['whiskers'][2*i + 1].set_linewidth(1.5)
#
#        # Caps (2 per box)
#        box['caps'][2*i].set_color(cm(i/5))
#        box['caps'][2*i + 1].set_color(cm(i/5))


    # Scatter (jittered) data points
    for i, group in enumerate(data):
        x = np.random.normal(i + 1, 0.05, size=len(group))  # jitter around group index
        ax.scatter(x, group, alpha=0.6, s=15, color=cm(i/5), edgecolor='black', linewidth=0.3, zorder=2, label=x_ticks[i] + ': ' + r'${}$'.format(labels[i]))

    ax.set_ylabel(r'$\delta$' + 'DL')
    ax.set_xlabel('Top 5 Functions Overall')
    ax.set_xticks(np.arange(1,6), x_ticks[:5])
    plt.legend()
    plt.tight_layout()
    plt.savefig('box_plot.pdf', bbox_inches='tight')
    plt.clf()

    if rank == 1:
        print('--------------------------------------------')
        print('Plot 4 made')
        print('--------------------------------------------')




        


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else None

    if step == 'step1':
        hmf_sims = np.arange(0, 100, 10)
        get_uniq_top_200(hmf_sims)

    elif step == 'step2':
        size = comm.Get_size()
        hmf_sims = np.arange(0, 100)
        # Distribute simulations across MPI ranks
        my_sims = [s for s in hmf_sims if s % size == rank]
        for hmf_sim in my_sims:
            run_fits(hmf_sim)

    elif step == 'step3':
        hmf_sims = np.arange(0, 100)
        zipped, hmf_sims = gather_barchart_data(hmf_sims)
        create_all_plots(zipped, hmf_sims)

    else:
        print("Usage: python3 sample_top_200.py [step1|step2|step3]")
        print("  step1: Regenerate top 500 list from 10 comprehensive sims")
        print("  step2: Fit top 200 to all 100 simulations")
        print("  step3: Generate plots from fitted results")

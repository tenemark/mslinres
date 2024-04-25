# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:04:40 2023

@author: wqg436
"""

import mikeio
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../')
from mslinres import linear_reservoir
from mslinres_helpers import load_sim_data
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats

then = datetime.now()

# Setup
# load outflow areas
outs = np.load('./input files/outflow_areas.npy')

# load river branches
res = mikeio.read('./input files/linearreservoirs.dfs2')
riv = res['river_branches'].values[0, :, :]

# load conceptual model definitions
models = pd.read_csv('./input files/models.txt', sep='\t', dtype=str)
models.index = models.IF
model = 'sub8_elev'  # conceptual model

ifres = res[model].values[0, :, :]  # interflow reservoirs
bfres = res[models.loc[model, 'BF']].values[0, :, :]  # baseflow reservoirs
if_connects = pd.read_csv('./input files/'+models.loc[model, 'ifconnect'],
                          index_col=0, dtype=str)
if_connects = if_connects.where(pd.notnull(if_connects), None).to_dict()

# names of interflow and baseflow reservoirs
bf_nmes = [str(int(nme)) for nme in np.unique(bfres)]
if '0' in bf_nmes:
    bf_nmes.remove('0')
if_nmes = [str(int(nme)) for nme in np.unique(ifres)]
if '0' in if_nmes:
    if_nmes.remove('0')

# load simulated data
result_ff = './mike she result files/'+model+'/'
mod_nme = 'test.she'
df, qpumps, qins = load_sim_data(if_nmes, bf_nmes, mod_nme, result_ff)
tindex = df.index

# Parameters
bounds = pd.read_csv('./input files/par_bounds.csv', sep=';', index_col=0)
par_if = ['ki', 'kp', 'hit', 'isy', 'hibot', 'hi0']
par_bf = ['kb', 'hbt', 'bsy', 'hb0', 'bffrac', 'hbbot']
ptypes = par_if + par_bf
par_nmes = [par+'_'+if_name for par in par_if for if_name in if_nmes] +\
    [par+'_'+bf_name for par in par_bf for bf_name in bf_nmes]
params = {}
for ptype in bounds.index:
    idx = [i for i, par_nme in enumerate(par_nmes) if ptype in par_nme]
    pars = [par_nme for i, par_nme in enumerate(par_nmes) if ptype in par_nme]
    par_ini = bounds.loc[ptype, 'ini']
    for p in pars:
        params[p] = par_ini
par = params

# Running model
lr = linear_reservoir(bfres, ifres, if_connects, tindex=tindex)
lr.assign_inflow_and_qpumps(qins, qpumps)
lr.run_model(par, riv, outs)
outflow = lr.outflow
flow = lr.flow
head = lr.head
bfr = '1'
ifr = '1'
sur_df = flow
sim_df = df

# %% Running all
outflows = [None]*len(models)
r2 = [None]*len(models)
for i, model in enumerate(models.index):
    # loading conceptual model specific data
    result_ff = './mike she result files/'+model+'/'
    ifres = res[model].values[0, :, :]  # interflow reservoirs
    bfres = res[models.loc[model, 'BF']].values[0, :, :]  # baseflow reservoirs
    if_connects = pd.read_csv('./input files/'+models.loc[model, 'ifconnect'],
                              index_col=0, dtype=str)
    if_connects = if_connects.where(pd.notnull(if_connects), None).to_dict()

    # names of interflow and baseflow reservoirs
    bf_nmes = [str(int(nme)) for nme in np.unique(bfres)]
    if '0' in bf_nmes:
        bf_nmes.remove('0')
    if_nmes = [str(int(nme)) for nme in np.unique(ifres)]
    if '0' in if_nmes:
        if_nmes.remove('0')
    # loading simulated data
    df, qpumps, qins = load_sim_data(if_nmes, bf_nmes, mod_nme, result_ff)

    # Running model
    lr = linear_reservoir(bfres, ifres, if_connects, tindex=tindex)
    lr.assign_inflow_and_qpumps(qins, qpumps)
    lr.run_model(par, riv, outs)
    outflow = lr.outflow
    outflow.columns = ['C2', 'C1']
    outflows[i] = outflow

    mssim = mikeio.read('./mike she result files/'+model+'/testDetailedTS_M11.dfs0').to_dataframe()
    x, y = outflow['C2'].values, mssim['C2'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    r2[i] = scipy.stats.linregress(x[mask], y[mask])[2]

# %%
outflow = lr.outflow.resample('1D').sum()
outflow.columns = ['C2', 'C1']
mssim = mikeio.read('./mike she result files/'+model+'/testDetailedTS_M11.dfs0').to_dataframe()
r = r2[i]

fig, ax = plt.subplots(figsize=(4, 4))
ax.grid()
maxi = np.nanmax(x)
ax.scatter(x, y, alpha=0.5)
ax.plot([0, maxi], [0, maxi], '-k', zorder=0)
ax.axis('equal')
ax.set_ylim(0, maxi)
ax.set_xlim(0, maxi)
ax.annotate('Correlation coefficient: {:0.2f}'.format(r), xy=(0, maxi-20))

# %% Plotting correlation coefficient for all runs
fig, ax = plt.subplots()
ax.plot(np.arange(len(models)), r2, marker='.')
plt.xticks(np.arange(len(models)), rotation=90)
ax.set_xticklabels(models.index)
ax.grid()
ax.set_xlabel('Models')
ax.set_ylabel('Correlation coefficient')

# %% Plotting hydrographs of surrogate and mike she model
i = 5
index = outflow.index.month.isin([7, 8, 9])
mssim = mikeio.read('./mike she result files/'+model+'/testDetailedTS_M11.dfs0').to_dataframe()
mssim.loc[index, 'C2'].plot(label='sim')
ts = outflows[i].loc[index, 'C2']
ts.plot(label='sur')
plt.legend()

# %% Interflow verification
ifr = '2'

def plot_interflow_flow(ifr):
    # inflow
    upstream = if_connects[ifr]['upstream']
    sur_df['qin_'+ifr].cumsum().plot(color='green', label='inflow, surrogate')
    if upstream is not None:
        inflow = sim_df['IF_Inflow, '+ifr] + sim_df['IF_Outflow, ' + upstream]
    else:
        inflow = sim_df['IF_Inflow, '+ifr]
    inflow.cumsum().plot(color='green', linestyle='--',
                         label='inflow, mike she')
    # interflow
    sur_df['qi_'+ifr].cumsum().plot(color='blue', label='interflow, surrogate')
    sim_df['IF_Outflow, '+ifr].cumsum().plot(color='blue', linestyle='--',
                                             label='interflow, mike she')
    # percolation
    sur_df['qp_'+ifr].cumsum().plot(color='orange',
                                    label='percolation, surrogate')
    sim_df['IF_Recharge, '+ifr].cumsum().plot(color='orange', linestyle='--',
                                              label='percolation, mike she')
    # plot settings
    plt.ylabel('cumulative flow [m$^3$/s]')
    plt.legend()
    plt.title('Interflow reservoir '+ifr)

plot_interflow_flow(ifr)

# %% Baseflow verification
bfr = '1'
def plot_baseflow_flow(bfr):
    sur_df['qv1_'+bfr].cumsum().plot(color='green', label='recharge, surrogate')
    sim_df['BF1_Recharge, '+bfr].cumsum().plot(color='green', linestyle='--', label='recharge, mike she')
    sur_df['qb1_'+bfr].cumsum().plot(color='blue', label='outflow, surrogate')
    sim_df['BF1_Outflow, '+bfr].cumsum().plot(color='blue', linestyle='--', label='outflow, mike she')
    sur_df['qpump1_'+bfr].cumsum().plot(color='orange', label='pumping, surrogate')
    sim_df['BF1_IrrPump, '+bfr].cumsum().plot(color='orange', linestyle='--', label='pumping, mike she')
    # plot settings
    plt.ylabel('cumulative flow [m$^3$/s]')
    plt.legend()
    plt.title('Baseflow reservoir '+bfr)

plot_baseflow_flow(bfr)


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
import datetime
import matplotlib.pyplot as plt
import scipy.stats

then = datetime.datetime.now()

# Setup
# load outflow areas
outs = np.load('./input files/outflow_areas.npy')

# load river branches
res = mikeio.read('./input files/linearreservoirs.dfs2')
riv = res['river_branches'].values[0, :, :]

# load conceptual model definitions
models = pd.read_csv('./input files/models.txt', sep='\t', dtype=str)
models.index = models.IF
model = 'sub1_prox'  # conceptual model

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
value = 2  # parameter value
result_ff = './mike she result files/par'+str(value)+'/'
ms_file = 'test.she'  # mike she file
df, qpumps, qins = load_sim_data(if_nmes, bf_nmes, ms_file, result_ff)
tindex = df.index

# Parameters
bounds = pd.read_csv('./input files/par_bounds.csv', sep=';', index_col=0)
par_if = ['ki', 'kp', 'hit', 'isy', 'hibot', 'hi0']
par_bf = ['kb', 'hbt', 'bsy', 'hb0', 'bffrac', 'hbbot']
ptypes = par_if + par_bf
par_nmes = [par+'_'+if_name for par in par_if for if_name in if_nmes] +\
    [par+'_'+bf_name for par in par_bf for bf_name in bf_nmes]
params = {}
for ptype in ptypes:
    idx = [i for i, par_nme in enumerate(par_nmes) if ptype in par_nme]
    pars = [par_nme for i, par_nme in enumerate(par_nmes) if ptype in par_nme]
    par_ini = bounds.loc[ptype, 'ini']
    for p in pars:
        params[p] = par_ini
par = params
par['kp_1'] = 5
par['kp_2'] = 5

# for updating
par['ki_1'] = value
par['ki_2'] = value
par['kb_1'] = value

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
index = sim_df.index.year.isin([2011]) & sim_df.index.month.isin([7, 8, 9])

# %% Running all
values = [2, 4, 8, 16, 32, 64, 128, 256, 512]
outflows = [None]*len(values)
r2 = [None]*len(values)
for i, value in enumerate(values):
    par['ki_1'] = value
    par['ki_2'] = value
    par['kb_1'] = value
    result_ff = './mike she result files/par'+str(value)+'/'
    df, qpumps, qins = load_sim_data(if_nmes, bf_nmes, ms_file, result_ff)

    # Running model
    lr = linear_reservoir(bfres, ifres, if_connects, tindex=tindex)
    lr.assign_inflow_and_qpumps(qins, qpumps)
    lr.run_model(par, riv, outs)
    outflow = lr.outflow
    outflow.columns = ['C2', 'C1']
    outflows[i] = outflow

    mssim = mikeio.read('./mike she result files/par'+str(value)+'/testDetailedTS_M11.dfs0').to_dataframe()
    x, y = outflow['C2'].values, mssim['C2'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    r = scipy.stats.linregress(x[mask], y[mask])[2]
    r2[i] = r
# %% Plotting correlation coefficient for all runs
plt.plot(values, r2, marker='.')
plt.grid()
plt.xlabel('Time constant')
plt.ylabel('Correlation coefficient')

# %% Plotting hydrographs of surrogate and mike she model
i = 1
value = values[i]
index = outflow.index.month.isin([7, 8, 9])
mssim = mikeio.read('./mike she result files/par'+str(value)+'/testDetailedTS_M11.dfs0').to_dataframe()
mssim.loc[index, 'C2'].plot(label='sim')
ts = outflows[i].loc[index, 'C2']
ts.plot(label='sur')
plt.legend()

# %% Scatterplot of simulated and surrogate values
i = 0

value = values[i]
mssim = mikeio.read('./mike she result files/par'+str(value)+'/testDetailedTS_M11.dfs0').to_dataframe()
x, y = np.log(outflows[i]['C1'].values), np.log(mssim['C1'].values)
x, y = outflows[i]['C1'].values, mssim['C1'].values
r = r2[i]

fig, ax = plt.subplots(figsize=(4, 4))
ax.grid()
maxi = np.nanmax(x)
ax.scatter(x, y, alpha=0.5)

ax.plot([0, maxi], [0, maxi], '--k', zorder=0)
ax.axis('equal')
ax.set_ylim(0, maxi)
ax.set_xlim(0, maxi)
ax.annotate('Correlation coefficient: {:0.2f}'.format(r), xy=(0, maxi-2),
            verticalalignment="top")
ax.set_ylabel('Discharge [m$^3$/s], simulation')
ax.set_xlabel('Discharge [m$^3$/s], surrogate')
# plt.savefig('./vis/verification.png',  dpi=300, bbox_inches='tight')

# %% if flow
dt = 1/24/60/60
findex = sim_df.index.year.isin([2011]) & sim_df.index.month.isin([7, 8, 9, 10])
# test = (par['hit_'+ifr] - df['IF_DepthWTable, '+ifr])/par['ki_'+ifr]/dt*-1
# test.plot()
sim_df.loc[findex, 'IF_Outflow, '+ifr].plot()
sur_df.loc[findex, 'qi_'+ifr].rolling(2).mean().plot()

# %% bf flow
index = outflow.index.month.isin([7, 8, 9])
test = (par['hbt_'+bfr] - df['BF1_DepthWTable, '+bfr])/par['kb_'+bfr]/dt*-1
test.loc[index].plot(label='test')
sim_df.loc[index, 'BF1_Outflow, '+bfr].plot(label='sim')
sur_df.loc[index, 'qb1_'+bfr].plot()
plt.legend()

# %% Interflow verification
ifr = '1'

def plot_interflow_flow(ifr, sur_df, sim_df, ax):
    # inflow
    upstream = if_connects[ifr]['upstream']
    sur_df['qin_'+ifr].cumsum().plot(color='green', label='inflow, surrogate',
                                     ax=ax)
    if upstream is not None:
        inflow = sim_df['IF_Inflow, '+ifr] + sim_df['IF_Outflow, ' + upstream]
    else:
        inflow = sim_df['IF_Inflow, '+ifr]
    inflow.cumsum().plot(color='green', linestyle='--',
                         label='inflow, mike she', ax=ax)
    # interflow
    sur_df['qi_'+ifr].cumsum().plot(color='blue',
                                    label='interflow, surrogate', ax=ax)
    sim_df['IF_Outflow, '+ifr].cumsum().plot(color='blue', linestyle='--',
                                             label='interflow, mike she',
                                             ax=ax)
    # percolation
    sur_df['qp_'+ifr].cumsum().plot(color='orange',
                                    label='percolation, surrogate', ax=ax)
    sim_df['IF_Recharge, '+ifr].cumsum().plot(color='orange', linestyle='--',
                                              label='percolation, mike she',
                                              ax=ax)
    # plot settings
    ax.set_ylabel('cumulative flow [m$^3$/s]')
    plt.legend()
    ax.set_title('Interflow reservoir '+ifr)


fig, ax = plt.subplots()
plot_interflow_flow(ifr, sur_df, sim_df, ax)

# %% Baseflow verification
bfr = '1'
sub = '1'

def plot_baseflow_flow(bfr, sur_df, sim_df, ax=ax):
    sur_df['qv'+sub+'_'+bfr].cumsum().plot(color='green',
                                           label='recharge, surrogate', ax=ax)
    sim_df['BF'+sub+'_Recharge, '+bfr].cumsum().plot(color='green',
                                                     linestyle='--',
                                                     label='recharge, mike she',
                                                     ax=ax)
    sur_df['qb'+sub+'_'+bfr].cumsum().plot(color='blue',
                                           label='outflow, surrogate', ax=ax)
    sim_df['BF'+sub+'_Outflow, '+bfr].cumsum().plot(color='blue',
                                                    linestyle='--',
                                                    label='outflow, mike she',
                                                    ax=ax)
    sur_df['qpump'+sub+'_'+bfr].cumsum().plot(color='orange',
                                              label='pumping, surrogate',
                                              ax=ax)
    sim_df['BF'+sub+'_IrrPump, '+bfr].cumsum().plot(color='orange',
                                                    linestyle='--',
                                                    label='pumping, mike she',
                                                    ax=ax)
    ax.set_ylabel('cumulative flow [m$^3$/s]')
    plt.legend()
    ax.set_title('Baseflow reservoir '+bfr)


fig, ax = plt.subplots()
plot_baseflow_flow(bfr, sur_df, sim_df, ax=ax)


# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:56:32 2023

@author: wqg436
"""
import pandas as pd
import numpy as np
from mslinres import linear_reservoir
import mikeio

# Conceptual model files
ms_file = 'Udai_MS_3.she'
models = pd.read_csv('./input files/models.txt', sep='\t', dtype=str)
models.index = models.IF

# Observations
dis_locs = ['Bigod', 'Chittorgarh']
obs = [None]*len(dis_locs)
for i in range(len(dis_locs)):
    load = mikeio.read('./input files/Discharge_'+dis_locs[i]+'_TS.dfs0')[0]
    obs[i] = pd.DataFrame(data=load.values, index=load.time, columns=[dis_locs[i]])
obs = pd.concat(obs, axis=1)
obs = obs.loc[obs.index.year.isin([i for i in range(2011, 2020)])]

# load outflow areas
outs = np.load('./input files/outflow_areas.npy')

# load river branches
res = mikeio.read('./input files/linearreservoirs.dfs2')
riv = res['river_branches'].values[0, :, :]


def load_sim_data(if_nmes, bf_nmes, ms_file, result_ff):
    dfs = []
    for c in bf_nmes:
        tfile = result_ff + ms_file.split('.')[0] + '_SC_Grid code = '+c+'.dfs0'
        df = mikeio.read(tfile).to_dataframe()
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    cols = [col.replace('Grid code = ', '') for col in df.columns]
    df.columns = cols
    qins = {}
    qpumps = {}
    for ifnme in if_nmes:
        qin = df['IF_Inflow, ' + ifnme].values
        qins[ifnme] = qin
    for bfnme in bf_nmes:
        qpumps[bfnme+'_1'] = df['BF1_IrrPump, '+bfnme].values
        qpumps[bfnme+'_2'] = df['BF2_IrrPump, '+bfnme].values

    ifcols = ['IF_Inflow, ', 'IF_Outflow, ', 'IF_Recharge, ',
              'IF_DepthWTable, ']
    bfcols = ['BF1_Recharge, ', 'BF1_Outflow, ', 'BF1_IrrPump, ',
              'BF1_DepthWTable, ',
              'BF2_Recharge, ', 'BF2_Outflow, ', 'BF2_IrrPump, ',
              'BF2_DepthWTable, ']
    cols = [ic + ifn for ic in ifcols for ifn in if_nmes] + \
        [bc + bfn for bc in bfcols for bfn in bf_nmes]
    df = df[cols]
    return df, qpumps, qins


def assign_inflow_from_file(recharge_dfs, ifres):  # not working quite right
    recharge = mikeio.read(recharge_dfs)['exchange between UZ and SZ (pos.up)']
    if_nmes = list(np.unique(ifres).astype(int))[1:]
    qins = {}
    for ifnme in if_nmes:
        mask2d = ifres == int(ifnme)
        mask3d = np.broadcast_to(mask2d, recharge.shape)
        rch_if = recharge[mask3d].reshape((recharge.shape[0], np.sum(mask2d)))
        rch_if = np.nansum(rch_if, axis=1)
        factor = mask2d.sum()*1000/60/60/24
        factor = 1000/60/60/24*29859.837464640
        # flow['qin_'+ifnme] = rch_if*-1 / factor
        qins[ifnme] = rch_if*-1 / factor


def calc_kge(y_true, y_pred):
    m1, m2 = np.mean(y_true), np.mean(y_pred)
    r = np.sum((y_true-m1) * (y_pred-m2))/(np.sqrt(np.sum((y_true-m1)**2))
                                           * np.sqrt(np.sum((y_pred - m2) ** 2)))
    beta = m2 / m1
    gamma = (np.std(y_pred) / m2) / (np.std(y_true) / m1)
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    # kge  = -9 if np.isnan(kge) else kge
    return kge


def load_model_data(model, result_ff):
    ifres = res[model].values[0, :, :]
    bfres = res[models.loc[model, 'BF']].values[0, :, :]

    if_connects = pd.read_csv(models.loc[model, 'ifconnect'], index_col=0,
                              dtype=str)
    if_connects = if_connects.where(pd.notnull(if_connects), None).to_dict()

    bf_nmes = [str(int(nme)) for nme in np.unique(bfres)]
    if '0' in bf_nmes:
        bf_nmes.remove('0')
    if_nmes = [str(int(nme)) for nme in np.unique(ifres)]
    if '0' in if_nmes:
        if_nmes.remove('0')

    # loading input from ms
    df, qpumps, qins = load_sim_data(if_nmes, bf_nmes, ms_file, result_ff)
    tindex = df.index
    return ifres, bfres, if_connects, df, qpumps, qins, tindex


def forward_func_global(model, run):
    result_ff = './'+run+'/'+model+'/'
    ifres, bfres, if_connects, df, qpumps, qins, tindex = load_model_data(model, result_ff)
    lr = linear_reservoir(bfres, ifres, if_connects, tindex=tindex)
    lr.assign_inflow_and_qpumps(qins, qpumps)

    # parameters
    par_if = ['ki', 'kp', 'hit', 'isy', 'hi0']
    par_bf = ['kb', 'hbt', 'bsy', 'hb0']
    par_nmes = par_if+par_bf

    par_nmes.sort()
    p = np.loadtxt(result_ff+'/global_parameters.txt')

    # parameters option 3
    par = {nme+'_'+str(i): val for val, nme in zip(p, par_nmes)
           for i in lr.if_nmes if nme in par_if}
    par.update({nme+'_'+str(i): val for val, nme in zip(p, par_nmes)
                for i in lr.bf_nmes if nme in par_bf})

    # add fixed parameters (option 2, 3)
    par.update({'bffrac_'+i: 1 for i in lr.bf_nmes})
    par.update({'hbbot_'+i: -20 for i in lr.bf_nmes})
    par.update({'hibot_'+i: -5 for i in lr.if_nmes})

    # Running model
    lr.run_model(par, riv, outs)
    outflow = lr.outflow
    return outflow


def forward_func_local(model, run):
    result_ff = './'+run+'/'+model+'/'

    ifres, bfres, if_connects, df, qpumps, qins, tindex = load_model_data(model, result_ff)
    
    # Parameters
    par = pd.read_csv(result_ff + 'local_parameters.txt', sep=' = ',
                      index_col=0, engine='python', header=None).to_dict()[1]
    
    # Running model
    lr = lr = linear_reservoir(bfres, ifres, if_connects, tindex=tindex)
    lr.assign_inflow_and_qpumps(qins, qpumps)
    lr.run_model(par, riv, outs)
    outflow = lr.outflow

    return outflow


def forward_func(p, *args):
    # bfres = kwargs['bfres']
    # ifres = kwargs['ifres']
    # outs = kwargs['outs']
    # if_connects = kwargs['if_connects']
    # tindex = kwargs['tindex']
    # bf_areas = kwargs['bf_areas']
    # outflow_areas = kwargs['outflow_areas']
    # qins = kwargs['qins']
    # qpumps = kwargs['qpumps']
    bfres = args[0]
    ifres = args[1]
    outs = args[3]
    if_connects = args[4]
    tindex = args[5]
    bf_areas = args[6]
    outflow_areas = args[7]
    qins = args[8]
    qpumps = args[9]

    # initializing model
    lr = linear_reservoir(bfres, ifres, if_connects, tindex=tindex)
    # outflow_areas ad bf_areas should not be calculated if assigned
    lr.outflow_areas = outflow_areas
    lr.bf_areas = bf_areas
    # assign inflow and pumping
    lr.assign_inflow_and_qpumps(qins, qpumps)

    # parameters
    par_if = ['ki', 'kp', 'hit', 'isy', 'hi0']
    par_bf = ['kb', 'hbt', 'bsy', 'hb0']
    par_nmes = par_if+par_bf

    par_nmes.sort()
    par = {nme+'_'+str(i): val for val, nme in zip(p, par_nmes)
           for i in lr.if_nmes if nme in par_if}
    par.update({nme+'_'+str(i): val for val, nme in zip(p, par_nmes)
                for i in lr.bf_nmes if nme in par_bf})

    # add fixed parameters (option 2, 3)
    par.update({'bffrac_'+i: 1 for i in lr.bf_nmes})
    par.update({'hbbot_'+i: -20 for i in lr.bf_nmes})
    par.update({'hibot_'+i: -5 for i in lr.if_nmes})

    # Running model
    lr.run_model(par, riv, outs)
    outflow = lr.outflow
    wsse = outflow_stats(outflow, tindex)
    return wsse


def outflow_stats(outflow, tindex, st_date='2011-01-01', en_date='2016-12-31'):
    wsse = [None]*len(dis_locs)
    for i, loc in enumerate(dis_locs):
        # load obs discharge
        sim_df = pd.DataFrame(outflow[i].values, index=tindex)
        # combine by interpolating sim to obs index
        dis = pd.DataFrame(obs[loc].values, columns=['obs'], index=obs.index)
        dis["sim"] = np.interp(dis.index, sim_df.index, sim_df[0],
                               left=np.nan, right=np.nan)
        dis.dropna(axis=0, inplace=True)
        dis = dis.loc[st_date:en_date]

        y_true, y_pred = np.array(dis["obs"]), np.array(dis["sim"])
        value = calc_kge(y_true, y_pred)
        wsse[i] = (1-value)**2
    return wsse


def global_func(p, args):
    print(args[5])
    outflow = forward_func(p, *args)
    wsse = outflow_stats(outflow, args[5])
    return wsse

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:09:21 2023

@author: wqg436
"""
import mikeio
import numpy as np
import pandas as pd

from datetime import datetime
then = datetime.now()


class linear_reservoir:
    """
    Class that mimics response of linear reservoir module in MIKE SHE.

    Parameters
    ----------
    bfres : np array
        2D in shape of model grid with grid codes indicating baseflow
        reservoirs.
    ifres : np array
        2D in shape of model grid with grid codes indicating interflow
        reservoirs.
    if_connects : dict
        The first key is the name of the interflow reservoir.
        The second keys are upstream and downstream.
        The value refers to the name of the downstream or upstream interflow
        reservoir.
    tindex : DatetimeIndex
        time index of the simulation.
    """

    dt = 1/24/60/60  # time step
    bfparnmes = ['1', '2']  # name of parallel baseflow reservoirs

    def __init__(self, bfres, ifres, if_connects, tindex):
        self.bfres = bfres  # baseflow reservoirs
        self.ifres = ifres  # interflow reservoirs
        self.if_connects = if_connects  # interflow connections
        self.tindex = tindex  # time index
        
        # try:
        #     self.bf_areas
        # except AttributeError:
        #     self.bf_areas = None  # baseflow reservoir areas
        # try:
        #     self.outflow_areas
        # except AttributeError:
        #     self.outflow_areas = None  #
        if not hasattr(self, 'bf_areas'):
            self.bf_areas = None
        if not hasattr(self, 'outflow_areas'):
            self.outflow_areas = None
    
        bfparnmes = ['1', '2']  # name of parallel baseflow reservoirs

        # infer reservoir nmes
        self.bf_nmes = [str(int(nme)) for nme in np.unique(bfres)]
        self.if_nmes = [str(int(nme)) for nme in np.unique(ifres)]
        for lis in [self.bf_nmes, self.if_nmes]:
            if '0' in lis:
                lis.remove('0')

        # initialize dataframes head and flow for calculations
        # flow columns
        fcols = [pre+'_'+nme for nme in self.if_nmes for pre in ['qin', 'qi', 'qp']] + \
            [pre+bpn+'_'+name for name in self.bf_nmes for bpn in bfparnmes for pre in ['qv', 'qb']]
        flow = pd.DataFrame(index=tindex, columns=fcols)
        flow.fillna(0, inplace=True)
        # head columns
        hcols = ['hi_' + nme for nme in self.if_nmes] + \
            ['hb_' + bpn + bfname for bfname in self.bf_nmes for bpn in bfparnmes]
        head = pd.DataFrame(columns=hcols, index=tindex)

        self.flow = flow
        self.head = head

    def extract_bf_areas(self):
        """
        Calculate overlapping areas btw interflow and baseflow reservoirs.

        The output is saved in the nested dictionary bf_areas where the name of
        the interflow reservoir is the first key, the name of the baseflow
        reservoir is the second key and the value is the overlapping area.
        The overlapping area is given in percentage of full area of interflow
        reserovoir.
        bf_areas is used to distribute percolation from the interflow
        reservoirs to baseflow reservoirs.

        Returns
        -------
        None.

        """
        bf_areas = {}  # initialize disctionary
        # looping through interflow reservoirs
        for i, ifnme in enumerate(self.if_nmes):
            bf_areas[ifnme] = {}  # initialize
            # looping through baseflow reservoirs
            for j, bfnme in enumerate(self.bf_nmes):
                # calc overlapping area
                mask = (self.bfres == int(bfnme)) & (self.ifres == int(ifnme))
                if np.sum(mask) != 0:
                    bf_areas[ifnme][bfnme] = np.sum(mask)/np.sum((self.ifres == int(ifnme)))
                else:
                    bf_areas[ifnme][bfnme] = 0
        self.bf_areas = bf_areas

    def extract_outflow_areas(self, riv, outs):
        """
        Calculate the overlapping outflow areas with rivers and reservoirs.

        The output is a nested dictionary with the outflow area name in the
        first key, if or bf in the second ad the name of the interflow or
        baseflow reservoir in the third. The value is the overlapping area in
        percentage. The output is used to distribute interflow and baseflow
        to river branches.

        Parameters
        ----------
        riv : np array
            2D in shape of model grid with indication of position of rivers.
        outs : np array
            2D in shape of model grid with indication of contributing areas for
            runoff station.
        Returns
        -------
        None.

        """
        """ based on river branches """
        outflow = {}  # initialize dictionary
        # looping over outflow areas
        for o in range(outs.shape[0]):
            out = outs[o, :, :]
            outflow[o] = {}
            outflow[o]['if'] = {}
            outflow[o]['bf'] = {}
            # looping over interflow reservoirs
            for i, ifnme in enumerate(self.if_nmes):
                mask = (out == 1) & (self.ifres == int(ifnme)) & (riv == 1)
                if np.sum(mask) != 0:
                    m2 = (self.ifres == int(ifnme)) & (riv == 1)
                    outflow[o]['if'][ifnme] = np.sum(mask)/np.sum(m2)
                else:
                    outflow[o]['if'][ifnme] = 0
            # looping over baseflow reservoirs
            for j, bfnme in enumerate(self.bf_nmes):
                mask = (out == 1) & (self.bfres == int(bfnme)) & (riv == 1)
                if np.sum(mask) != 0:
                    m2 = (self.bfres == int(bfnme)) & (riv == 1)
                    outflow[o]['bf'][bfnme] = np.sum(mask)/np.sum(m2)
                else:
                    outflow[o]['bf'][bfnme] = 0
            # correct for interflow reservoirs that feed other interflow
            # reservoirs and not river
            for ifnme in self.if_nmes:
                if self.if_connects[ifnme]['downstream'] is not None:
                    outflow[o]['if'][ifnme] = 0

        self.outflow_areas = outflow

    def calc_outflow(self, outs, riv=None):
        """
        Calculate outflow from interflow and baseflow to rivers.

        Parameters
        ----------
        riv : np array
            2D in shape of model grid with indication of position of rivers.
        outs : np array
            2D in shape of model grid with indication of contributing areas for
            runoff station.

        Returns
        -------
        None.

        """
        flow = self.flow
        if self.outflow_areas is None:
            print('Calculating outflow_areas')
            self.extract_outflow_areas(riv, outs)
        outflow_areas = self.outflow_areas
        outflow = pd.DataFrame(columns=[i for i in range(outs.shape[0])],
                               index=self.tindex)
        outflow.fillna(0, inplace=True)
        for outnme in range(outs.shape[0]):
            for ifnme in self.if_nmes:
                outflow[outnme] += flow['qi_'+ifnme] * outflow_areas[outnme]['if'][ifnme]
            for bfnme in self.bf_nmes:
                for bpn in self.bfparnmes:  # loop over parallel baseflow res
                    outflow[outnme] += flow['qb'+bpn+'_'+bfnme] * outflow_areas[outnme]['bf'][bfnme]
            # directly applying correction?
            outflow[outnme] = correct_ts(outflow[outnme])
        self.outflow = outflow

    def assign_inflow_and_qpumps(self, qins, qpumps):
        for name in self.if_nmes:
            self.flow['qin_'+name] = qins[name]
        for name in self.bf_nmes:
            for bpn in self.bfparnmes:  # looping over parallel baseflow reservoirs
                qp_key = [key for key in qpumps.keys() if key[-1] == bpn][0]
                self.flow['qpump'+bpn+'_'+name] = qpumps[qp_key]

    def assign_inflow_from_file(self, recharge_dfs):  # not working quite right
        recharge = mikeio.read(recharge_dfs)['exchange between UZ and SZ (pos.up)']
        for ifnme in self.if_nmes:
            mask2d = self.ifres == int(ifnme)
            mask3d = np.broadcast_to(mask2d, recharge.shape)
            rch_if = recharge[mask3d].reshape((recharge.shape[0], np.sum(mask2d)))
            rch_if = np.nansum(rch_if, axis=1)
            factor = mask2d.sum()*1000/60/60/24
            factor = 1000/60/60/24*29859.837464640
            self.flow['qin_'+ifnme] = rch_if*-1 / factor

    def interflow(self, name, par):
        """
        Calculate output from single interflow reservoir.

        The equation numbers refer to:
            "MIKE SHE User Manual, Volume 2: Reference Guide"
            https://manuals.mikepoweredbydhi.help/2017/MIKE_SHE.htm

        Parameters
        ----------
        name : str
            name of interflow reservoir
        par : dict
            dictionary with parameter names as keys.

        Returns
        -------
        None.

        """
        if self.bf_areas is None:
            print('Calculating bf_areas')
            self.extract_bf_areas()

        # initialize arrays
        qin = self.flow['qin_'+name]
        qi = np.zeros(len(qin))  # interflow outflow
        qp = np.zeros(len(qin))  # percolation outflow
        hi = np.zeros(len(qin))  # interflow head
        hi[0] = par['hi0_'+name]
        interflow_part = par['kp_'+name]/(par['ki_'+name]+par['kp_'+name])
        # calculation of head and flow ts
        for i in range(len(qin)-1):
            # calculate flow
            if hi[i] > par['hit_'+name]:  # two outlets
                qp[i] = abs(hi[i]-par['hibot_'+name])/par['kp_'+name]/self.dt  # eq. 12.38
                qi[i] = abs(hi[i]-par['hit_'+name])/par['ki_'+name]/self.dt  # eq. 12.37
            else:  # one outlet
                qp[i] = (hi[i]-par['hibot_'+name])/par['kp_'+name]/self.dt  # eq. 12.38
                qi[i] = 0

            # calculate head
            hi[i+1] = hi[i] + (qin[i]-qi[i]-qp[i])*self.dt/par['isy_'+name]  # eq. 12.39

            if hi[i+1] < par['hibot_'+name]:  # recalculate if below bottom
                hi[i+1] = par['hibot_'+name]
                if hi[i] > par['hit_'+name]:
                    qout = abs(hi[i+1] - hi[i])*par['isy_'+name]/self.dt+qin[i]  # eq. 12.42
                    qi[i] = interflow_part*(qout - par['hit_'+name]/par['kp_'+name]*self.dt/par['isy_'+name])  # eq. 12.43
                    qp[i] = qout - qi[i]  # eq. 12.44
                elif hi[i] > par['hibot_'+name]:
                    qi[i] = 0
                    qp[i] = -(hi[i+1]-hi[i])*par['isy_'+name]/self.dt+qi[i]  # eq. 12.45

        # division of flow between baseflow reservoirs upper (1) and lower (2)
        for bfnme in self.bf_areas[name].keys():
            values = qp * self.bf_areas[name][bfnme]*par['bffrac_'+bfnme]  # eq. 12.46
            self.flow['qv1_'+bfnme] += values
            values = qp * self.bf_areas[name][bfnme]*(1 - par['bffrac_'+bfnme])  # eq. 12.46
            self.flow['qv2_'+bfnme] += values

        # assigning interflow to next interflow reservoir or outflow
        self.flow['qi_'+name] = qi
        if self.if_connects[name]['downstream'] is not None:
            dsif = 'qin_'+self.if_connects[name]['downstream']  # downstream interflow reservoir
            self.flow[dsif] += qi

        # assing head and flow to dataframes
        self.flow['qp_'+name] = qp
        self.head['hi_'+name] = hi

    def baseflow(self, name, par):
        """
        Calculate output from two parallel baseflow reservoirs.

        The equation numbers refer to:
            "MIKE SHE User Manual, Volume 2: Reference Guide"
            https://manuals.mikepoweredbydhi.help/2017/MIKE_SHE.htm

        Parameters
        ----------
        name : str
            name of interflow reservoir
        par : dict
            dictionary with parameter names as keys.

        Returns
        -------
        None.

        """
        for bpn in self.bfparnmes:  # looping over parallel baseflow reservoirs
            # initialize arrays
            qv = self.flow['qv'+bpn+'_'+name]
            qpump = self.flow['qpump'+bpn+'_'+name]
            qb = np.zeros(len(qv))  # baseflow outflow
            hb = np.zeros(len(qv)) + par['hb0_'+name]  # baseflow head
            if np.sum(qv) != 0:  # if bffrac is such that baseflow to reservoir
                # calculation of head and flow ts
                for i in range(len(qv)-1):
                    # calculate flow
                    if (hb[i] > par['hbt_'+name]):
                        qb[i] = (hb[i] - par['hbt_'+name])/par['kb_'+name]/self.dt  # eq. 12.48
                    else:
                        qb[i] = 0
                    # baseflow head (eq. 12.49)
                    hb[i+1] = hb[i] + (qv[i]-qpump[i]-qb[i])*self.dt/par['bsy_'+name]
                    if hb[i+1] < par['hbbot_'+name]:  # recalculate if below bottom
                        hb[i+1] = par['hbbot_'+name]
                        if hb[i] > par['hbt_'+name]:
                            qb[i] = abs(hb[i+1]-hb[i])*par['bsy_'+name]/self.dt+qv[i]  # eq. 12.42
                        else:
                            qb[i] = 0  # new line 24/1

            # assign to dataframe
            self.flow['qb'+bpn+'_'+name] = qb
            self.head['hb'+bpn+'_'+name] = hb

    def run_model(self, par, riv, outs):
        """
        Calculate outflow from all interflow and baseflow reservoirs.

        Parameters
        ----------
        par : dict
            dictionary with parameter names as keys.
        riv : np array
            2D in shape of model grid with indication of position of rivers.
        outs : np array
            2D in shape of model grid with indication of contributing areas for
            runoff station.

        Returns
        -------
        None.

        """
        # running interflow reservoirs (highest grid code first)
        self.if_nmes.sort(key=int)
        self.if_nmes.reverse()
        for ifnme in self.if_nmes:
            self.interflow(name=ifnme, par=par)
        # running baseflow reservoirs
        for bfnme in self.bf_nmes:
            self.baseflow(name=bfnme, par=par)
        # new line (was seperate before)
        self.calc_outflow(outs, riv)


def correct_ts(ts, rolling=2, ffills=10):
    """
    Time series correction to improve performance of linear_reservoir class.

    Parameters
    ----------
    ts : pandas series
        time series
    rolling : int, optional
        roling mean moving window. The default is 2.
    ffills : int, optional
        Only apply correction to ffills following days. The default is 10.

    Returns
    -------
    corrected time series
        time series corrected with above method.

    """
    def ffill_true_once(maske):
        maske[maske-maske.shift(1) == -1] = np.nan
        maske = maske.ffill()
        return maske
    reverse_type = False
    if type(ts) == np.ndarray:
        ts = pd.DataFrame(ts)
        reverse_type = True
    # mask of daily change in flow > 0.1, i.e. only recession curves corrected.
    mask = (ts - ts.shift(-1)) > .1
    # extend mask to include the following ffills days.
    for i in range(ffills):
        mask = ffill_true_once(mask)  # ffill once
    # apply rolling mean to the masked values.
    ts_roll = ts.rolling(rolling).mean()
    ts_corr = ts.copy()
    ts_corr[mask] = ts_roll
    ts_corr = ts_corr.shift(-1)
    if reverse_type:
        ts_corr = ts_corr[0].values
    return ts_corr

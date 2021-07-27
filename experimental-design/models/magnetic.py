import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
plt.rcParams['figure.figsize'] = (9,7)
plt.rcParams['figure.dpi'] = 600

import refl1d.material, refl1d.model, refl1d.probe, refl1d.experiment, refl1d.magnetism
import bumps.parameter, bumps.fitproblem

from simulate import simulate_magnetic, refl1d_experiment
from utils import fisher, Sampler, save_plot
from base import BaseSample, VariableUnderlayer

class SampleYIG(BaseSample, VariableUnderlayer):
    def __init__(self, vary=True):
        self.name = 'YIG_sample'
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'YIG_sample')
        self.labels = ['Up', 'Down']
        
        self.mag_angle = 90
        self.scale = 1.025
        self.bkg = 4e-7
        self.dq = 2.8

        self.Pt_sld = bumps.parameter.Parameter(5.646, name='Pt SLD')
        self.Pt_thick = bumps.parameter.Parameter(21.08, name='Pt Thickness')
        self.Pt_rough = bumps.parameter.Parameter(8.211, name='Air/Pt Roughness')
        self.Pt_mag = bumps.parameter.Parameter(0.0128, name='Pt Magnetic SLD')
        
        self.FePt_sld = bumps.parameter.Parameter(4.678, name='FePt SLD')
        self.FePt_thick = bumps.parameter.Parameter(19.67, name='FePt Thickness')
        self.FePt_rough = bumps.parameter.Parameter(2, name='Pt/FePt Roughness')
        
        self.YIG_sld = bumps.parameter.Parameter(5.836, name='YIG SLD')
        self.YIG_thick = bumps.parameter.Parameter(713.8, name='YIG Thickness')
        self.YIG_rough = bumps.parameter.Parameter(13.55, name='FePt/YIG Roughness')
        self.YIG_mag = bumps.parameter.Parameter(0.349, name='YIG Magnetic SLD')
        
        self.YAG_sld = bumps.parameter.Parameter(5.304, name='YAG SLD')
        self.YAG_rough = bumps.parameter.Parameter(30, name='YIG/YAG Roughness')
                
        self.params = [self.Pt_sld,
                       self.Pt_thick,
                       self.Pt_rough,
                       self.Pt_mag,
                       self.FePt_sld,
                       self.FePt_thick,
                       self.FePt_rough,
                       self.YIG_sld,
                       self.YIG_thick,
                       self.YIG_rough,
                       self.YIG_mag,
                       self.YAG_sld,
                       self.YAG_rough]
        
        if vary:
            self.Pt_sld.range(5, 6)
            self.Pt_thick.range(2, 30)
            self.Pt_rough.range(0, 9)
            self.Pt_mag.range(0, 0.2)
            
            self.FePt_sld.range(4.5, 5.5)
            self.FePt_thick.range(0, 25)
            self.FePt_rough.range(2, 10)
            
            self.YIG_sld.range(5, 6)
            self.YIG_thick.range(100, 900)
            self.YIG_rough.range(0, 70)
            self.YIG_mag.range(0, 0.6)
            
            self.YAG_sld.range(4.5, 5.5)
            self.YAG_rough.range(20, 30)
        
        pt_magnetism = refl1d.magnetism.Magnetism(rhoM=self.Pt_mag, thetaM=self.mag_angle)
        yig_magnetism = refl1d.magnetism.Magnetism(rhoM=self.YIG_mag, thetaM=self.mag_angle)
        
        air = refl1d.material.SLD(rho=0, name='Air')
        Pt = refl1d.material.SLD(rho=self.Pt_sld, name='Pt')(self.Pt_thick, self.Pt_rough, magnetism=pt_magnetism)
        FePt = refl1d.material.SLD(rho=self.FePt_sld, name='FePt')(self.FePt_thick, self.FePt_rough)
        YIG = refl1d.material.SLD(rho=self.YIG_sld, name='YIG')(self.YIG_thick, self.YIG_rough, magnetism=yig_magnetism)
        sub = refl1d.material.SLD(rho=self.YAG_sld, name='Substrate')(0, self.YAG_rough)
        
        self.structure = sub | YIG | FePt | Pt | air
        
        self.__create_experiment()

    def __create_experiment(self):
        pp = refl1d.probe.load4(os.path.join(self.data_path, 'YAG_2_Air.u'), sep='\t', intensity=self.scale, background=self.bkg, columns='Q R dR dQ')
        mm = refl1d.probe.load4(os.path.join(self.data_path, 'YAG_2_Air.d'), sep='\t', intensity=self.scale, background=self.bkg, columns='Q R dR dQ')
        pm = None
        mp = None
        
        self.__set_dq(pp)
        self.__set_dq(mm)
        
        probe = refl1d.probe.PolarizedQProbe(xs=(pp, pm, mp, mm), name='Probe')
        self.experiment = refl1d.experiment.Experiment(sample=self.structure, probe=probe)
    
    def angle_info(self, angle_times, contrasts=None):
        models, datasets = simulate_magnetic(self.structure, angle_times, scale=1, bkg=5e-7, dq=2,
                                             pp=True, pm=False, mp=False, mm=True)

        qs = [data[:,0] for data in datasets]
        counts = [data[:,3] for data in datasets]

        return fisher(qs, self.params, counts, models)
    
    def underlayer_info(self, angle_times, yig_thick, pt_thick):
        pt_thick -= self.Pt_thick.value
        assert pt_thick >= 0
        
        pt_magnetism = refl1d.magnetism.Magnetism(rhoM=self.Pt_mag, thetaM=self.mag_angle)
        yig_magnetism = refl1d.magnetism.Magnetism(rhoM=self.YIG_mag, thetaM=self.mag_angle)
        
        air = refl1d.material.SLD(rho=0, name='Air')
        pt = refl1d.material.SLD(rho=self.Pt_sld, name='Pt')(self.Pt_thick, self.Pt_rough, magnetism=pt_magnetism)
        
        pt_added = refl1d.material.SLD(rho=self.Pt_sld, name='Pt Added')(pt_thick, 0)
        
        intermediate = refl1d.material.SLD(rho=self.FePt_sld, name='FePt')(self.FePt_thick, self.FePt_rough)
        yig = refl1d.material.SLD(rho=self.YIG_sld, name='YIG')(yig_thick, self.YIG_rough, magnetism=yig_magnetism)
        sub = refl1d.material.SLD(rho=self.YAG_sld, name='Substrate')(0, self.YAG_rough)
        
        structure = sub | yig | intermediate | pt_added | pt | air
        
        models, datasets = simulate_magnetic(structure, angle_times, scale=1, bkg=5e-7, dq=2,
                                             pp=True, pm=False, mp=False, mm=True)

        qs = [data[:,0] for data in datasets]
        counts = [data[:,3] for data in datasets]
        
        """
        self.params = [self.Pt_sld,
                       self.Pt_rough,
                       self.Pt_mag,
                       self.FePt_sld,
                       self.FePt_thick,
                       self.FePt_rough,
                       self.YIG_sld,
                       self.YIG_rough,
                       self.YIG_mag,
                       self.sub_sld,
                       self.sub_rough]
        """
        self.params = [self.Pt_mag]
        
        return fisher(qs, self.params, counts, models)
    
    def __set_dq(self, probe):
        # Transform the resolution from refnx to Refl1D format.
        dq = self.dq / (100*np.sqrt(8*np.log(2)))
    
        q_array = probe.Q
    
        # Calculate the dq array and use it to define a Q probe.
        dq_array = probe.Q * dq
        probe.dQ = dq_array
    
        # Adjust probe calculation for constant resolution.
        argmin, argmax = np.argmin(q_array), np.argmax(q_array)
        probe.calc_Qo = np.linspace(q_array[argmin] - 3.5*dq_array[argmin],
                                    q_array[argmax] + 3.5*dq_array[argmax],
                                    21*len(q_array))
   
    def sld_profile(self, save_path):
        """Plots the SLD profile of the sample.

        Args:
            save_path (str): path to directory to save SLD profile to.

        """
        q = np.geomspace(0.005, 0.3, 500)
        scale, bkg, dq = 1, 1e-6, 2
        experiment = refl1d_experiment(self.structure, q, scale, bkg, dq, 0)
        z, slds, _, slds_mag, _ = experiment.magnetic_smooth_profile()
            
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

        # Plot the SLD profile.
        ax.plot(z, slds, label='SLD', color='black')
        ax.plot(z, slds_mag, label='Magnetic SLD', color='red')

        ax.set_xlabel('$\mathregular{Distance\ (\AA)}$', fontsize=11, weight='bold')
        ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
        ax.legend()
        
        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'sld_profile')
   
    def reflectivity_profile(self, save_path):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        colours = ['b', 'g']
        count = 0
        for probe, qr in zip(self.experiment.probe.xs, self.experiment.reflectivity()):
            if qr is not None:
                ax.errorbar(probe.Q, probe.R, probe.dR,
                            marker='o', ms=2, lw=0, elinewidth=0.5, capsize=0.5,
                            label=self.labels[count]+' Data', color=colours[count])
                ax.plot(probe.Q, qr[1], color=colours[count], zorder=20,
                        label=self.labels[count]+' Fitted')
                count += 1
    
        ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
        ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
        ax.set_yscale('log')
        ax.legend(loc='lower left')

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'reflectivity_profile')
    
    def nested_sampling(self, angle_times, save_path, filename, dynamic=False):
        """Runs nested sampling on simulated data of the sample.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            save_path (str): path to directory to save corner plot to.
            filename (str): name of file to save corner plot to.
            dynamic (bool): whether to use static or dynamic nested sampling.

        """
        objective = bumps.fitproblem.FitProblem(self.experiment)
  
        # Sample the objective using nested sampling.
        sampler = Sampler(objective)
        fig = sampler.sample(dynamic=dynamic)

        # Save the sampling corner plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'nested_sampling_'+filename)
        
if __name__ == '__main__':
    save_path = '../results'

    yig_sample = SampleYIG() 
    yig_sample.sld_profile(save_path)    
    yig_sample.reflectivity_profile(save_path)

    plt.close('all')
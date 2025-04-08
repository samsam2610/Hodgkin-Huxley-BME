# -*- coding: utf-8 -*-

import scipy as sp
import numpy as np
import pylab as plt
from scipy.integrate import odeint
import sys

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    """ __init__ uses optional arguments """
    """ when no argument is passed default values are used """

    def __init__(self, C_m=1, gmax_Na=120, gmax_K=36, gmax_L=0.3, E_Na=50,
                 E_K=-77, E_L=-54.387, t_n=450, delta_t=0.01,
                 I_inj_amplitude=0, I_inj_duration=0, I_inj_delay=0, I_inj_frequency=0, I_inj_waveform='rect',
                 vc_delay=10, vc_duration=30, vc_condVoltage=-65,
                 vc_testVoltage=10, vc_returnVoltage=-65, runMode='iclamp',
                 injected_current_plot=False, gating_plot=False, cond_scaling_plot=False,
                 cond_dens_plot=False, driving_force_plot=False,
                 current_plot=False, memb_pot_plot=False):

        self.C_m  = C_m
        """ membrane capacitance, in uF/cm^2 """

        self.gmax_Na = gmax_Na
        """ Sodium (Na) maximum conductances, in mS/cm^2 """

        self.gmax_K  = gmax_K
        """ Postassium (K) maximum conductances, in mS/cm^2 """

        self.gmax_L  = gmax_L
        """ Leak maximum conductances, in mS/cm^2 """

        self.E_Na = E_Na
        """ Sodium (Na) Nernst reversal potentials, in mV """

        self.E_K  = E_K
        """ Postassium (K) Nernst reversal potentials, in mV """

        self.E_L  = E_L
        """ Leak Nernst reversal potentials, in mV """

        self.t    = np.arange(0, t_n, delta_t)
        """ The time to integrate over """

        """ Advanced input - injection current (single rectangular pulse only) """

        self.I_inj_amplitude   = I_inj_amplitude
        """ maximum value or amplitude of injection pulse """

        self.I_inj_duration = I_inj_duration
        """ duration or width of injection pulse """

        self.I_inj_delay = I_inj_delay
        """ start time of injection pulse """
        
        self.I_inj_frequency = I_inj_frequency
        """ frequency of injection pulse """
        
        self.I_inj_waveform = I_inj_waveform
        """ waveform of injection pulse """
        
        #vclamp parameters
        self.run_mode = runMode
        """default is current clamp"""

        self.delay = vc_delay
        """Delay before switching from conditioningVoltage to testingVoltage, in ms"""

        self.duration = vc_duration
        """Duration to hold at testingVoltage, in ms"""

        self.conditioningVoltage = vc_condVoltage
        """Target voltage before time delay, in mV"""

        self.testingVoltage = vc_testVoltage
        """Target voltage between times delay and delay + duration, in mV"""

        self.returnVoltage = vc_returnVoltage
        """Target voltage after time duration, in mV"""

        self.simpleSeriesResistance = 1e7
        """Current will be calculated by the difference in voltage between the target and parent, divided by this value, in mOhm"""

        # plotting conditionals
        self.injected_current_plot = injected_current_plot
        self.gating_plot = gating_plot
        self.cond_scaling_plot = cond_scaling_plot
        self.cond_dens_plot = cond_dens_plot
        self.driving_force_plot = driving_force_plot
        self.current_plot = current_plot
        self.memb_pot_plot = memb_pot_plot

        self.num_plots = (int(self.injected_current_plot) +
                          int(self.gating_plot)+ int(self.cond_scaling_plot) +
                          int(self.cond_dens_plot) + int(self.driving_force_plot) +
                          int(self.current_plot) + int(self.memb_pot_plot))

        self.plot_count = 0

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*np.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*np.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*np.exp(-(V+65) / 80.0)

    def g_Na(self, m, h):
        """
        Conductance density (in mS/cm^2)
        Sodium (Na = element name)

        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.gmax_Na * m**3 * h

    def I_Na(self, V, m, h):
        """
        Membrane current (in uA/cm^2)
        Sodium (Na = element name)

        |  :param V:
        |  :param m:
        |  :param h:
        |  :return:
        """
        return self.g_Na(m, h) * (V - self.E_Na)


    def g_K(self, n):
        """
        Conductance density (in mS/cm^2)
        Potassium (K = element name)

        |  :param n:
        |  :return:
        """
        return self.gmax_K  * n**4

    def I_K(self, V, n):
        """
        Membrane current (in uA/cm^2)
        Potassium (K = element name)

        |  :param V:
        |  :param n:
        |  :return:
        """
        return self.g_K(n) * (V - self.E_K)

    #  Leak
    def I_L(self, V):
        """
        Membrane current (in uA/cm^2)
        Leak

        |  :param V:
        |  :param h:
        |  :return:
        """
        return self.gmax_L * (V - self.E_L)

    def I_inj(self, t):
        """
        External Current

        :param t: time in ms
        :return: current in uA/cm^2
                - Supports waveform types: step, sine, square
        """
        waveform = self.I_inj_waveform.lower()
        A = self.I_inj_amplitude
        delay = self.I_inj_delay
        duration = self.I_inj_duration
        f = self.I_inj_frequency  # in Hz

        if waveform == 'sine':
            # print('Running sine wave injection')
            if delay <= t <= delay + duration:
                return A * np.sin(2 * np.pi * f * (t - delay) / 1000)  # ms to s
            else:
                return 0

        elif waveform == 'square':
            # print('Running square wave injection')
            if delay <= t <= delay + duration:
                return A * np.sign(np.sin(2 * np.pi * f * (t - delay) / 1000))
            else:
                return 0
            
        elif waveform == 'triangle':
            # print('Running triangle wave injection')
            if delay <= t <= delay + duration:
                period = 1000 / f  # ms
                t_mod = (t - delay) % period
                # Triangle wave: ramp from -A to +A and back
                if t_mod < period / 2:
                    return -A + (4 * A / period) * t_mod
                else:
                    return A - (4 * A / period) * (t_mod - period / 2)
            else:
                return 0

        elif waveform == 'biphasic':
            # print('Running biphasic wave injection')
            # One biphasic pulse (positive then negative) within duration
            pulse_width = duration / 2
            if delay <= t < delay + pulse_width:
                return A
            elif delay + pulse_width <= t < delay + 2 * pulse_width:
                return -A
            else:
                return 0
        
        else:
            # print(f"Unknown waveform type '{waveform}', defaulting to step")
            return A * (t > delay) - A * (t > delay + duration)


    def I_inj_vclamp(self,t,v):
        """
        External Current (vclamp)

        |  :param t: time
        |  :return: injector current for voltage clamp
        |
        """
        if   t > (self.delay + self.duration):
            current_A = (self.returnVoltage - v) / self.simpleSeriesResistance
        elif t >= self.delay:
            current_A = (self.testingVoltage - v) / self.simpleSeriesResistance
        elif t < self.delay:
            current_A = (self.conditioningVoltage - v) / self.simpleSeriesResistance
        else:
            print('Problem in injection current calculation for voltage clamp...')
            return 0

        #convert current to current density (uA/cm^2)
        current_uA = current_A*10**6        #convert ampere to micro ampere
        surface_area = 1000*10**-8          #surface area of 1000 um^2 converted to cm^2
        current_density = current_uA/surface_area

        return current_density

    @staticmethod
    def dALLdt(X, t, self):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        V, m, h, n = X
        if self.is_vclamp():
            dVdt = (self.I_inj_vclamp(t,V) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m
        else:
            dVdt = (self.I_inj(t) - self.I_Na(V, m, h) - self.I_K(V, n) - self.I_L(V)) / self.C_m

        dmdt = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m
        dhdt = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h
        dndt = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        return dVdt, dmdt, dhdt, dndt

    def is_vclamp(self):
        return self.run_mode=='vclamp' or self.run_mode=='Voltage Clamp'
    
    def compute_total_charge(self):
        """
        Computes the total absolute charge injected over the simulation
        Returns charge in µC/cm²
        """
        dt = self.t[1] - self.t[0]  # time step in ms

        if self.is_vclamp():
            raise NotImplementedError("Charge calculation only supported for current clamp (iclamp) mode.")

        # Compute I_inj over time
        i_inj_values = np.array([self.I_inj(t) for t in self.t])

        # Integrate absolute value to get total charge
        total_charge = np.sum(np.abs(i_inj_values)) * dt  # µA·ms/cm²

        return total_charge / 1000  # Convert µA·ms to µC (1 ms = 1e-3 s, so 1 µA·ms = 1e-3 µC)

    def is_charge_balanced(self):
        if self.is_vclamp():
            return False  # Not relevant in voltage clamp mode

        i_inj_values = np.array([self.I_inj(t) for t in self.t])
        net_charge = np.trapz(i_inj_values, self.t) / 1000  # µC/cm²
        return np.isclose(net_charge, 0.0, atol=1e-3)
    
    def simulate(self, init_values=[-64.99584, 0.05296, 0.59590, 0.31773]):
        """
        Main simulate method for the Hodgkin Huxley neuron model
        """

        # init_values are the steady state values for v,m,h,n at zero current injection
        X = odeint(self.dALLdt, init_values, self.t, args=(self,))
        V = X[:,0]
        m = X[:,1]
        h = X[:,2]
        n = X[:,3]
        ina = self.I_Na(V, m, h)
        ik = self.I_K(V, n)
        il = self.I_L(V)
        gna = self.g_Na(m, h)
        gk = self.g_K(n)

        # Save some of the data to file
        with open('hh_py_v.dat','w') as f:
            for ti in range(len(self.t)):
                f.write('%s\t%s\n'%(self.t[ti],V[ti]))

        if not '-nogui' in sys.argv:
            #increase figure and font size for display in jupyter notebook
            if __name__ != '__main__':
                plt.rcParams['figure.figsize'] = [7, 7]
                #plt.rcParams['font.size'] = 15
                #plt.rcParams['legend.fontsize'] = 12
                plt.rcParams['legend.loc'] = "upper right"
                #
            else:
                plt.rcParams['figure.figsize'] = [10, 7]

            plt.close()

            fig=plt.figure(figsize=(7, self.num_plots * 2))
            fig.canvas.header_visible = False
            # plt.xlim([np.min(self.t),np.max(self.t)])  #for all subplots

            if self.injected_current_plot:
                ax1 = plt.subplot(self.num_plots,1,self.plot_count + 1)
                plt.title('Simulation of Hodgkin Huxley model neuron')
                if self.is_vclamp():
                    i_inj_values = [self.I_inj_vclamp(t,v) for t,v in zip(self.t,V)]
                else:
                    i_inj_values = [self.I_inj(t) for t in self.t]
                    # Calculate total charge injected
                    total_charge = self.compute_total_charge()
                    # Check if charge is balanced
                    charge_balanced = self.is_charge_balanced()
                    

                if self.is_vclamp(): plt.ylim(-2000,3000)

                plt.plot(self.t, i_inj_values, 'k')
                plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
                
                if total_charge is not None:
                    ax1.text(0.95, 0.95, f'Total Charge = {total_charge:.3f} uC/cm', transform=ax1.transAxes, verticalalignment='top',horizontalalignment='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                if charge_balanced:
                    ax1.text(0.95, 0.80, 'Charge is balanced', transform=ax1.transAxes,verticalalignment='top', horizontalalignment='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                else:
                    ax1.text(0.95, 0.80, 'Charge is NOT balanced',
                             transform=ax1.transAxes,
                                verticalalignment='top',
                                horizontalalignment='right',
                                fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


                self.plot_count += 1


            if self.gating_plot:
                try:
                    plt.subplot(self.num_plots,1,self.plot_count+1, sharex = ax1)
                except NameError:
                    ax1 = plt.subplot(self.num_plots,1,self.plot_count + 1)
                    plt.title('Simulation of Hodgkin Huxley model neuron')
                plt.plot(self.t, m, 'r', label='$Activation Na$')
                plt.plot(self.t, h, 'g', label='$Inactivation Na$')
                plt.plot(self.t, n, 'b', label='$Activation K$')
                plt.ylabel('Gating variable')
                plt.legend()
                self.plot_count += 1

            if self.cond_scaling_plot:
                try:
                    plt.subplot(self.num_plots,1,self.plot_count+1, sharex = ax1)
                except NameError:
                    ax1 = plt.subplot(self.num_plots,1,self.plot_count + 1)
                    plt.title('Simulation of Hodgkin Huxley model neuron')
                scale_na = m*m*m*h
                scale_k = n*n*n*n
                plt.plot(self.t, scale_na, 'Sodium', label='$m^{3}h$')
                plt.plot(self.t, scale_k, 'Potassium', label='$n^{4}$')
                plt.ylabel('Cond scaling')
                plt.legend()
                self.plot_count += 1

            if self.cond_dens_plot:
                try:
                    plt.subplot(self.num_plots,1,self.plot_count+1, sharex = ax1)
                except NameError:
                    ax1 = plt.subplot(self.num_plots,1,self.plot_count + 1)
                    plt.title('Simulation of Hodgkin Huxley model neuron')
                plt.plot(self.t, gna, 'c', label='$g_{Na}$')
                plt.plot(self.t, gk, 'y', label='$g_{K}$')
                plt.ylabel('Cond dens ($mS/cm^2$)')
                plt.legend()
                self.plot_count += 1


            if self.driving_force_plot:
                try:
                    ax_here = plt.subplot(self.num_plots,1,self.plot_count+1, sharex = ax1)
                except NameError:
                    ax1 = plt.subplot(self.num_plots,1,self.plot_count + 1)
                    plt.title('Simulation of Hodgkin Huxley model neuron')
                    ax_here = ax1

                dna = V - self.E_Na
                dk = V - self.E_K
                zero = [0 for v in V]

                #plt.plot(self.t, dna, 'c', label='$V - E_{Na}$')
                ax_here.fill_between(self.t, dna, color='c', alpha=0.5)
                ax_here.fill_between(self.t, dk, color='y', alpha=0.5)

                plt.plot(self.t, dna, 'c', label='$V_{m} - E_{Na}$', linewidth=0.8)
                plt.plot(self.t, dk, 'y', label='$V_{m} - E_{K}$', linewidth=0.8)
                plt.plot(self.t, zero, 'k', linestyle='dashed', linewidth=0.5)
                plt.ylabel('Driving force (mV)')
                plt.legend()
                #if not self.is_vclamp(): plt.ylim(-85,60)
                #plt.ylim(-1, 40)
                self.plot_count += 1

            if self.current_plot:
                try:
                    plt.subplot(self.num_plots,1,self.plot_count+1, sharex = ax1)
                except NameError:
                    ax1 = plt.subplot(self.num_plots,1,self.plot_count + 1)
                    plt.title('Simulation of Hodgkin Huxley model neuron')
                plt.plot(self.t, ina, 'c', label='$I_{Na}$')
                plt.plot(self.t, ik, 'y', label='$I_{K}$')
                plt.plot(self.t, il, 'm', label='$I_{L}$')
                plt.ylabel('Curr dens ($\\mu{A}/cm^2$)')
                plt.legend()
                self.plot_count += 1

            if self.memb_pot_plot:
                try:
                    plt.subplot(self.num_plots,1,self.plot_count+1, sharex = ax1)
                except NameError:
                    ax1 = plt.subplot(self.num_plots,1,self.plot_count + 1)
                    plt.title('Simulation of Hodgkin Huxley model neuron')
                plt.plot(self.t, V, 'k')
                plt.ylabel('$V_{m}$ (mV)')
                plt.xlabel('Time (ms)')
                if not self.is_vclamp(): plt.ylim(-90,60)
                #plt.ylim(-1, 40)
                self.plot_count += 1

            plt.tight_layout()
            plt.show()

if __name__ == '__main__':
    import argparse
    import csv

    parser = argparse.ArgumentParser(description='Run Hodgkin-Huxley Neuron Simulation or Parameter Sweep')

    parser.add_argument('--runMode', type=str, default='iclamp', choices=['iclamp', 'vclamp'],
                        help='Simulation mode: iclamp or vclamp')
    parser.add_argument('--amplitude', type=float, default=20,
                        help='Amplitude of injected current (uA/cm²)')
    parser.add_argument('--delay', type=float, default=100,
                        help='Delay before current starts (ms)')
    parser.add_argument('--duration', type=float, default=200,
                        help='Duration of current injection (ms)')
    parser.add_argument('--frequency', type=float, default=5,
                        help='Frequency for waveforms (Hz)')
    parser.add_argument('--delta_t', type=float, default=0.01,
                        help='Time step (ms)')
    parser.add_argument('--waveform', type=str, default='step',
                        choices=['step', 'sine', 'square', 'triangle', 'biphasic'],
                        help='Waveform type')
    parser.add_argument('--batch', action='store_true',
                        help='Run a batch sweep over amplitudes for each waveform')
    parser.add_argument('--out', type=str, default='hh_sweep_results.csv',
                        help='Output CSV filename for batch mode')

    args = parser.parse_args()

    if args.batch:
        # --- Batch Mode ---
        waveforms = ['step', 'sine', 'square', 'triangle', 'biphasic']
        amplitudes = np.arange(1, 5, 0.2)
        t_n = 400
        threshold_voltage = 0

        results = []

        for waveform in waveforms:
            for amplitude in amplitudes:
                model = HodgkinHuxley(
                    runMode=args.runMode,
                    t_n=t_n,
                    delta_t=args.delta_t,
                    I_inj_amplitude=amplitude,
                    I_inj_delay=args.delay,
                    I_inj_duration=args.duration,
                    I_inj_frequency=args.frequency,
                    I_inj_waveform=waveform
                )

                X = odeint(model.dALLdt, [-64.99584, 0.05296, 0.59590, 0.31773], model.t, args=(model,))
                V = X[:, 0]

                total_charge = model.compute_total_charge()
                charge_balanced = model.is_charge_balanced()
                ap_fired = np.any(V > threshold_voltage)

                results.append({
                    'waveform': waveform,
                    'amplitude': amplitude,
                    'duration': args.duration,
                    'frequency': args.frequency,
                    'charge_uC_per_cm2': total_charge,
                    'charge_balanced': charge_balanced,
                    'ap_fired': ap_fired
                })

        with open(args.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"[✔] Batch sweep complete — results saved to {args.out}")

    else:
        # --- Single Simulation Mode ---
        runner = HodgkinHuxley(
            runMode=args.runMode,
            t_n=450,
            delta_t=args.delta_t,
            I_inj_amplitude=args.amplitude,
            I_inj_delay=args.delay,
            I_inj_duration=args.duration,
            I_inj_frequency=args.frequency,
            I_inj_waveform=args.waveform,
            injected_current_plot=True,
            memb_pot_plot=True,
            gating_plot=True,
        )

        runner.simulate()

        

import numpy as np
import cmath

FAULT_MODELS = ('constant_current', 'k_factor')


class FullConverter:
    def __init__(self, name, bus_1, kVA_nominal, cos_phi, kV_nominal, *,
                 P=None, Q=None, K_scc=1.2, V_min=0.15, K_factor=3.0, status=True,
                 fault_model='constant_current'):
        '''
            fault_model, the current injected during a fault (get_I):
                'constant_current'  the pre-fault current, held: no fault contribution
                'k_factor'          K-factor reactive injection with q-priority current limit
                                    (K_scc, V_min, K_factor are used only by this model)
        '''
        if fault_model not in FAULT_MODELS:
            raise ValueError(f"fault_model must be one of {FAULT_MODELS}, got {fault_model!r}")
        self.fault_model = fault_model
        self.element_type = "full_converter"
        self.name = name
        self.bus_1 = bus_1
        self.P = P
        self.Q = Q
        self.kVA_nominal = kVA_nominal
        self.cos_phi = cos_phi
        self.kV_nominal = kV_nominal
        self.K_scc = K_scc # maximum nominal current multiplier
        self.V_min = V_min
        self.K_factor = K_factor # voltage drop sensitivity
        self.status = status

        sin_phi = (1 - self.cos_phi**2)**0.5

        # compute operating points if not included
        if (self.P is None) and (self.Q is None): # nominal operation
            self.P = self.kVA_nominal * (1e3/100e6) * self.cos_phi 
            self.Q = ((self.kVA_nominal * (1e3/100e6))**2 - self.P**2)**0.5
        elif self.P is None: # only P set, constant cos_phi operation
            if self.cos_phi == 1:
                raise ValueError("inconsistent data, cos_phi=1 with Q != 0 defined.")
            self.P = (self.Q / sin_phi) * self.cos_phi
        elif self.Q is None: # only Q set, constant cos_phi operation
            if self.cos_phi == 0:
                raise ValueError("inconsistent data, cos_phi=0 with P != 0 defined.")
            self.Q = (self.P/self.cos_phi ) * sin_phi

        assert (self.P**2 + self.Q**2)**0.5 <= self.K_scc*(self.kVA_nominal*1e3/100e6), f"{self.name} operation defined acceeds inverter {self.K_scc:.1%} capability."

        self.I_n = self.kVA_nominal / ((3**0.5)*self.kV_nominal)
        self.I_b = 100e6 / ((3**0.5)*(self.kV_nominal*1e3)) # hardcoded 100MVA S_base for now. assumed V_n of conv = V_b

        # maximum converter current output during short circuit in P.U.
        self.I_max = K_scc * (self.I_n/self.I_b)

    def get_Y(self):
        '''
            ideal converter assumption
        '''
        return ([self.bus_1], [self.bus_1], [0])
    
    def get_I(self, V_terminal, V_pre_fault, S_pre_fault):
        '''
            compute full converter current injection under fault
            implemented as K-factor with q-priority, reactive injection
            assumption: V_terminal is phase 'A'

            V_terminal: complex
            V_pre_fault: complex
            S_pre_fault: complex

            V_terminal: actual voltage on load terminal (under fault condition)
            V_pre_fault, S_pre_fault: V and S of load before fault
                (Power flow result or SCADA measurement)
        '''
        # compute pre fault conditions
        I_pre_fault = np.divide(np.conjugate(S_pre_fault), np.conjugate(V_pre_fault))
        if self.fault_model == 'constant_current':
            return I_pre_fault
        theta = cmath.phase(V_terminal)
        deltaV = abs(V_terminal) - abs(V_pre_fault)

        # low voltage ride through
        if abs(deltaV) > (1-self.V_min):
            return 0
        
        # compute park transform for balanced phasors
        I_dq = I_pre_fault * np.exp(-1j * theta)
        i_d = I_dq.real
        i_q = I_dq.imag

        # compute the added i_q/i_n for K-factor
        delta_i_q = self.K_factor * deltaV

        # change for common base (system)
        delta_i_q *= (self.I_n/self.I_b)

        # q_priotization current limitation
        i_q_max = np.clip(i_q + delta_i_q, -self.I_max, self.I_max)
        i_d_max = np.clip(i_d, -(self.I_max**2 - i_q_max**2)**0.5, (self.I_max**2 - i_q_max**2)**0.5)

        I_a = (i_d_max + 1j * i_q_max) * np.exp(1j * theta)

        return I_a


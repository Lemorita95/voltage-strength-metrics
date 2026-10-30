import numpy as np

class ExternalGrid:
    def __init__(self, name, bus_1, kV_nom, *, Z_SC_ohms=None, S_SC_MVA=None, I_SC_kamps=None, status=True):
        '''
            information needed is the short circuit impedance and the current injection
            kV_nom: external grid nominal voltage in the connection point

            one of the following is needed:
                Z_SC_ohms: impedance in ohms
                S_SC_MVA: short circuit capacity in MVA
                I_SC_kamps: short circuit current in kAmperes

            assumption:
                - external grid is unloaded (its not included in the power flow)
                - uses the R/X ratio from IEC 60909
                - external grid voltage is at nominal and = base voltage
                - external grid internal voltage is constant
            reality:
                - can be measured and the current source can be quantified as a 
                    constant current source (similar as a SG)
                - this objects work for both cases
        '''
        self.element_type = "external_grid"
        self.name = name
        self.bus_1 = bus_1
        self.kV_nom = kV_nom
        self.status = status
        
        Z_b = (self.kV_nom**2)/100 # converter to the 100 MVA common base

        if Z_SC_ohms is not None:
            self.Z_series = Z_SC_ohms / Z_b
        elif S_SC_MVA is not None:
            Z = (self.kV_nom**2) / S_SC_MVA
            X = 0.995 * Z
            R = 0.1 * X
            self.Z_series = (R + 1j* X) / Z_b
        elif I_SC_kamps is not None:
            Z = self.kV_nom / ((3**0.5)*I_SC_kamps)
            X = 0.995 * Z
            R = 0.1 * X
            self.Z_series = (R + 1j* X) / Z_b
        else:
            raise ValueError("Unable to compute the grid equivalent. Either Z_SC_ohms, S_SC_MVA or I_SC_kamps must be provided.")

    def get_Y(self):
        rows = [self.bus_1]
        cols = [self.bus_1]
        Ydata = [1/self.Z_series if i == j else -1/self.Z_series for i, j in zip(rows,cols)]

        return (rows, cols, Ydata)
    
    def get_I(self, V_pre_fault, S_pre_fault):
        '''
            compute norton equivalent current injection. This is similar to a SG

            V_pre_fault: complex
            S_pre_fault: complex

            V_pre_fault, S_pre_fault: V and S of generator before fault
                (Power flow result or SCADA measurement)
        '''
        # compute thevenin equivalent
        Eth = V_pre_fault + self.Z_series * np.divide(np.conjugate(S_pre_fault), np.conjugate(V_pre_fault))

        return Eth/self.Z_series
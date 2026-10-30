import numpy as np
import warnings

class Generator:
    def __init__(self, name, bus_1, Z_series, P, *, V=None, Q=None, Q_max=None, Q_min=None, status=True):
        self.element_type = "generator"
        self.name = name
        self.bus_1 = bus_1
        self.Z_series = Z_series
        self.V = V
        self.P = P
        self.Q = Q
        self.Q_max = Q_max
        self.Q_min = Q_min
        self.status = status

        # ideal source warning, let user know when the network contains these
        if self.Z_series == float("inf"):
            warnings.warn(f"{self.name} is defined as an ideal grid equivalent with Z={self.Z_series}", UserWarning)
            print("Execution continues...")

        assert self.P is not None, f"Generator {name} must have a non-null P value."
        assert self.Z_series != 0+1j*0, f"Generator {name} must have a non-zero Z value."

    def get_Y(self):
        rows = [self.bus_1]
        cols = [self.bus_1]
        Ydata = [1/self.Z_series if i == j else -1/self.Z_series for i, j in zip(rows,cols)]
        return (rows, cols, Ydata)
    
    def get_I(self, V_pre_fault, S_pre_fault):
        '''
            compute norton equivalent current injection
            assumption: generator internal voltage is constant

            V_pre_fault: complex
            S_pre_fault: complex

            V_pre_fault, S_pre_fault: V and S of generator before fault
                (Power flow result or SCADA measurement)
        '''
        # generator as ideal grid equivalent, supply the pre fault current only
        if self.Z_series == float("inf"):
            return np.divide(np.conjugate(S_pre_fault), np.conjugate(V_pre_fault))

        # compute generator internal voltage
        Eth = V_pre_fault + self.Z_series * np.divide(np.conjugate(S_pre_fault), np.conjugate(V_pre_fault))

        return Eth/self.Z_series
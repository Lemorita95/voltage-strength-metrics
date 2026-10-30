from cmath import polar, rect

class Load:
    def __init__(self, name, bus_1, *, bus_2=0, P=0.0, Q=0.0, status=True):
        '''
            just shunt loads for now, i.e. bus_2 = 0
            modeled as a constant power
        '''
        self.element_type = "load"
        self.name = name
        self.bus_1 = bus_1
        self.bus_2 = bus_2
        self.P = P
        self.Q = Q
        self.status = status

    def get_Y(self, V_pre_fault, S_pre_fault):
        '''
            compute load impedance given the power flow solution
            if constant impedance, it does not change
        '''

        # properly handle division by zero
        if S_pre_fault == 0+1j*0:
            Z = float("inf")
        else:
            mod_S, arg_S = polar(S_pre_fault)
            Z = rect((abs(V_pre_fault)**2) / mod_S, arg_S)

        return ([self.bus_1], [self.bus_1], [1/Z])
    
    def get_I(self):
        '''
            placeholder for load current source.
            right now not used, constant Z...

            V_terminal: complex
            V_pre_fault: complex
            S_pre_fault: complex

            V_terminal: actual voltage on load terminal (under fault condition)
            V_pre_fault, S_pre_fault: V and S of load before fault
                (Power flow result or SCADA measurement)
        '''
        return 0
    
class Capacitor:
    def __init__(self, name, bus_1, *, bus_2=0, Z_series=0j, Y_shunt=0j, status=True):
        '''
            just shunt capacitor for now, i.e. bus_2 = 0
            modeled as a constant impedance
        '''
        self.name = name
        self.bus_1 = bus_1
        self.bus_2 = bus_2
        self.Z_series = Z_series
        self.Y_shunt = Y_shunt
        self.status = status

    def get_Y(self):
        rows = [self.bus_1]
        cols = [self.bus_1]
        Ydata = [self.Y_shunt if i == j else -self.Y_shunt for i, j in zip(rows,cols)]

        return (rows, cols, Ydata)
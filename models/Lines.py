class Line:
    def __init__(self, name, bus_1, bus_2, Z_series, *, Y_shunt=0j, status=True):
        self.name = name
        self.bus_1 = bus_1
        self.bus_2 = bus_2
        self.Z_series = Z_series
        self.Y_shunt = Y_shunt
        self.status = status


    def get_Y(self):
        rows = [self.bus_1, self.bus_1, self.bus_2, self.bus_2]
        cols = [self.bus_1, self.bus_2, self.bus_1, self.bus_2]
        Ydata = [1/self.Z_series + self.Y_shunt/2 if i == j else -1/self.Z_series for i, j in zip(rows,cols)]
        return (rows, cols, Ydata)

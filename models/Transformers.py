import numpy as np

class Transformer:
    def __init__(self, name, bus_1, bus_2, Z_series, *, \
                Y_shunt=0j, m0=1.0, m_min=None, m_max=None, m_step=None, n0=0, status=True):
        '''
            Two-winding transformer: physics and tap state only. Voltage regulation
            lives in Controls.TapController, which holds references to transformers.

            Arguments (name, bus_1, bus_2 and Z_series are required; the rest are keyword-only):
                Z_series  series impedance, system pu

            Defaults (used when an argument is not given, in a script or in the JSON record):
                Y_shunt = 0j     total magnetizing shunt admittance, system pu (split half/half)
                m0      = 1.0    reference tap ratio, system pu, on the bus_1 side
                m_min   = None   -> m0 (fixed tap)
                m_max   = None   -> m0 (fixed tap)
                m_step  = None   ratio change per tap step; required when m_min < m_max
                n0      = 0      initial tap position, clipped to [n_min, n_max]
                status  = True   in service

            With all defaults the tap is fixed at m0 (n_min = n_max = 0). A tap range
            without a controller is allowed; the tap then stays at n0.
        '''
        self.name = name
        self.bus_1 = bus_1
        self.bus_2 = bus_2
        self.Z_series = Z_series
        self.Y_shunt = Y_shunt
        self.status = status

        # Tap grid: m = m0 + n * m_step, limited to [m_min, m_max]
        self.m0 = m0                    # reference (neutral) ratio, system pu
        self.m_min = m0 if m_min is None else m_min
        self.m_max = m0 if m_max is None else m_max
        self.m_step = m_step
        if self.m_min > self.m_max:
            raise ValueError(f"{name}: m_min={self.m_min} above m_max={self.m_max}")
        if self.m_min < self.m_max and not m_step:
            raise ValueError(f"{name}: tap range [{self.m_min}, {self.m_max}] needs m_step")
        if m_step:
            self.n_min = int(np.ceil((self.m_min - m0) / m_step - 1e-9))
            self.n_max = int(np.floor((self.m_max - m0) / m_step + 1e-9))
        else:
            self.n_min = self.n_max = 0     # fixed tap
        self.n = int(np.clip(n0, self.n_min, self.n_max))   # current tap position

    # ---- tap state -------------------------------------------------
    @property
    def m(self):
        """Current tap ratio (read-only, derived from n)."""
        return self.m0 + self.n * self.m_step if self.n else self.m0

    @property
    def tap_ratio(self):
        m = self.m
        return [1 / m**2, 1 / m, 1 / m, 1]

    def can_step(self, direction):
        """direction = +1 / -1 in m."""
        return self.status and self.n_min <= self.n + direction <= self.n_max

    def step(self, direction):
        if not self.can_step(direction):
            raise ValueError(f"{self.name}: tap limit reached or out of service")
        self.n += direction

    def reset_tap(self, n0=0):
        self.n = int(np.clip(n0, self.n_min, self.n_max))

    # ---- admittance ------------------------------------------------
    def get_Y(self):
        '''
            Y11 = (y + ysh) / m²
            Y12 = -y / m
            Y21 = -y / m
            Y22 =  y + ysh
        '''
        rows = [self.bus_1, self.bus_1, self.bus_2, self.bus_2]
        cols = [self.bus_1, self.bus_2, self.bus_1, self.bus_2]
        if not self.status:
            return rows, cols, [0, 0, 0, 0]

        y = 1 / self.Z_series
        ysh = self.Y_shunt / 2
        base = [y + ysh, -y, -y, y + ysh]
        Ydata = [b * r for b, r in zip(base, self.tap_ratio)]
        return rows, cols, Ydata

    def get_dY_dm(self):
        """d[Y11, Y12, Y21, Y22]/dm, for the voltage sensitivities."""
        y = 1 / self.Z_series
        ysh = self.Y_shunt / 2
        m = self.m
        return [-2 * (y + ysh) / m**3, y / m**2, y / m**2, 0]
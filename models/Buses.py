class Bus:
    '''
        nom_kv              nominal voltage, kV; selects the bus band in VoltageLimits.
                            None (default) -> VoltageLimits fallback band, applied silently;
                            listed by VoltageLimits.unclassified()
        v_min_pu, v_max_pu  optional per-bus band override; None -> band from VoltageLimits ranges
    '''
    def __init__(self, id, name=None, bus_type=None, V=None, angle=None, *,
                 nom_kv=None, v_min_pu=None, v_max_pu=None):
        self.id = id
        self.name = name
        self.bus_type = bus_type
        self.V = V
        self.angle = angle
        self.nom_kv = nom_kv
        self.v_min_pu = v_min_pu
        self.v_max_pu = v_max_pu

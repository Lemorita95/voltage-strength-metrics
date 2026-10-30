import json
import numpy as np
from scipy.sparse import coo_matrix

from .Buses import Bus
from .Lines import Line
from .Capacitors import Capacitor
from .Transformers import Transformer
from .Generators import Generator
from .Loads import Load
from .FullConverter import FullConverter
from .ExternalGrid import ExternalGrid
from .Controls import TapController

def duplicates_error(key, element_dict):
    if key in element_dict:
        raise ValueError(f"duplicated name for '{key}'")


# ---- JSON value converters ---------------------------------------------------
def as_is(value):
    return value


def rx(value):
    ''' {"R": r, "X": x} -> r + jx (missing part = 0) '''
    return float(value.get("R", 0)) + 1j * float(value.get("X", 0))


def gb(value):
    ''' {"G": g, "B": b} -> g + jb (missing part = 0) '''
    return float(value.get("G", 0)) + 1j * float(value.get("B", 0))


def source_z(value):
    ''' generator series impedance; an infinite part marks an ideal source '''
    r, x = float(value["R"]), float(value["X"])
    return float("inf") if float("inf") in (r, x) else r + 1j * x


def rx_or_none(value):
    ''' external-grid ohmic impedance, only if both R and X are given '''
    if value.get("R") is None or value.get("X") is None:
        return None
    return float(value["R"]) + 1j * float(value["X"])


'''
    JSON schema of every element group:
        json group -> (class, name prefix, {json key: (constructor argument, converter)})
    Only keys present (not None) in the JSON are passed on, so every default lives
    in the element class constructor only. A missing required argument raises in
    the constructor. To support a new field: add one row here and one keyword in
    the class.
'''
ELEMENT_SCHEMA = {
    "lines": (Line, "line", {
        "from_bus": ("bus_1", as_is),
        "to_bus": ("bus_2", as_is),
        "z_series": ("Z_series", rx),
        "y_shunt": ("Y_shunt", gb),
        "status": ("status", bool),
    }),
    "capacitors": (Capacitor, "capacitor", {
        "from_bus": ("bus_1", as_is),
        "to_bus": ("bus_2", as_is),
        "z_series": ("Z_series", rx),
        "y_shunt": ("Y_shunt", gb),
        "status": ("status", bool),
    }),
    "transformers": (Transformer, "transformer", {
        "from_bus": ("bus_1", as_is),
        "to_bus": ("bus_2", as_is),
        "z_series": ("Z_series", rx),
        "y_shunt": ("Y_shunt", gb),
        "m0": ("m0", float),
        "m_min": ("m_min", float),
        "m_max": ("m_max", float),
        "m_step": ("m_step", float),
        "n0": ("n0", int),
        "status": ("status", bool),
    }),
    "generators": (Generator, "generator", {
        "from_bus": ("bus_1", as_is),
        "z_series": ("Z_series", source_z),
        "V": ("V", float),
        "P": ("P", float),
        "Q": ("Q", float),
        "Qg_max": ("Q_max", float),
        "Qg_min": ("Q_min", float),
        "status": ("status", bool),
    }),
    "loads": (Load, "load", {
        "from_bus": ("bus_1", as_is),
        "to_bus": ("bus_2", as_is),
        "P": ("P", float),
        "Q": ("Q", float),
        "status": ("status", bool),
    }),
    "full_converters": (FullConverter, "full_converter", {
        "from_bus": ("bus_1", as_is),
        "P": ("P", float),
        "Q": ("Q", float),
        "kVA_nominal": ("kVA_nominal", float),
        "cos_phi": ("cos_phi", float),
        "kV_nominal": ("kV_nominal", float),
        "K_scc": ("K_scc", float),
        "V_min": ("V_min", float),
        "K_factor": ("K_factor", float),
        "status": ("status", bool),
    }),
    "external_grids": (ExternalGrid, "external_grid", {
        "from_bus": ("bus_1", as_is),
        "kV_nom": ("kV_nom", float),
        "Z_SC_ohms": ("Z_SC_ohms", rx_or_none),
        "S_SC_MVA": ("S_SC_MVA", float),
        "I_SC_kamps": ("I_SC_kamps", float),
        "status": ("status", bool),
    }),
}

BUS_SCHEMA = {
    "name": ("name", as_is),
    "bus_type": ("bus_type", as_is),
    "V": ("V", float),
    "angle": ("angle", float),
    "nom_kv": ("nom_kv", float),
}


'''
    JSON schema of every controller group:
        json group -> (class, name prefix, controlled element group, {json key: (constructor argument, converter)})
    Each record needs "name" and "elements" (JSON names of the controlled elements,
    e.g. parallel transformers); every other key is optional and the controller
    class holds the defaults. Controllers are discrete outer-loop devices of the
    power flow. To add one (e.g. switched shunts): one row here and one class in Controls.
'''
CONTROLLER_SCHEMA = {
    "tap_controllers": (TapController, "tap_controller", "transformers", {
        "reg_bus": ("reg_bus", as_is),
        "v_low": ("v_low", float),
        "v_high": ("v_high", float),
        "auto": ("auto", bool),
    }),
}


def schema_kwargs(record, fields):
    ''' constructor keyword arguments for the fields present in a JSON record '''
    return {arg: convert(record[key])
            for key, (arg, convert) in fields.items()
            if record.get(key) is not None}


class CustomNetwork():
    """

    """
    def __init__(self, file):
        self.file = file

    def create_network(self):
        with open(self.file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.buses = {}
        self.slack_id = None
        # element full name -> constructor arguments not given in the JSON (class defaults used)
        self.defaulted = {}

        # buses
        for bus in data.get("buses", []):

            # only record slack id, PV and PQ buses will be inferred on the fly
            if (bus.get('bus_type') or "").lower() == "slack":
                if self.slack_id is not None:
                    raise ValueError("Multiple slack buses defined.")
                self.slack_id = bus["id"]

            duplicates_error(bus.get("id"), self.buses)
            self.buses[bus["id"]] = Bus(bus["id"], **schema_kwargs(bus, BUS_SCHEMA))

        # all elements: one attribute per group, e.g. self.lines, self.generators
        elements = data.get("elements", {})
        for group, (element_class, prefix, fields) in ELEMENT_SCHEMA.items():
            store = {}
            setattr(self, group, store)
            for equipment in elements.get(group, []):
                name = f'{prefix}.{equipment["name"].lower()}'
                duplicates_error(name, store)
                kwargs = schema_kwargs(equipment, fields)
                store[name] = element_class(name=name, **kwargs)
                self.defaulted[name] = sorted(arg for arg, _ in fields.values() if arg not in kwargs)

        # controllers reference elements, so they are built after them
        self.build_controllers(data)

    def build_controllers(self, data):
        '''
            one attribute per controller group from CONTROLLER_SCHEMA, e.g.
            self.tap_controllers = {full name: TapController}. JSON record example:
                {"name": "48-47", "elements": ["Transformer 48-47 ckt 1"],
                 "reg_bus": 48, "v_low": 1.015, "v_high": 1.025, "auto": true}
            Only "name" and "elements" are required; missing keys use the controller
            class defaults. An element belongs to at most one controller of a group;
            elements without one are not controlled (e.g. a transformer keeps its tap).

            In a script, create a controller directly and add it to the group dict:
                net.tap_controllers["tap_controller.x"] = TapController(
                    "tap_controller.x", [net.transformers["transformer.x"]], reg_bus=12, auto=True)
            Without auto = True (JSON or constructor) a controller holds its taps.
        '''
        for group, (controller_class, prefix, element_group, fields) in CONTROLLER_SCHEMA.items():
            store = {}
            setattr(self, group, store)
            elements = getattr(self, element_group)
            owner = {}
            for record in data.get(group, []):
                name = f'{prefix}.{record["name"].lower()}'
                duplicates_error(name, store)
                members = []
                for element_name in record["elements"]:
                    key = f'{ELEMENT_SCHEMA[element_group][1]}.{element_name.lower()}'
                    if key not in elements:
                        raise KeyError(f"{name}: unknown element '{element_name}' in {element_group}")
                    if key in owner:
                        raise ValueError(f"{name}: '{element_name}' is already controlled by {owner[key]}")
                    owner[key] = name
                    members.append(elements[key])
                kwargs = schema_kwargs(record, fields)
                store[name] = controller_class(name, members, **kwargs)
                self.defaulted[name] = sorted(arg for arg, _ in fields.values() if arg not in kwargs)

    def controllers(self):
        ''' every controller of every group, in CONTROLLER_SCHEMA order '''
        return [c for group in CONTROLLER_SCHEMA for c in getattr(self, group).values()]

    def buses_type(self):
        PV_id = set()

        for gen in self.generators.values():
            # if generator is active and not slack
            if gen.status and gen.bus_1 != self.slack_id:
                PV_id.add(gen.bus_1)

        PQ_id = set(self.buses.keys()) - {self.slack_id} - PV_id

        return PQ_id, PV_id
                    
    def compute_bus_net_power(self):
        # initialize buses
        for bus_values in self.buses.values():
            bus_values.P = 0
            bus_values.Q = 0
            bus_values.Qg_min = None
            bus_values.Qg_max = None

        # add generator contributions
        for gen in self.generators.values():
            if not gen.status:
                continue
            bus = self.buses[gen.bus_1]
            if gen.P is not None:
                bus.P += gen.P
            if gen.Q is not None:
                bus.Q += gen.Q
            if gen.Q_min is not None:
                bus.Qg_min = (bus.Qg_min or 0) + gen.Q_min
            if gen.Q_max is not None:
                bus.Qg_max = (bus.Qg_max or 0) + gen.Q_max

        # subtract load contributions
        for load in self.loads.values():
            if not load.status:
                continue
            bus = self.buses[load.bus_1]
            if load.P is not None:
                bus.P -= load.P
            if load.Q is not None:
                bus.Q -= load.Q

        # add converter contributions (negative load)
        for conv in self.full_converters.values():
            if not conv.status:
                continue
            bus = self.buses[conv.bus_1]
            if conv.P is not None:
                bus.P += conv.P
            if conv.Q is not None:
                bus.Q += conv.Q

    def build_ybus(self):
        '''
            build Y bus matrix from the edges (loop each element only once)
        '''
        self.idx = {b: i for i, b in enumerate(self.buses.keys())} # bus number -> python index
        self.id = {i: b for i, b in enumerate(self.buses.keys())} # python index <-> bus number
        n = len(self.buses)
        
        rows = []
        cols = []
        Ydata = []

        '''
            the following elements are added individually to control each ones 
            i want to add to the base Y matrix (power flow)
        '''
        # add each line element
        for line in self.lines.values():
            
            # if status is False (element is off)
            if not line.status:
                continue

            r, c, d = line.get_Y()

            rows.extend(r)
            cols.extend(c)
            Ydata.extend(d)

        # add each capacitor element
        for cap in self.capacitors.values():
            
            # if status is False (element is off)
            if not cap.status:
                continue

            r, c, d = cap.get_Y()

            rows.extend(r)
            cols.extend(c)
            Ydata.extend(d)

        # add each trafo element
        for trafo in self.transformers.values():
            
            # if status is False (element is off)
            if not trafo.status:
                continue
            
            r, c, d = trafo.get_Y()

            rows.extend(r)
            cols.extend(c)
            Ydata.extend(d)

        rows = [self.idx[id] for id in rows] # python index
        cols = [self.idx[id] for id in cols] # python index
        
        self.YBus = coo_matrix((Ydata, (rows, cols)), shape=(n, n)).tocsr()

    def get_active_elements(self, V_complex, active_element_flow, element_status):
        '''
            duplicate keys are already handle in create_network
            (name, bus_idx): (element, v_pre, s_pre)
        '''
        elements = self.loads | self.generators | self.full_converters | self.external_grids

        return {
            (full_name, self.idx[element.bus_1]): (element, V_complex[self.idx[element.bus_1]], active_element_flow[(full_name, self.idx[element.bus_1])])
            for full_name, element in elements.items() 
            if element_status[full_name]}
    
    def change_element_status(self, full_name, status: np.bool):
        '''
            change one generator active power setpoint
        '''
        assert isinstance(status, (np.bool, bool)), "status must be a boolean"

        elements = self.capacitors | self.lines | self.transformers | \
            self.loads | self.generators | self.full_converters | self.external_grids

        elements[full_name.lower()].status = status


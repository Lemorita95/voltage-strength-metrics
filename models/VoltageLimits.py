"""Steady-state bus voltage bands, entered as nominal-voltage ranges.

A VoltageLimits object maps each bus to its acceptable band [v_min_pu, v_max_pu]:
  1. a per-bus override (Bus.v_min_pu / Bus.v_max_pu), if given;
  2. otherwise the first range whose kV interval contains Bus.nom_kv;
  3. otherwise the fallback band (bus listed by `unclassified`).

Missing nominal voltage: a bus with nom_kv = None (no "nom_kv" in the case JSON, or a
blank PowerWorld NomkV) matches no range and silently gets the fallback band, unless it
has its own v_min_pu / v_max_pu. Nothing raises. Call `unclassified(net.buses)` to list
such buses and report them, so a bus is never judged against an unintended band.

A range is {"kv": [kv_low, kv_high], "pu": [v_min_pu, v_max_pu]}, with
kv_low <= nom_kv < kv_high and None for an open end.

Bands are a study / TSO choice, not network data: they are never read from the case
JSON. Select a named preset or pass ranges from the experiment code:
    limits = VoltageLimits.preset("default")
    limits = VoltageLimits([{"kv": [0, 300], "pu": [0.90, 1.05]},
                            {"kv": [300, None], "pu": [0.90, 1.05]}],
                           fallback_pu=[0.90, 1.05])
To add a TSO preset: one entry in PRESETS.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VoltageRange:
    '''Band [v_min_pu, v_max_pu] for buses with kv_low <= nom_kv < kv_high (None = open end).'''
    kv_low: float | None
    kv_high: float | None
    v_min_pu: float
    v_max_pu: float

    def __post_init__(self):
        if self.v_min_pu > self.v_max_pu:
            raise ValueError(f"v_min_pu={self.v_min_pu} above v_max_pu={self.v_max_pu}")
        if self.kv_low is not None and self.kv_high is not None and self.kv_low >= self.kv_high:
            raise ValueError(f"empty kV range [{self.kv_low}, {self.kv_high})")

    @classmethod
    def from_record(cls, record):
        ''' {"kv": [kv_low, kv_high], "pu": [v_min_pu, v_max_pu]} -> VoltageRange '''
        (kv_low, kv_high), (v_min_pu, v_max_pu) = record["kv"], record["pu"]
        return cls(kv_low, kv_high, float(v_min_pu), float(v_max_pu))

    def contains(self, nom_kv):
        if nom_kv is None:
            return False
        low = -math.inf if self.kv_low is None else self.kv_low
        high = math.inf if self.kv_high is None else self.kv_high
        return low <= nom_kv < high


# name -> {"ranges": [...], "fallback_pu": [v_min_pu, v_max_pu]}
PRESETS = {
    # range: [min, max[
    "default": {
        "ranges": [
            {"kv": [0.0, 300.0], "pu": [0.90, 1.05]},
            {"kv": [300.0, None], "pu": [0.90, 1.05]},
        ],
        "fallback_pu": [0.90, 1.05],
    },
}


@dataclass(frozen=True)
class Violation:
    bus: int
    nom_kv: float | None
    v_pu: float
    v_min_pu: float
    v_max_pu: float

    @property
    def deviation_pu(self):
        '''distance outside the band: negative below v_min_pu, positive above v_max_pu.'''
        return self.v_pu - self.v_min_pu if self.v_pu < self.v_min_pu else self.v_pu - self.v_max_pu


class VoltageLimits:
    def __init__(self, ranges, fallback_pu):
        self.ranges = tuple(r if isinstance(r, VoltageRange) else VoltageRange.from_record(r)
                            for r in ranges)
        v_min_pu, v_max_pu = fallback_pu
        self.fallback = VoltageRange(None, None, float(v_min_pu), float(v_max_pu))

    @classmethod
    def preset(cls, name="default"):
        if name not in PRESETS:
            raise KeyError(f"unknown voltage-limit preset '{name}'; available: {sorted(PRESETS)}")
        return cls(**PRESETS[name])

    def range_for(self, nom_kv):
        '''First range containing nom_kv; the fallback band when none does or nom_kv is None.'''
        return next((r for r in self.ranges if r.contains(nom_kv)), self.fallback)

    def band_pu(self, bus):
        '''(v_min_pu, v_max_pu) for a Bus; per-bus values override the range band.'''
        r = self.range_for(bus.nom_kv)
        v_min_pu = r.v_min_pu if bus.v_min_pu is None else bus.v_min_pu
        v_max_pu = r.v_max_pu if bus.v_max_pu is None else bus.v_max_pu
        return v_min_pu, v_max_pu

    def unclassified(self, buses):
        '''
            bus ids that use the fallback band: nom_kv is None or outside every range, and
            v_min_pu / v_max_pu are not both set. Report these; the fallback is applied silently.
        '''
        return [b.id for b in buses.values()
                if not any(r.contains(b.nom_kv) for r in self.ranges)
                and (b.v_min_pu is None or b.v_max_pu is None)]

    def violations(self, net, v_pu, tol_pu=1e-6):
        '''Out-of-band buses for |V| in pu (array indexed like net.idx), worst first.'''
        out = []
        for bus_id, bus in net.buses.items():
            v = float(v_pu[net.idx[bus_id]])
            v_min_pu, v_max_pu = self.band_pu(bus)
            if v < v_min_pu - tol_pu or v > v_max_pu + tol_pu:
                out.append(Violation(bus_id, bus.nom_kv, v, v_min_pu, v_max_pu))
        return sorted(out, key=lambda x: -abs(x.deviation_pu))

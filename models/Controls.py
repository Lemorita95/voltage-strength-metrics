"""Discrete voltage controllers used by the power-flow outer loop.

A TapController owns one transformer, or a group of parallel transformers stepped
together, and keeps the voltage at its regulated bus inside [v_low, v_high].
Transformers hold only physics and tap state; they know nothing about controllers.
A transformer without a controller keeps its tap at n0 (fixed tap).

Tap convention (Transformer.get_Y): the ratio m sits on the bus_1 side, so
V_bus1 ~ m * V_bus2. A +1 step raises the bus_1 voltage relative to bus_2.
"""


class TapController:
    '''
        On-load tap-changer (OLTC) control for one transformer or a parallel group.

        Arguments (only name and transformers are required; the rest are keyword-only):
            name          controller name; CustomNetwork stores it as "tap_controller.<name>"
            transformers  Transformer references, stepped together. All must share the same
                          bus_1 and bus_2 (parallel circuits); each needs a tap range and
                          m_step, otherwise it cannot move.

        Defaults (used when an argument is not given, in a script or in the JSON record):
            reg_bus = None   -> the group's tapped side, bus_1. Must be bus_1 or bus_2;
                                a remote regulated bus is not supported.
            v_low   = 0.95   pu, lower edge of the regulation band at reg_bus
            v_high  = 1.05   pu, upper edge of the regulation band at reg_bus
            auto    = False  taps held at their current position, consistent with the
                             Transformer fixed-tap default. Regulation is on only when set
                             explicitly: auto=True here, "auto": true in the JSON record,
                             or AutoControl = YES in the PowerWorld export.

        Runtime state:
            frozen = False   set True by the power-flow anti-hunting guard; reset by the caller

        Keep v_high - v_low wider than about two tap steps (2 * m_step in pu) to avoid hunting.
    '''

    def __init__(self, name, transformers, *, reg_bus=None, v_low=0.95, v_high=1.05, auto=False):
        transformers = list(transformers)
        if not transformers:
            raise ValueError(f"{name}: no transformers")
        terminals = {(t.bus_1, t.bus_2) for t in transformers}
        if len(terminals) != 1:
            raise ValueError(f"{name}: grouped transformers must share terminals, got {terminals}")
        (bus_1, bus_2), = terminals
        reg_bus = bus_1 if reg_bus is None else reg_bus
        if reg_bus not in (bus_1, bus_2):
            raise ValueError(f"{name}: remote regulated bus {reg_bus} is not supported")
        if v_low > v_high:
            raise ValueError(f"{name}: v_low={v_low} above v_high={v_high}")

        self.name = name
        self.transformers = transformers
        self.bus_1 = bus_1
        self.bus_2 = bus_2
        self.reg_bus = reg_bus
        self.v_low = v_low
        self.v_high = v_high
        self.auto = auto
        self.frozen = False

    @property
    def raise_direction(self):
        '''Tap step (+1 or -1) that raises the regulated-bus voltage: +1 when reg_bus is bus_1.'''
        return 1 if self.reg_bus == self.bus_1 else -1

    @property
    def taps(self):
        '''Current tap position n of each transformer, {transformer name: n}.'''
        return {t.name: t.n for t in self.transformers}

    def action(self, v_reg, held_buses=frozenset()):
        '''
            Tap step for the whole group: +1, -1 or 0.

            v_reg       |V| at reg_bus from the converged power flow, pu
            held_buses  bus ids whose |V| is fixed by the slack or a generator in PV mode

            Returns 0 (no action) when auto is False or the controller is frozen, when
            reg_bus is in held_buses, when v_low <= v_reg <= v_high, or when any
            transformer of the group is out of service or at its tap limit.
        '''
        if not self.auto or self.frozen or self.reg_bus in held_buses:
            return 0
        if v_reg < self.v_low:
            direction = self.raise_direction
        elif v_reg > self.v_high:
            direction = -self.raise_direction
        else:
            return 0
        if all(t.can_step(direction) for t in self.transformers):
            return direction
        return 0

    def apply(self, direction):
        '''Step every transformer of the group by direction (+1 or -1).'''
        for t in self.transformers:
            t.step(direction)

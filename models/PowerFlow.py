import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.sparse import diags

class PowerFlow():
    '''
        run the newton-raphson algorithm to solve the non-linear power flow.

        The power flow solution is not necessary for the SSC methodology, however it gives
        the flexibility of changing the bus voltage other then assuming values.

        it uses the CustomNetwork object to represent the topology of the system
        and the JacobianBuilder object to manage the jacobian values update at each iteration.

        - solve_power_flow() method returns the computed complex voltages (module and angle) when convergence
        - PQ() method compute the bus power injections given the network topology and bus complex voltages
        - ...
    '''

    def PQ(self, YBus, ThetaV):
        n = len(ThetaV) // 2
        theta = ThetaV[:n]
        V = ThetaV[n:]

        P = np.zeros(n)
        Q = np.zeros(n)

        YBus = YBus.tocsr()

        for k in range(n):
            row_start = YBus.indptr[k]
            row_end = YBus.indptr[k + 1]

            # get non-zero entries of k-th row
            cols = YBus.indices[row_start:row_end]
            Y_vals = YBus.data[row_start:row_end]

            G = Y_vals.real
            B = Y_vals.imag

            theta_diff = theta[k] - theta[cols]

            P[k] = V[k] * np.sum(V[cols] * (G * np.cos(theta_diff) + B * np.sin(theta_diff)))
            Q[k] = V[k] * np.sum(V[cols] * (G * np.sin(theta_diff) - B * np.cos(theta_diff)))

        return P, Q

    def _newton_raphson_inner(self, ThetaV, PQVec, YBus, slack_idx, PV_active_idx, max_iter, tol):
        '''
            plain Newton-Raphson for a fixed bus-type set: J . dThetaV = mismatch.
            PV_active_idx: absolute indices of buses whose |V| is held (PV), their
            voltages must already be set in ThetaV.
            returns (ThetaV, converged, iterations); ThetaV is modified in place.
        '''
        n_bus = YBus.shape[0]
        theta_unknown = np.delete(np.arange(n_bus), slack_idx) # exclude slack
        V_unknown = np.setdiff1d(theta_unknown, PV_active_idx) # exclude slack and PV
        unknown_indices = np.concatenate([theta_unknown, n_bus + V_unknown])

        for iteration in range(1, max_iter + 1):
            P_calc, Q_calc = self.PQ(YBus, ThetaV)
            mismatch = PQVec[unknown_indices] - np.concatenate((P_calc, Q_calc))[unknown_indices]

            # convergence check
            if np.all(np.abs(mismatch) < tol):
                return ThetaV, True, iteration

            # solve linear through LU decomposition, slack and PV voltages removed
            J_full = build_jacobian(YBus, ThetaV)
            J_reduced = J_full[np.ix_(unknown_indices, unknown_indices)]
            dThetaV = splu(csc_matrix(J_reduced)).solve(mismatch)
            ThetaV[unknown_indices] += dThetaV

        return ThetaV, False, max_iter

    def newton_raphson(self, ThetaV0, PQVec, YBus, slack_idx, PV_idx, Q_load, Qmin, Qmax,
                       max_iter=100, tol=1e-4, verbose=False, pv_state0=None, Vset=None,
                       max_outer=20, q_tol=1e-6, v_tol=1e-6, max_releases=2):
        '''
            Newton-Raphson power flow with generator reactive-power limits.

            Q limits are enforced in an outer loop, only on converged solutions
            (never on intermediate NR iterates):
              - PV bus with Q_g > Qmax -> PQ at Qmax (state +1)
              - PV bus with Q_g < Qmin -> PQ at Qmin (state -1)
              - PQ bus at Qmax with |V| > Vset, or at Qmin with |V| < Vset -> back to PV
            Each outer pass warm-starts from the previous solution. A generator released
            back to PV more than `max_releases` times stays at its limit (anti-cycling).

            ThetaV0 holds the initial guess; Vset (aligned with PV_idx) holds the PV voltage
            setpoints, defaulting to |V| of ThetaV0 at PV buses.
            Qmin/Qmax are aligned with PV_idx (use -inf/inf for no limit).
            pv_state0: optional initial state per PV bus (0 PV, +1 at Qmax, -1 at Qmin).
            The final state is stored in self.pv_state (dict bus index -> state).
        '''
        if verbose:
            print('\t\tSolving Power Flow using Newton-Raphson method...')
        n_bus = YBus.shape[0]
        ThetaV = ThetaV0.copy()
        PQVec = PQVec.copy()
        PV_idx = np.asarray(PV_idx, dtype=int)
        n_pv = len(PV_idx)
        Vset = ThetaV0[n_bus + PV_idx].copy() if Vset is None else np.asarray(Vset, dtype=float)
        Qmin = np.full(n_pv, -np.inf) if Qmin is None else np.asarray(Qmin, dtype=float)
        Qmax = np.full(n_pv, np.inf) if Qmax is None else np.asarray(Qmax, dtype=float)
        state = np.zeros(n_pv, dtype=int) if pv_state0 is None else np.asarray(pv_state0, dtype=int).copy()
        releases = np.zeros(n_pv, dtype=int)
        self.pv_state = {}
        self.outer_iterations = 0

        for outer in range(1, max_outer + 1):
            self.outer_iterations = outer
            pv = state == 0
            ThetaV[n_bus + PV_idx[pv]] = Vset[pv]
            at_max, at_min = state == 1, state == -1
            PQVec[n_bus + PV_idx[at_max]] = Qmax[at_max] - Q_load[PV_idx[at_max]]
            PQVec[n_bus + PV_idx[at_min]] = Qmin[at_min] - Q_load[PV_idx[at_min]]

            ThetaV, converged, _ = self._newton_raphson_inner(
                ThetaV, PQVec, YBus, slack_idx, PV_idx[pv], max_iter, tol)
            if not converged:
                print(f'\t\t\t> power flow did not converged in {max_iter} iterations')
                return None
            if n_pv == 0:
                break

            # generator Q at PV buses (net injection + load)
            Q_g = self.PQ(YBus, ThetaV)[1][PV_idx] + Q_load[PV_idx]
            V = ThetaV[n_bus + PV_idx]

            new_state = state.copy()
            new_state[pv & (Q_g > Qmax + q_tol)] = 1
            new_state[pv & (Q_g < Qmin - q_tol)] = -1
            release = ((at_max & (V > Vset + v_tol)) | (at_min & (V < Vset - v_tol))) & (releases < max_releases)
            new_state[release] = 0
            releases[release] += 1

            if np.array_equal(new_state, state):
                break
            if verbose:
                changed = PV_idx[new_state != state]
                print(f'\t\t\t... PV/PQ change at bus index {changed}: {state[new_state != state]} -> {new_state[new_state != state]}')
            state = new_state
        else:
            print(f'\t\t\t> Q-limit loop did not settle in {max_outer} passes')
            return None

        self.pv_state = dict(zip(PV_idx.tolist(), state.tolist()))
        return ThetaV[:len(ThetaV)//2], ThetaV[len(ThetaV)//2:]
    
    def solve_power_flow(self, CustomNetwork, verbose=False, warm_start=None, pv_state=None,
                         controls=True, max_control_passes=30, max_reversals=2):
        '''
            power flow with generator Q limits and discrete voltage controllers.

            1. solve with Q limits (newton_raphson outer loop), controlled devices fixed
            2. ask every auto controller (CustomNetwork.controllers(), e.g. tap changers)
               for a step at its regulated bus; apply all steps at once
            3. rebuild YBus and re-solve, warm-started from the previous voltages and
               generator states; repeat until no controller acts

            Defaults:
                controls = True           False keeps every controlled device where it is
                                          (e.g. the ESCR voltage-step probe)
                max_control_passes = 30   controller passes before giving up (warning, last
                                          solution returned)
                max_reversals = 2         a controller whose step direction reverses this many
                                          times is frozen where it is (anti-hunting)

            Controlled devices keep their final positions in CustomNetwork (e.g. transformer
            taps), so a later YBus or short-circuit calculation uses them.

            Records: self.pv_state, self.control_passes, self.control_steps
            (controller name -> net steps), self.frozen_controllers.
            Returns (theta, V), or None if a solve does not converge.
        '''
        CN = CustomNetwork
        self.control_passes = 0
        self.control_steps = {}
        self.frozen_controllers = []
        result = self._solve_fixed_controls(CN, verbose, warm_start, pv_state)
        if result is None or not controls:
            return result

        controllers = [c for c in CN.controllers() if c.auto]
        for c in controllers:
            c.frozen = False
        last_step = {}
        reversals = {}
        for control_pass in range(1, max_control_passes + 1):
            V = result[1]
            # buses whose voltage is already held: slack and generators in PV mode
            held = {CN.slack_id} | {bus for bus, state in self.pv_state.items() if state == 0}
            moved = False
            for c in controllers:
                step = c.action(V[CN.idx[c.reg_bus]], held)
                if not step:
                    continue
                if last_step.get(c.name) == -step:
                    reversals[c.name] = reversals.get(c.name, 0) + 1
                    if reversals[c.name] >= max_reversals:
                        c.frozen = True
                        self.frozen_controllers.append(c.name)
                        continue
                c.apply(step)
                last_step[c.name] = step
                self.control_steps[c.name] = self.control_steps.get(c.name, 0) + step
                moved = True
            if not moved:
                break
            self.control_passes = control_pass
            CN.build_ybus()
            result = self._solve_fixed_controls(CN, verbose, warm_start=result, pv_state=self.pv_state)
            if result is None:
                return None
        else:
            print(f'\t\t\t> voltage controllers did not settle in {max_control_passes} passes')
        return result

    def _solve_fixed_controls(self, CustomNetwork, verbose=False, warm_start=None, pv_state=None):
        '''
            set up the network and the input data for problem solving
            - create the initial values vector (for theta and V) - power flow objective
            - create the target P and Q values vector (`true values`)
            - solve using newton-raphson, with generator Q limits
            controlled devices (e.g. taps) stay at their current positions
        '''
        # i might not need the folowing line, just use the CustomNetwork passed as argument
        CN = CustomNetwork # an instance of a built network with passive YBus calculated
        CN.compute_bus_net_power()
        PQ_id, PV_id = CN.buses_type()

        '''
            if i remove a generator, i need to get the PV bus right -> PQ
        '''

        bus_idx = CN.idx # bus id <> python index mapping
        num_bus = len(bus_idx)
        
        # map network bus ID to numpy matrix index
        slack_idx = bus_idx[CN.slack_id] # python index, always a single value as per assertion
        PV_idx = np.array([bus_idx[n] for n in PV_id], dtype=int) # python index
        PQ_idx = np.array([bus_idx[n] for n in PQ_id], dtype=int) # python index

        # create Theta, V vector
        slack_angle = np.deg2rad(CN.buses[CN.slack_id].angle)
        Theta0 = np.full(num_bus, slack_angle)
        V0 = np.ones(num_bus)
        V0[slack_idx] = CN.buses[CN.slack_id].V # set slack bus voltage magnitude from network data

        # set fixed bus voltages from generators
        for gen in CN.generators.values():
            V0[bus_idx[gen.bus_1]] = gen.V

        # create P, Q vector
        P = np.zeros(num_bus)
        Q = np.zeros(num_bus)

        # create Q load vector Qg_min/max calculation
        Q_load = np.zeros(num_bus)

        for load in CN.loads.values():
            if not load.status:
                continue
            bus_n = load.bus_1
            Q_load[bus_idx[bus_n]] += load.Q

        # PV buses
        if len(PV_id) != 0:
            
            P[PV_idx] = np.array([CN.buses[n].P for n in PV_id])

            # Q limits, aligned with PV_idx; a missing limit is unbounded
            Qmin = np.array([-np.inf if CN.buses[n].Qg_min is None else CN.buses[n].Qg_min for n in PV_id], dtype=float)
            Qmax = np.array([np.inf if CN.buses[n].Qg_max is None else CN.buses[n].Qg_max for n in PV_id], dtype=float)

        else:
            Qmin = Qmax = None

        # PQ buses
        if len(PQ_id) != 0:
            P[PQ_idx] = np.array([CN.buses[n].P for n in PQ_id])
            Q[PQ_idx] = np.array([CN.buses[n].Q for n in PQ_id])
            
        # PV voltage setpoints (before any warm start overwrites the initial guess)
        Vset = V0[PV_idx].copy()

        # optional warm start from a previous solution (theta, V), indexed like CN.idx
        if warm_start is not None:
            theta_ws, v_ws = (np.asarray(x, dtype=float) for x in warm_start)
            if theta_ws.shape != (num_bus,) or v_ws.shape != (num_bus,):
                raise ValueError('warm_start must be (theta, V) arrays for every bus')
            Theta0, V0 = theta_ws.copy(), v_ws.copy()
            Theta0[slack_idx] = slack_angle
            V0[slack_idx] = CN.buses[CN.slack_id].V

        # optional initial PV/PQ state per PV bus id (0 PV, +1 at Qmax, -1 at Qmin)
        pv_state0 = None
        if pv_state is not None:
            pv_state0 = [int(pv_state.get(n, 0)) for n in PV_id]

        # unify vectors
        ThetaV0 = np.concatenate((Theta0, V0))
        PQ = np.concatenate((P, Q))

        # call NR solver, return ThetaV if converged or None otherwise
        result = self.newton_raphson(ThetaV0, PQ, CN.YBus, slack_idx, PV_idx, Q_load, Qmin, Qmax,
                                     tol=1e-9, verbose=verbose, pv_state0=pv_state0, Vset=Vset)
        # final generator state per bus id: 0 PV, +1 at Qmax, -1 at Qmin
        idx_to_bus = {i: n for n, i in bus_idx.items()}
        self.pv_state = {idx_to_bus[i]: s for i, s in self.pv_state.items()}
        return result
    
    def bus_flow(self, CustomNetwork, ThetaV):
        '''
            ThetaV = [Theta, V]
            compute flow from shunt elements, Gen, Loads and Cap/Reactors
            do not compute the flow from line shunt branches, right now...
        '''
        n = len(CustomNetwork.idx)

        # compute net injections
        PQ_net = np.array(self.PQ(CustomNetwork.YBus, ThetaV))

        # load power, loads are lumped at bus
        PQ_load = np.zeros_like(PQ_net)
        for load in CustomNetwork.loads.values():
            if not load.status:
                continue
            bus_n = load.bus_1
            PQ_load[0][CustomNetwork.idx[bus_n]] += load.P
            PQ_load[1][CustomNetwork.idx[bus_n]] += load.Q

        # converter power, converters are lumped at bus
        PQ_converter = np.zeros_like(PQ_net)
        for conv in CustomNetwork.full_converters.values():
            if not conv.status:
                continue
            bus_n = conv.bus_1
            PQ_converter[0][CustomNetwork.idx[bus_n]] += conv.P
            PQ_converter[1][CustomNetwork.idx[bus_n]] += conv.Q

        PQ_gen = PQ_net + PQ_load - PQ_converter

        # for shunt elements (cap/react)
        Ysh_diag = np.zeros(n, dtype=complex)

        for shunt in CustomNetwork.capacitors.values():
            if not shunt.status:
                continue

            rows, cols, data = shunt.get_Y()

            for r, c, y in zip(rows, cols, data):
                Ysh_diag[CustomNetwork.idx[r]] += y

        # Build sparse diagonal YBus for shunts only
        YBus_shunt = diags(Ysh_diag, 0, format='csr')

        # Compute PQ contribution for all shunts at once
        PQ_shunt = np.array(self.PQ(YBus_shunt, ThetaV))

        return (PQ_net, PQ_gen, PQ_converter, PQ_load, PQ_shunt)
    
    def element_flow(self, CustomNetwork, ThetaV):
        '''
            series element power flow
        '''
        n = len(CustomNetwork.idx)
        V_complex = ThetaV[n:] * np.exp(1j*ThetaV[:n])
        element_flow = {}

        element_flow['Lines'] = []
        for name, element in CustomNetwork.lines.items():
            bus_1_idx = CustomNetwork.idx[element.bus_1] # i
            bus_2_idx = CustomNetwork.idx[element.bus_2] # j

            I = np.zeros(n, dtype=complex)

            rows, cols, data = element.get_Y()

            for r, c, y in zip(rows, cols, data):
                I[CustomNetwork.idx[r]] += y * V_complex[CustomNetwork.idx[c]]

            S = V_complex * np.conj(I)

            element_flow['Lines'].append({
                'name': name,
                'i': element.bus_1,
                'j': element.bus_2,
                'Sij': S[bus_1_idx],
                'Sji': S[bus_2_idx]
            })

        element_flow['Capacitors'] = []
        for name, element in CustomNetwork.capacitors.items():
            bus_1_idx = CustomNetwork.idx[element.bus_1] # i

            I = np.zeros(n, dtype=complex)

            rows, cols, data = element.get_Y()

            for r, c, y in zip(rows, cols, data):
                I[CustomNetwork.idx[r]] += y * V_complex[CustomNetwork.idx[c]]

            S = V_complex * np.conj(I)

            element_flow['Capacitors'].append({
                'name': name,
                'i': element.bus_1,
                'j': element.bus_2,
                'Sij': S[bus_1_idx],
                'Sji': -S[bus_1_idx]
            })

        element_flow['Transformers'] = []
        for name, element in CustomNetwork.transformers.items():
            bus_1_idx = CustomNetwork.idx[element.bus_1] # i
            bus_2_idx = CustomNetwork.idx[element.bus_2] # j

            I = np.zeros(n, dtype=complex)

            rows, cols, data = element.get_Y()

            for r, c, y in zip(rows, cols, data):
                I[CustomNetwork.idx[r]] += y * V_complex[CustomNetwork.idx[c]]

            S = V_complex * np.conj(I)

            element_flow['Transformers'].append({
                'name': name,
                'i': element.bus_1,
                'j': element.bus_2,
                'Sij': S[bus_1_idx],
                'Sji': S[bus_2_idx]
            })
        
        return element_flow
    
    def power_flow_summary(self, CustomNetwork, ThetaV):

        net_flow, gen_flow, converter_flow, load_flow, shunt_flow = self.bus_flow(CustomNetwork, ThetaV)

        Theta, V = ThetaV[:len(ThetaV)//2], ThetaV[len(ThetaV)//2:]

        rows = []

        for i, (theta, v, Pnet, Qnet, Pg, Qg, Pconv, Qconv, Pl, Ql, Psh, Qsh) in enumerate(
            zip(Theta, V,
                net_flow[0], net_flow[1],
                gen_flow[0], gen_flow[1],
                converter_flow[0], converter_flow[1],
                load_flow[0], load_flow[1],
                shunt_flow[0], shunt_flow[1])
        ):

            rows.append([
                CustomNetwork.id[i],
                v,
                np.degrees(theta),
                Pnet, Qnet,
                Pg, Qg,
                Pconv, Qconv,
                Pl, Ql,
                Psh, Qsh
            ])

        # --- system totals ---
        total_gen   = sum(r[5]  for r in rows)   # Pg
        total_conv  = sum(r[7]  for r in rows)   # Pconv
        total_load  = sum(r[9] for r in rows)   # Pl
        total_shunt = sum(r[11] for r in rows)   # Psh
        total_losses = total_gen + total_conv - total_load - total_shunt

        totals = {
            "total_gen_MW":    total_gen * 100, # PU > MW
            "total_load_MW":   total_load  * 100, # PU > MW
            "total_shunt_MW":  total_shunt * 100, # PU > MW
            "total_conv_MW":   total_conv * 100, # PU > MW
            "total_losses_MW": total_losses * 100, # PU > MW
            "loss_pct":        abs(total_losses / (total_gen+total_conv)) * 100 if total_gen != 0 else 0,
        }
        
        return rows, totals

    def generator_P_dispatch(self, P_bus, gen_bus):
        '''
            assign bus P flow from power flow to individual generators
            proportional to their rated power. Sync condensers do not get any P
        '''
        # if only one element, no need for dispatch calculation
        if len(gen_bus) == 1:
            gen_bus[0]["P_dispatch"] = P_bus
            return

        total_P = sum(gen["P_weight"] for gen in gen_bus)
        for gen in gen_bus:
            ratio_P = gen["P_weight"] / total_P if total_P > 0 else 0 # if for all synchronous condenser bus
            gen["P_dispatch"] = ratio_P * P_bus

    def generator_Q_dispatch(self, Q_bus, gen_bus, episilon=1e-6):
        '''
            assign bus Q flow from power flow to individual generators
            inversely proportional to their impedance
        '''
        # if only one element, no need for dispatch calculation
        if len(gen_bus) == 1:
            gen_bus[0]["Q_dispatch"] = Q_bus
            return

        total_Y = sum(gen["Q_weight"] for gen in gen_bus)
        for gen in gen_bus:
            # initial alocation
            ratio_Q = gen["Q_weight"] / total_Y
            gen["Q_dispatch"] = ratio_Q * Q_bus

        # if only one generator in bus, no need to make limits check, already
        # enforced in power flow
        if len(gen_bus) == 1:
            return

        # loop through each generator in bus and collect residuals
        for _ in range(len(gen_bus)):
            residual = 0.0
            free = []

            for gen in gen_bus:
                
                # if its already over limit
                if gen["fixed"]:
                    continue

                if gen["Q_dispatch"] < gen["Q_min"]:
                    residual += gen["Q_dispatch"] - gen["Q_min"]
                    gen["Q_dispatch"], gen["fixed"] = gen["Q_min"], True
                elif gen["Q_dispatch"] > gen["Q_max"]:
                    residual += gen["Q_dispatch"] - gen["Q_max"]
                    gen["Q_dispatch"], gen["fixed"] = gen["Q_max"], True
                else:
                    free.append(gen)

            if abs(residual) < episilon or not free:
                break

            free_total = sum(gen["Q_weight"] for gen in free)
            for g in free:
                g["Q_dispatch"] += (g["Q_weight"] / free_total) * residual

    def generators_flow(self, CustomNetwork, gen_bus_flow):
        '''
            compute individual ACTIVE generators power flow
            in case generators are not lumped at bus
            if there is more then one generator at the same bus, make the division between them
            P: proportional to generator P rated
            Q: inversely proportional to |Z|, limited to Q limits
        '''
        state = {
            (full_name, CustomNetwork.idx[gen.bus_1]): {
                "P_weight": gen.P,
                "Q_weight": 1/abs(gen.Z_series),
                "Q_min": gen.Q_min if gen.Q_min is not None else -float("inf"),
                "Q_max": gen.Q_max if gen.Q_max is not None else +float("inf"),
                "P_dispatch": 0,
                "Q_dispatch": 0,
                "fixed": False,
                "breaker_on": gen.status
            }
            for full_name, gen in CustomNetwork.generators.items()
        }

        for bus_idx, (P_bus, Q_bus) in enumerate(zip(gen_bus_flow[0], gen_bus_flow[1])):

            gen_bus = [g for k, g in state.items() if k[1] == bus_idx and g["breaker_on"]]

            # skip dispatch calculation for buses without generators
            if not gen_bus:
                continue

            # make changes directly on state
            self.generator_P_dispatch(P_bus, gen_bus)
            self.generator_Q_dispatch(Q_bus, gen_bus) 

        # gen_flow[(gen.name, bus)] = gen_bus_flow[0][bus] * ratio_P + 1j* gen_Q
        return {k: v["P_dispatch"]+1j*v["Q_dispatch"] for k, v in state.items()}
    
    def loads_flow(self, CustomNetwork, load_bus_flow):
        '''
            compute individual ACTIVE loads power flow
            in case loads are not lumped at bus
            if there is more then one load at the same bus, make the division between them
            P/Q: proportional to load P/Q
        '''
        state = {
            (full_name, CustomNetwork.idx[element.bus_1]): {
                "P_weight": element.P,
                "Q_weight": element.Q,
                "P_dispatch": 0,
                "Q_dispatch": 0,
                "breaker_on": element.status
            }
            for full_name, element in CustomNetwork.loads.items()
        }

        sum_P = np.zeros(load_bus_flow.shape[1])
        sum_Q = np.zeros(load_bus_flow.shape[1])
        for (_, bus_idx), element in state.items():
            if not element["breaker_on"]:
                continue # dont share unconnected elements
            sum_P[bus_idx] += element["P_weight"]
            sum_Q[bus_idx] += element["Q_weight"]

        scale_P =  np.divide(load_bus_flow[0], sum_P, out=np.ones_like(load_bus_flow[0], dtype=float), where=sum_P != 0)
        scale_Q =  np.divide(load_bus_flow[1], sum_Q, out=np.ones_like(load_bus_flow[1], dtype=float), where=sum_Q != 0)

        for (_, bus_idx), element in state.items():
            if not element["breaker_on"]:
                continue # unconnected elements stays zeroed

            element["P_dispatch"] = element["P_weight"] * scale_P[bus_idx]
            element["Q_dispatch"] = element["Q_weight"] * scale_Q[bus_idx]

        return {k: v["P_dispatch"]+1j*v["Q_dispatch"] for k, v in state.items()}
    
    def converters_flow(self, CustomNetwork, converter_bus_flow):
        '''
            compute individual ACTIVE converters power flow
            in case converters are not lumped at bus
            if there is more then one converter at the same bus, make the division between them
            P/Q: proportional to converter P/Q
        '''
        state = {
            (full_name, CustomNetwork.idx[element.bus_1]): {
                "P_weight": element.P,
                "Q_weight": element.Q,
                "P_dispatch": 0,
                "Q_dispatch": 0,
                "breaker_on": element.status
            }
            for full_name, element in CustomNetwork.full_converters.items()
        }

        sum_P = np.zeros(converter_bus_flow.shape[1])
        sum_Q = np.zeros(converter_bus_flow.shape[1])
        for (_, bus_idx), element in state.items():
            if not element["breaker_on"]:
                continue # dont share unconnected elements
            sum_P[bus_idx] += element["P_weight"]
            sum_Q[bus_idx] += element["Q_weight"]

        scale_P =  np.divide(converter_bus_flow[0], sum_P, out=np.ones_like(converter_bus_flow[0], dtype=float), where=sum_P != 0)
        scale_Q =  np.divide(converter_bus_flow[1], sum_Q, out=np.ones_like(converter_bus_flow[1], dtype=float), where=sum_Q != 0)

        for (_, bus_idx), element in state.items():
            if not element["breaker_on"]:
                continue # unconnected elements stays zeroed

            element["P_dispatch"] = element["P_weight"] * scale_P[bus_idx]
            element["Q_dispatch"] = element["Q_weight"] * scale_Q[bus_idx]

        return {k: v["P_dispatch"]+1j*v["Q_dispatch"] for k, v in state.items()}

    def external_grids_flow(self, CustomNetwork, external_grid_bus_flow):
        '''
            only one external grid in a bus
            this function just format the bus flow into the right format
            for feeding into SCC NR and to serve as the SCADA interface
            right now the external grids have no flow exchange to the internal network
        '''

        state = {
            (full_name, CustomNetwork.idx[element.bus_1]): {
                "P_dispatch": 0,
                "Q_dispatch": 0,
                "breaker_on": element.status
            }
            for full_name, element in CustomNetwork.external_grids.items()
        }

        # current implementation: external grids does not exchange power
        if external_grid_bus_flow is not None:
            for (_, bus_idx), element in state.items():
                if not element["breaker_on"]:
                    continue # unconnected elements stays zeroed
                element["P_dispatch"] = external_grid_bus_flow[0][bus_idx]
                element["Q_dispatch"] = external_grid_bus_flow[1][bus_idx]

        return {k: v["P_dispatch"]+1j*v["Q_dispatch"] for k, v in state.items()}

    def get_active_elements_flow(self, CustomNetwork, ThetaV):
        '''
            ThetaV = [Theta, V]
            get the complex power for each active element for instantiation
            it returns even the disconnected elements flow to have compatibility with 
                SCADA endpoints
        '''
        _, gen_flow, converter_flow, load_flow, _ = self.bus_flow(CustomNetwork, ThetaV)

        # get the power for each element as power flow solution output bus values
        load_S = self.loads_flow(CustomNetwork, load_flow)
        gen_S = self.generators_flow(CustomNetwork, gen_flow)
        converter_S = self.converters_flow(CustomNetwork, converter_flow)
        external_grid_S = self.external_grids_flow(CustomNetwork, None) # no power from power flow

        return load_S | gen_S | converter_S | external_grid_S

    def get_active_element_status(self, CustomNetwork):
        ''''
            acquire the status of production facilities
            emulate SCADA data aquisition of these states
        '''
        active_element_status = {}

        # generators
        for full_name, element in CustomNetwork.generators.items():
            active_element_status[full_name] = element.status

        # converters
        for full_name, element in CustomNetwork.full_converters.items():
            active_element_status[full_name] = element.status

        # loads
        for full_name, element in CustomNetwork.loads.items():
            active_element_status[full_name] = element.status

        # external grids
        for full_name, element in CustomNetwork.external_grids.items():
            active_element_status[full_name] = element.status

        return active_element_status


def build_jacobian(YBus, ThetaV):
    '''
        auxiliary function to compute the jacobian to solve the power flow problem
        with newton-raphson

        use explicit formulas for the partial derivatives of P and Q w.r.t. Theta and V
        J1: dP/dTheta
        J2: dP/dV
        J3: dQ/dTheta
        J4: dQ/dV

        the class does not hold any attribute, instead is the skeleton that will update
        the value of the jacobian given a change in the VTheta vector since YBus is fixed.
        
        - VTheta in the newton-raphson is updates at each iteration until convergence.
    '''
    num_bus = YBus.shape[0]
    theta = ThetaV[:num_bus]
    V = ThetaV[num_bus:]

    rows = []
    cols = []
    data = []

    YBus = YBus.tocsr()

    for k in range(num_bus):
        V_k = V[k]
        theta_k = theta[k]

        # for the k-th row of the YBus matrix, get non-zero entries
        row_start = YBus.indptr[k]
        row_end = YBus.indptr[k + 1]

        indices = YBus.indices[row_start:row_end]
        Y_vals = YBus.data[row_start:row_end]

        G = Y_vals.real
        B = Y_vals.imag

        # Diagonal placeholders
        G_kk = 0.0
        B_kk = 0.0

        sum_J1 = 0.0
        sum_J2 = 0.0
        sum_J3 = 0.0
        sum_J4 = 0.0

        for idx, n in enumerate(indices):
            G_kn = G[idx]
            B_kn = B[idx]
            d_theta = theta_k - theta[n]

            if n == k:
                G_kk = G_kn
                B_kk = B_kn
                continue

            # off-diagonal entries

            # J1 (dP/dTheta)
            rows.append(k)
            cols.append(n)
            data.append(V_k * V[n] * (G_kn * np.sin(d_theta) - B_kn * np.cos(d_theta)))

            # J2 (dP/dV)
            rows.append(k)
            cols.append(n + num_bus)
            data.append(V_k * (G_kn * np.cos(d_theta) + B_kn * np.sin(d_theta)))

            # J3 (dQ/dTheta)
            rows.append(k + num_bus)
            cols.append(n)
            data.append(-V_k * V[n] * (G_kn * np.cos(d_theta) + B_kn * np.sin(d_theta)))

            # J4 (dQ/dV)
            rows.append(k + num_bus)
            cols.append(n + num_bus)
            data.append(V_k * (G_kn * np.sin(d_theta) - B_kn * np.cos(d_theta)))

            # accumulate diagonal sums
            sum_J1 += V[n] * (-G_kn * np.sin(d_theta) + B_kn * np.cos(d_theta))
            sum_J2 += V[n] * (G_kn * np.cos(d_theta) + B_kn * np.sin(d_theta))
            sum_J3 += V[n] * (G_kn * np.cos(d_theta) + B_kn * np.sin(d_theta))
            sum_J4 += V[n] * (G_kn * np.sin(d_theta) - B_kn * np.cos(d_theta))

        # --- Diagonal entries ---
        # J1_kk
        rows.append(k)
        cols.append(k)
        data.append(V_k * sum_J1)

        # J2_kk
        rows.append(k)
        cols.append(k + num_bus)
        data.append(2 * V_k * G_kk + sum_J2)

        # J3_kk
        rows.append(k + num_bus)
        cols.append(k)
        data.append(V_k * sum_J3)

        # J4_kk
        rows.append(k + num_bus)
        cols.append(k + num_bus)
        data.append(-2 * V_k * B_kk + sum_J4)

    # Build sparse COO matrix and convert to CSR
    J = coo_matrix((data, (rows, cols)), shape=(2*num_bus, 2*num_bus))
    return J.tocsr()
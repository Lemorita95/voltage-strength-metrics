import numpy as np
from scipy.sparse import csc_matrix


class ShortCircuit:
    
    @staticmethod
    def build_load_task(bus_idx, data):

        src = data[0]
        V_pre = data[1]
        S_pre = data[2]

        def Y(V):
            # V_terminal = V[bus]
            return src.get_Y(V_pre, S_pre)

        def I(V):
            # V_terminal = V[bus]
            return src.get_I()

        return {"bus_idx": bus_idx, "Y": Y, "I": I}
    
    @staticmethod
    def build_generator_task(bus_idx, data):

        src = data[0]
        V_pre = data[1]
        S_pre = data[2]

        def Y(V):
            # V_terminal = V[bus]
            return src.get_Y()

        def I(V):
            # V_terminal = V[bus]
            return src.get_I(V_pre, S_pre)

        return {"bus_idx": bus_idx, "Y": Y, "I": I}

    @staticmethod 
    def build_full_converter_task(bus_idx, data):

        src = data[0]
        V_pre = data[1]
        S_pre = data[2]

        def Y(V):
            # V_terminal = V[bus]
            return src.get_Y()

        def I(V):
            V_terminal = V[bus_idx]
            return src.get_I(V_terminal, V_pre, S_pre)

        return {"bus_idx": bus_idx, "Y": Y, "I": I}
    
    @staticmethod
    def build_external_grid_task(bus_idx, data):

        src = data[0]
        V_pre = data[1]
        S_pre = data[2]

        def Y(V):
            # V_terminal = V[bus]
            return src.get_Y()

        def I(V):
            # V_terminal = V[bus]
            return src.get_I(V_pre, S_pre)

        return {"bus_idx": bus_idx, "Y": Y, "I": I}
    
    @staticmethod
    def SCC_tasks(active_elements):

        tasks = {}

        for (name, bus_idx), element_data in active_elements.items():

            element_type = element_data[0].element_type

            if element_type == "load":
                tasks[(name, bus_idx)] = ShortCircuit.build_load_task(bus_idx, element_data)

            elif element_type == "generator":
                tasks[(name, bus_idx)] = ShortCircuit.build_generator_task(bus_idx, element_data)

            elif element_type == "full_converter":
                tasks[(name, bus_idx)] = ShortCircuit.build_full_converter_task(bus_idx, element_data)

            elif element_type == "external_grid":
                tasks[(name, bus_idx)] = ShortCircuit.build_external_grid_task(bus_idx, element_data)

        return tasks

    @staticmethod
    def build_Y_SCC(V, tasks, Y_passive):
        """
        Assemble the SCC nodal admittance matrix at voltage state V.

        Y_passive must not include the explicit fault admittance if the goal is
        to obtain the converged network Y_SCC for analysis.
        """
        Y_SCC = Y_passive.copy().tocsc()

        rows = []
        cols = []
        data = []

        for t in tasks.values():
            bus_idx = t["bus_idx"]

            _, _, Y_prim = t["Y"](V)

            rows.append(bus_idx)
            cols.append(bus_idx)
            data.extend(Y_prim)

        if data:
            Y_active = csc_matrix(
                (data, (rows, cols)),
                shape=Y_SCC.shape,
            )

            Y_SCC += Y_active

        return Y_SCC

    @staticmethod
    def compute_residual(V, tasks, Y_passive_fault):

        Y_SCC = ShortCircuit.build_Y_SCC(
            V,
            tasks,
            Y_passive_fault,
        )

        I_inj = np.zeros_like(V)

        for t in tasks.values():
            bus_idx = t["bus_idx"]

            I_inj[bus_idx] += t["I"](V)

        return Y_SCC @ V - I_inj

    @staticmethod
    def compute_jacobian(V, tasks, Y_passive_fault, dV=1e-6):

        n = len(V)
        f0 = ShortCircuit.compute_residual(V, tasks, Y_passive_fault)

        J = np.zeros((n, n), dtype=complex)

        for j in range(n):

            Vp = V.copy()
            Vp[j] += dV

            f1 = ShortCircuit.compute_residual(Vp, tasks, Y_passive_fault)

            J[:, j] = (f1 - f0) / dV

        return J

    @staticmethod
    def SCC_NR(fault_bus_idx, V_complex, passive_YBus, active_elements, max_iter=100, tol=1e-6, zf=1e-6, return_details=False):
        '''
            ThetaV: the initial guess. it is the power flow solution
            Y_SCC -> csc sparse: contain all impedances. passive network, generators equivalent and load equivalents
            lin_sources: contains the linear sources elements, i.e. synchronous generators
            nonlin_sources: contains the non linear sources elements, i.e. full converters

            Returns None if the iteration does not converge, like PowerFlow.solve_power_flow;
            the caller decides how to report it.
        '''
        convergence = False

        # initial voltages vector: power flow solution
        V_fault = V_complex.copy()

        # add fault impedance
        Y_passive_fault = passive_YBus.tocsc()
        Y_passive_fault += csc_matrix(([1/zf], ([fault_bus_idx], [fault_bus_idx])), shape=Y_passive_fault.shape)

        residuals = []
        iteration = 0
        while iteration <= max_iter:

            # generate taks with argments for computing Y and I for each active element
            tasks = ShortCircuit.SCC_tasks(active_elements)

            # compute residual
            f = ShortCircuit.compute_residual(V_fault, tasks, Y_passive_fault)
            residuals.append(f)
            # print(f"iter {iteration+1}: ||f|| = {np.linalg.norm(f):.6e}")

            if max(abs(f)) < tol:
                convergence = True
                break

            # compute jacobian
            J = ShortCircuit.compute_jacobian(V_fault, tasks, Y_passive_fault)

            # dV = spsolve(J, -f)
            dV = np.linalg.solve(J, -f)

            # fine tune the damping parameter to help convergence
            alpha = 1.0
            f_norm = np.linalg.norm(f)
            while np.linalg.norm(ShortCircuit.compute_residual(V_fault + alpha * dV, tasks, Y_passive_fault)) > f_norm:
                alpha *= 0.5
                if alpha < 1e-4:
                    break

            V_fault += alpha*dV # damping factor
            iteration += 1

        if not convergence:
            print(f'\t\t\t> short circuit calculation did not converge in {max_iter} iterations '
                  f'for fault at bus index {fault_bus_idx} (last max |residual| {max(abs(f)):.3e})')
            return None

        i_fault = V_fault[fault_bus_idx] / zf

        if return_details:

            # Rebuild Y_SCC at the accepted converged fault-voltage state,
            # but WITHOUT the explicit fault admittance.
            Y_SCC_converged = ShortCircuit.build_Y_SCC(
                V_fault,
                tasks,
                passive_YBus,
            )

            return {
                "I_fault": i_fault,
                "V_fault": V_fault.copy(),
                "Y_SCC": Y_SCC_converged,
            }

        return i_fault
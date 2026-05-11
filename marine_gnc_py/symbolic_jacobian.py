#!/usr/bin/env python3
"""SymPy symbolic process model and Jacobian for the 6-DOF marine vehicle model.

This exposes the continuous-time symbolic process Jacobian A = df/dx for
inspection and provides a lambdified evaluator for EKF covariance propagation.
The current velocity is treated as a known body-frame input [cu, cv, cw] during
linearization. That keeps the symbolic model tractable while still allowing the
EKF to linearize hydrodynamics around the current-relative velocity.
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import sympy as sp

from marine_gnc_py.dynamics_model import VehicleParams


class SymbolicJacobians:
    def __init__(self, params: Optional[VehicleParams] = None, damping_abs_eps: float = 1e-6):
        self.params = params if params is not None else VehicleParams()
        self.damping_abs_eps = float(damping_abs_eps)
        self._built = False
        self._state_symbols = None
        self._tau_symbols = None
        self._current_symbols = None
        self._f_symbolic = None
        self._A_symbolic = None
        self._A_func = None

    @property
    def state_symbols(self):
        if not self._built:
            self._build()
        return self._state_symbols

    @property
    def tau_symbols(self):
        if not self._built:
            self._build()
        return self._tau_symbols

    @property
    def current_symbols(self):
        if not self._built:
            self._build()
        return self._current_symbols

    def symbolic_f(self) -> sp.Matrix:
        if not self._built:
            self._build()
        return self._f_symbolic

    def symbolic_A(self, simplify: bool = False) -> sp.Matrix:
        if not self._built:
            self._build()
        return sp.simplify(self._A_symbolic) if simplify else self._A_symbolic

    def print_symbolic_A(self, simplify: bool = False, latex: bool = False, entry: Optional[tuple[int, int]] = None) -> None:
        A = self.symbolic_A(simplify=simplify)
        obj = A[entry[0], entry[1]] if entry is not None else A
        if latex:
            print(sp.latex(obj))
        else:
            sp.pprint(obj)

    def _build(self) -> None:
        p = self.params
        x, y, z, phi, theta, psi, u, v, w, pp, q, r = sp.symbols("x y z phi theta psi u v w p q r")
        X, Y, Z, K, M, N = sp.symbols("X Y Z K M N")
        cu, cv, cw = sp.symbols("cu cv cw")
        state = sp.Matrix([x, y, z, phi, theta, psi, u, v, w, pp, q, r])
        tau = sp.Matrix([X, Y, Z, K, M, N])
        current = sp.Matrix([cu, cv, cw])

        cphi = sp.cos(phi)
        sphi = sp.sin(phi)
        cth = sp.cos(theta)
        sth = sp.sin(theta)
        cpsi = sp.cos(psi)
        spsi = sp.sin(psi)

        R = sp.Matrix(
            [
                [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
                [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
                [-sth, cth * sphi, cth * cphi],
            ]
        )
        T = sp.Matrix(
            [
                [1, sphi * sth / cth, cphi * sth / cth],
                [0, cphi, -sphi],
                [0, sphi / cth, cphi / cth],
            ]
        )
        eta_dot = sp.Matrix.vstack(R * sp.Matrix([u, v, w]), T * sp.Matrix([pp, q, r]))

        xg, yg, zg = [sp.Float(val) for val in p.r_g]
        xb, yb, zb = [sp.Float(val) for val in p.r_b]
        m = sp.Float(p.m)
        W = sp.Float(p.W)
        B = sp.Float(p.B)
        Ix, Iy, Iz = sp.Float(p.I_x), sp.Float(p.I_y), sp.Float(p.I_z)
        Ixy, Ixz, Iyz = sp.Float(p.I_xy), sp.Float(p.I_xz), sp.Float(p.I_yz)

        M_RB = sp.Matrix(
            [
                [m, 0, 0, 0, m * zg, -m * yg],
                [0, m, 0, -m * zg, 0, m * xg],
                [0, 0, m, m * yg, -m * xg, 0],
                [0, -m * zg, m * yg, Ix, -Ixy, -Ixz],
                [m * zg, 0, -m * xg, -Ixy, Iy, -Iyz],
                [-m * yg, m * xg, 0, -Ixz, -Iyz, Iz],
            ]
        )
        M_A = sp.diag(p.X_u_dot, p.Y_v_dot, p.Z_w_dot, p.K_p_dot, p.M_q_dot, p.N_r_dot)
        Mass = M_RB + M_A

        C_RB = sp.Matrix(
            [
                [0, 0, 0, 0, m * w, -m * v],
                [0, 0, 0, -m * w, 0, m * u],
                [0, 0, 0, m * v, -m * u, 0],
                [0, m * w, -m * v, 0, Iz * r, -Iy * q],
                [-m * w, 0, m * u, -Iz * r, 0, Ix * pp],
                [m * v, -m * u, 0, Iy * q, -Ix * pp, 0],
            ]
        )

        ur = u - cu
        vr = v - cv
        wr = w - cw
        pr = pp
        qr = q
        rr = r
        C_A = sp.Matrix(
            [
                [0, 0, 0, 0, -p.Z_w_dot * wr, p.Y_v_dot * vr],
                [0, 0, 0, p.Z_w_dot * wr, 0, -p.X_u_dot * ur],
                [0, 0, 0, -p.Y_v_dot * vr, p.X_u_dot * ur, 0],
                [0, -p.Z_w_dot * wr, p.Y_v_dot * vr, 0, -p.N_r_dot * rr, p.M_q_dot * qr],
                [p.Z_w_dot * wr, 0, -p.X_u_dot * ur, p.N_r_dot * rr, 0, -p.K_p_dot * pr],
                [-p.Y_v_dot * vr, p.X_u_dot * ur, 0, -p.M_q_dot * qr, p.K_p_dot * pr, 0],
            ]
        )

        def smooth_abs(a):
            return sp.sqrt(a ** 2 + self.damping_abs_eps ** 2)

        D = sp.diag(
            p.X_u + p.X_uu * smooth_abs(ur),
            p.Y_v + p.Y_vv * smooth_abs(vr),
            p.Z_w + p.Z_ww * smooth_abs(wr),
            p.K_p + p.K_pp * smooth_abs(pr),
            p.M_q + p.M_qq * smooth_abs(qr),
            p.N_r + p.N_rr * smooth_abs(rr),
        )
        nu = sp.Matrix([u, v, w, pp, q, r])
        nu_rel = sp.Matrix([ur, vr, wr, pp, q, r])

        g_eta = sp.Matrix(
            [
                (W - B) * sth,
                -(W - B) * cth * sphi,
                -(W - B) * cth * cphi,
                -(yg * W - yb * B) * cth * cphi + (zg * W - zb * B) * cth * sphi,
                (zg * W - zb * B) * sth + (xg * W - xb * B) * cth * cphi,
                -(xg * W - xb * B) * cth * sphi - (yg * W - yb * B) * sth,
            ]
        )

        rhs = tau - C_RB * nu - C_A * nu_rel - D * nu_rel - g_eta
        nu_dot = Mass.LUsolve(rhs)
        f = sp.Matrix.vstack(eta_dot, nu_dot)
        A = f.jacobian(state)

        self._state_symbols = state
        self._tau_symbols = tau
        self._current_symbols = current
        self._f_symbolic = f
        self._A_symbolic = A
        args = list(state) + list(tau) + list(current)
        self._A_func = sp.lambdify(args, A, modules="numpy")
        self._built = True

    def continuous_A(self, state: np.ndarray, tau: np.ndarray, current_body: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._built:
            self._build()
        state = np.asarray(state, dtype=float)
        tau = np.asarray(tau, dtype=float)
        current_body = np.zeros(3, dtype=float) if current_body is None else np.asarray(current_body, dtype=float)
        args = list(state) + list(tau) + list(current_body)
        return np.asarray(self._A_func(*args), dtype=float)

    def discrete_F_euler(self, state: np.ndarray, tau: np.ndarray, dt: float, current_body: Optional[np.ndarray] = None) -> np.ndarray:
        return np.eye(12) + float(dt) * self.continuous_A(state, tau, current_body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print or evaluate the symbolic marine vehicle Jacobian")
    parser.add_argument("--print-symbolic", action="store_true", help="Print symbolic A=df/dx")
    parser.add_argument("--print-f", action="store_true", help="Print symbolic f(x,tau)")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX instead of pretty text")
    parser.add_argument("--simplify", action="store_true", help="Simplify before printing; can be slow")
    parser.add_argument("--entry", nargs=2, type=int, metavar=("ROW", "COL"), help="Print only A[row,col]")
    args = parser.parse_args()

    sj = SymbolicJacobians()
    if args.print_f:
        obj = sj.symbolic_f()
        print(sp.latex(obj) if args.latex else sp.pretty(obj))
    if args.print_symbolic or (not args.print_f):
        entry = tuple(args.entry) if args.entry is not None else None
        sj.print_symbolic_A(simplify=args.simplify, latex=args.latex, entry=entry)


if __name__ == "__main__":
    main()

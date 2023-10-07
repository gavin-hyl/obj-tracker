from sympy import *
from collections import namedtuple
import numpy as np
from utilities import boundary_dvx, averages, COFs


def solver(model):
    m, v1x, dvx, v1y, dvy, I, o1, do, de, X, Y, u, cx = symbols("m v1x dvx v1y dvy I o1 do de X Y u cx")
    eqn_E = Eq(m * (2 * v1x * dvx + dvx ** 2 + 2 * v1y * dvy + dvy ** 2) + I * (2 * o1 * do + do ** 2) - 2 * de, 0)
    eqn_p = Eq(m * (dvx * Y - dvy * X) / I - do, 0)

    if model == "g":
        eqn_3 = Eq((o1 + do) * Y + v1x + dvx, 0)
    elif model == "cx":
        eqn_3 = Eq(dvx, cx)
    else:
        eqn_3 = Eq(dvx, 0)

    results = solve([eqn_E, eqn_p, eqn_3], [dvx, dvy, do], dict=True)
    print(results[0].get(dvy))
    print(results[0].get(dvx))
    print(results[0].get(do))


def get_xyd(angle, l=0.02, r=0.0075):
    angle = np.deg2rad(angle)
    if not (angle < 0.01 or abs(angle - np.pi) < 0.01):
        ab = r / np.sin(angle)
        bc = r / np.tan(angle)
        ob = ab + l / 2
        x = ob * np.cos(angle) - bc
        y = ob * np.sin(angle)
        d = np.sqrt(x ** 2 + y ** 2)
        return x, y, d
    else:
        return l / 2, r, float(np.sqrt(l ** 2 / 4 + r ** 2))


def get_boundaries(experiments: list):
    for experiment_set in experiments:
        u = COFs.get(experiment_set)
        angle = np.rad2deg(np.arctan(1 / u))
        g1, _ = impact_models(v1x=0, v1y=averages.get(experiment_set).v1y, o1=averages.get(experiment_set).o1,
                              v2x=0, v2y=0, o2=0, impact_angle=angle, exp_num=experiment_set, cdvx=False)
        g2, _ = impact_models(v1x=0, v1y=averages.get(experiment_set).v1y, o1=averages.get(experiment_set).o1,
                              v2x=0, v2y=0, o2=0, impact_angle=180 - angle, exp_num=experiment_set, cdvx=False)
        print(f"{experiment_set}: ({angle}, {g1.dvx}, {g2.dvx}),")


def impact_models(v1x, v1y, o1, v2x, v2y, o2, impact_angle, exp_num, cdvx, rho=1130, r=7.5e-3, length=2e-2, dE=0) -> tuple:
    x, y, _ = get_xyd(impact_angle, length, r)
    t_imp = np.rad2deg(np.arctan(y / x))
    if t_imp < 0:
        t_imp += 180

    n = length / r
    m = rho * np.pi * (r ** 3) * (4 / 3 + n)
    I = np.pi * r ** 5 * rho * (n ** 3 / 12 + n ** 2 / 3 + n * 3 / 4 + 8 / 15)

    o1 = float(np.deg2rad(o1))
    o2 = float(np.deg2rad(o2))
    kt1 = m * (v1x ** 2 + v1y ** 2) / 2
    kr1 = I * (o1 ** 2) / 2
    e1 = kt1 + kr1
    kt2 = m * (v2x ** 2 + v2y ** 2) / 2
    kr2 = I * (o2 ** 2) / 2
    e2 = kt2 + kr2

    result = namedtuple("result", "theta dvx dvy do")

    if dE==0:
        d_e = (averages.get(exp_num).E_ret - 1) * e1
    else:
        d_e = dE*e1
    dvy_grip = (I * x * m * o1 - I * m * v1y - x * y * m ** 2 * v1x - y ** 2 * m ** 2 * v1y + np.sqrt(m * (
            I ** 2 * x ** 2 * m * o1 ** 2 - 2 * I ** 2 * x * m * o1 * v1y + I ** 2 * y ** 2 * m * o1 ** 2 + 2 * I **
            2 * y * m * o1 * v1x + 2 * I ** 2 * d_e + I ** 2 * m * v1x ** 2 + I ** 2 * m * v1y ** 2 + I * x ** 2 * y
            ** 2 * m ** 2 * o1 ** 2 + 2 * I * x ** 2 * d_e * m + I * x ** 2 * m ** 2 * v1x ** 2 - 2 * I * x * y ** 2
            * m ** 2 * o1 * v1y + 2 * I * x * y * m ** 2 * v1x * v1y + I * y ** 4 * m ** 2 * o1 ** 2 + 2 * I * y
            ** 3 * m ** 2 * o1 * v1x + 4 * I * y ** 2 * d_e * m + I * y ** 2 * m ** 2 * v1x ** 2 + 2 * I * y ** 2 *
            m ** 2 * v1y ** 2 + 2 * x ** 2 * y ** 2 * d_e * m ** 2 + x ** 2 * y ** 2 * m ** 3 * v1x ** 2 + 2 * x * y
            ** 3 * m ** 3 * v1x * v1y + 2 * y ** 4 * d_e * m ** 2 + y ** 4 * m ** 3 * v1y ** 2))) / (
                       m * (I + x ** 2 * m + y ** 2 * m))
    dvx_grip = (-I ** 2 * y * o1 - I ** 2 * v1x - I * x ** 2 * m * v1x - I * x * y * m * v1y - I * y ** 3 * m * o1 - I *
                y ** 2 * m * v1x - x ** 2 * y ** 2 * m ** 2 * v1x - x * y ** 3 * m ** 2 * v1y + x * y * np.sqrt(
                m * (I ** 2 * x ** 2 * m * o1 ** 2 - 2 * I ** 2 * x * m * o1 * v1y + I ** 2 * y ** 2 * m * o1 **
                     2 + 2 * I ** 2 * y * m * o1 * v1x + 2 * I ** 2 * d_e + I ** 2 * m * v1x ** 2 + I ** 2 * m *
                     v1y ** 2 + I * x ** 2 * y ** 2 * m ** 2 * o1 ** 2 + 2 * I * x ** 2 * d_e * m + I * x ** 2 *
                     m ** 2 * v1x ** 2 - 2 * I * x * y ** 2 * m ** 2 * o1 * v1y + 2 * I * x * y * m ** 2 * v1x *
                     v1y + I * y ** 4 * m ** 2 * o1 ** 2 + 2 * I * y ** 3 * m ** 2 * o1 * v1x + 4 * I * y ** 2 *
                     d_e * m + I * y ** 2 * m ** 2 * v1x ** 2 + 2 * I * y ** 2 * m ** 2 * v1y ** 2 + 2 * x ** 2 *
                     y ** 2 * d_e * m ** 2 + x ** 2 * y ** 2 * m ** 3 * v1x ** 2 + 2 * x * y ** 3 * m ** 3 * v1x
                     * v1y + 2 * y ** 4 * d_e * m ** 2 + y ** 4 * m ** 3 * v1y ** 2))) / \
               (I ** 2 + I * x ** 2 * m + 2 * I * y ** 2 * m + x ** 2 * y ** 2 * m ** 2 + y ** 4 * m ** 2)

    if cdvx:
        boundary_angle = boundary_dvx.get(exp_num).angle
        grip_range = (boundary_angle, 180 - boundary_angle)
        cx = 0
        if impact_angle <= grip_range[0]:
            cx = boundary_dvx.get(exp_num).pos
        elif impact_angle >= grip_range[1]:
            cx = boundary_dvx.get(exp_num).neg
        if not cx == 0:
            dvx_grip = cx
            dvy_grip = (I * x * m * o1 - I * m * v1y + x * y * cx * m ** 2 + np.sqrt(I * m * (
                    I * x ** 2 * m * o1 ** 2 - 2 * I * x * m * o1 * v1y - 2 * I * y * cx * m * o1 - I * cx ** 2 * m - 2
                    * I * cx * m * v1x + 2 * I * d_e + I * m * v1y ** 2 - x ** 2 * cx ** 2 * m ** 2 - 2 * x ** 2 * cx *
                    m ** 2 * v1x + 2 * x ** 2 * d_e * m - 2 * x * y * cx * m ** 2 * v1y - y ** 2 * cx ** 2 * m ** 2))) / (m * (I + x ** 2 * m))
    do_grip = m * (y * dvx_grip - x * dvy_grip) / I
    g = result(impact_angle, dvx_grip, dvy_grip, np.rad2deg(do_grip))

    # testing
    testing = {
        "theta": impact_angle,
        "de": e2 / e1
    }
    return g, testing


if __name__ == "__main__":
    solver(model="g")

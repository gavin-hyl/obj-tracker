from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def get_xyd(angle, l=0.02, r=0.0075):
    angle = np.deg2rad(angle)
    if not (angle < 0.01 or abs(angle - np.pi) < 0.01):
        ab = r / np.sin(angle)
        bc = r / np.tan(angle)
        ob = ab + l / 2
        x = float(ob * np.cos(angle) - bc)
        y = float(ob * np.sin(angle))
        d = float(np.sqrt(x ** 2 + y ** 2))
        return x, y, d
    else:
        return l / 2, r, float(np.sqrt(l ** 2 / 4 + r ** 2))


def impact_models(v1x:float, v1y:float, o1:float, impact_angle:float, u:float, rho=1130, r=7.5e-3, length=2e-2, dE=0, cdvx=True):
    x, y, _ = get_xyd(impact_angle, length, r)

    n = length / r
    m = float(rho * np.pi * (r ** 3) * (4 / 3 + n))
    I = float(np.pi * r ** 5 * rho * (n ** 3 / 12 + n ** 2 / 3 + n * 3 / 4 + 8 / 15))

    o1_rad = float(np.deg2rad(o1))
    kt1 = m * (v1x ** 2 + v1y ** 2) / 2
    kr1 = I * (o1_rad ** 2) / 2
    e1 = kt1 + kr1
    d_e = float(-dE*e1)

    dvy_grip = (I * x * m * o1_rad - I * m * v1y - x * y * m ** 2 * v1x - y ** 2 * m ** 2 * v1y + np.sqrt(m * (
            I ** 2 * x ** 2 * m * o1_rad ** 2 - 2 * I ** 2 * x * m * o1_rad * v1y + I ** 2 * y ** 2 * m * o1_rad ** 2 + 2 * I **
            2 * y * m * o1_rad * v1x + 2 * I ** 2 * d_e + I ** 2 * m * v1x ** 2 + I ** 2 * m * v1y ** 2 + I * x ** 2 * y
            ** 2 * m ** 2 * o1_rad ** 2 + 2 * I * x ** 2 * d_e * m + I * x ** 2 * m ** 2 * v1x ** 2 - 2 * I * x * y ** 2
            * m ** 2 * o1_rad * v1y + 2 * I * x * y * m ** 2 * v1x * v1y + I * y ** 4 * m ** 2 * o1_rad ** 2 + 2 * I * y
            ** 3 * m ** 2 * o1_rad * v1x + 4 * I * y ** 2 * d_e * m + I * y ** 2 * m ** 2 * v1x ** 2 + 2 * I * y ** 2 *
            m ** 2 * v1y ** 2 + 2 * x ** 2 * y ** 2 * d_e * m ** 2 + x ** 2 * y ** 2 * m ** 3 * v1x ** 2 + 2 * x * y
            ** 3 * m ** 3 * v1x * v1y + 2 * y ** 4 * d_e * m ** 2 + y ** 4 * m ** 3 * v1y ** 2))) / (
                       m * (I + x ** 2 * m + y ** 2 * m))
    dvx_grip = (-I ** 2 * y * o1_rad - I ** 2 * v1x - I * x ** 2 * m * v1x - I * x * y * m * v1y - I * y ** 3 * m * o1_rad - I *
                y ** 2 * m * v1x - x ** 2 * y ** 2 * m ** 2 * v1x - x * y ** 3 * m ** 2 * v1y + x * y * np.sqrt(
                m * (I ** 2 * x ** 2 * m * o1_rad ** 2 - 2 * I ** 2 * x * m * o1_rad * v1y + I ** 2 * y ** 2 * m * o1_rad **
                     2 + 2 * I ** 2 * y * m * o1_rad * v1x + 2 * I ** 2 * d_e + I ** 2 * m * v1x ** 2 + I ** 2 * m *
                     v1y ** 2 + I * x ** 2 * y ** 2 * m ** 2 * o1_rad ** 2 + 2 * I * x ** 2 * d_e * m + I * x ** 2 *
                     m ** 2 * v1x ** 2 - 2 * I * x * y ** 2 * m ** 2 * o1_rad * v1y + 2 * I * x * y * m ** 2 * v1x *
                     v1y + I * y ** 4 * m ** 2 * o1_rad ** 2 + 2 * I * y ** 3 * m ** 2 * o1_rad * v1x + 4 * I * y ** 2 *
                     d_e * m + I * y ** 2 * m ** 2 * v1x ** 2 + 2 * I * y ** 2 * m ** 2 * v1y ** 2 + 2 * x ** 2 *
                     y ** 2 * d_e * m ** 2 + x ** 2 * y ** 2 * m ** 3 * v1x ** 2 + 2 * x * y ** 3 * m ** 3 * v1x
                     * v1y + 2 * y ** 4 * d_e * m ** 2 + y ** 4 * m ** 3 * v1y ** 2))) / (I ** 2 + I * x ** 2 * m + 2 * I * y ** 2 * m + x ** 2 * y ** 2 * m ** 2 + y ** 4 * m ** 2)

    if cdvx:
        boundary_angle = np.rad2deg(np.arctan(1/u))
        grip_range = (boundary_angle, 180 - boundary_angle)
        cx = 0
        if impact_angle < grip_range[0]:
            cx = impact_models(v1x=v1x, v1y=v1y, o1=o1, impact_angle=grip_range[0], u=u, rho=rho, r=r, length=length, dE=dE, cdvx=False)[0]
        elif impact_angle > grip_range[1]:
            cx = impact_models(v1x=v1x, v1y=v1y, o1=o1, impact_angle=grip_range[1], u=u, rho=rho, r=r, length=length, dE=dE, cdvx=False)[0]
        if not cx == 0:
            dvx_grip = cx
            dvy_grip = (I * x * m * o1_rad - I * m * v1y + x * y * cx * m ** 2 + np.sqrt(I * m * (
                    I * x ** 2 * m * o1_rad ** 2 - 2 * I * x * m * o1_rad * v1y - 2 * I * y * cx * m * o1_rad - I * cx ** 2 * m - 2
                    * I * cx * m * v1x + 2 * I * d_e + I * m * v1y ** 2 - x ** 2 * cx ** 2 * m ** 2 - 2 * x ** 2 * cx *
                    m ** 2 * v1x + 2 * x ** 2 * d_e * m - 2 * x * y * cx * m ** 2 * v1y - y ** 2 * cx ** 2 * m ** 2))) / (m * (I + x ** 2 * m))
    do_grip = m * (y * dvx_grip - x * dvy_grip) / I
    results = dvx_grip, dvy_grip, np.rad2deg(do_grip)

    return results


fig_sliders, ax_sliders = plt.subplots(8, 1, constrained_layout=True)

v1x_slider = Slider(ax_sliders[0], 'v1x', valmin=-5, valmax=5, valinit=0, valstep=0.1)
v1y_slider = Slider(ax_sliders[1], 'v1y', valmin=-10, valmax=-1, valinit=-3, valstep=0.1)
om1_slider = Slider(ax_sliders[2], 'omega1', valmin=-20, valmax=20, valinit=0, valstep=0.1)
u_slider = Slider(ax_sliders[3], 'mu', valmin=0.1, valmax=1.5, valinit=0.3, valstep=0.05)
rho_slider = Slider(ax_sliders[4], 'rho', valmin=1e3, valmax=3e3, valinit=0, valstep=1)
r_slider = Slider(ax_sliders[5], 'R', valmin=5e-3, valmax=20e-3, valinit=15e-3, valstep=1e-3)
L_slider = Slider(ax_sliders[6], 'L', valmin=5e-3, valmax=40e-3, valinit=20e-3, valstep=1e-3)
dE_slider = Slider(ax_sliders[7], 'dE', valmin=0.1, valmax=0.8, valinit=0.3, valstep=0.05)


sliders = [v1x_slider, v1y_slider, om1_slider, u_slider, rho_slider, r_slider, L_slider, dE_slider]
fig_draw, ax_draw = plt.subplots(2, 2, constrained_layout=True)

alpha = [i+1 for i in range(179)]

def interpolate_sra(x_data, y_data, window=9):
    half_window = int(window/2)
    x_out = x_data[half_window:-half_window]
    y_out = []
    for i in range(half_window, len(y_data)-half_window):
        y_out.append(np.mean(y_data[i-half_window:i+half_window]))
    return x_out, y_out

def update(_):
    vals = [float(s.val) for _,s in enumerate(sliders)]
    results_wrt_a = [impact_models(v1x=vals[0], v1y=vals[1], o1=(vals[2])*360, impact_angle=float(a), u=vals[3], rho=vals[4], r=vals[5], length=vals[6], dE=vals[7], cdvx=True) for a in alpha]
    v2x = [res[0]+vals[0] for _,res in enumerate(results_wrt_a)]
    v2y = [res[1]+vals[1] for _,res in enumerate(results_wrt_a)]
    om2 = [res[2]/360+vals[2] for _,res in enumerate(results_wrt_a)]
    dE = [vals[7]*100 for _,_ in enumerate(results_wrt_a)]
    results = [v2x, v2y, om2, dE]
    labels = ['$v_{2x} (m/s)$', '$v_{2y} (m/s)$', '$\omega_2 (rev/s)$', '$\Delta E (\%)$']
    titles = ['$v_{2x}(\\alpha)$', '$v_{2y}(\\alpha)$', '$\omega_2(\\alpha)$', '$\Delta E(\\alpha)$']

    c = 0
    for i in [0,1]:
        for j in [0,1]:
            ax_draw[i][j].clear()
            ax_draw[i][j].plot(interpolate_sra(alpha, results[c])[0], interpolate_sra(alpha, results[c])[1])
            ax_draw[i][j].set_xlabel("$\\alpha$", fontsize=9)
            ax_draw[i][j].set_ylabel(labels[c], fontsize=9)
            ax_draw[i][j].set_title(titles[c], size=10)
            c+=1

update(0)
v1x_slider.on_changed(update)
v1y_slider.on_changed(update)
om1_slider.on_changed(update)
u_slider.on_changed(update)
rho_slider.on_changed(update)
r_slider.on_changed(update)
L_slider.on_changed(update)
dE_slider.on_changed(update)
plt.show()
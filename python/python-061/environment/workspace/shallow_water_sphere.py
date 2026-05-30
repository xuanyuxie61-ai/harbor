
import numpy as np


EARTH_RADIUS = 6.371e6
GRAVITY = 9.81
OMEGA = 7.2921159e-5
DRAG_COEFF = 1.2e-3


def coriolis_parameter(theta):
    return 2.0 * OMEGA * np.sin(theta)


def compute_cfl_condition(theta, h, u, dt_max_factor=0.8):
    dtheta = np.diff(theta)
    if len(dtheta) == 0:
        return 1.0
    dx = EARTH_RADIUS * np.min(dtheta)
    wave_speed = np.abs(u) + np.sqrt(GRAVITY * np.maximum(h, 0.1))
    max_speed = np.max(wave_speed)
    max_speed = max(max_speed, 1e-6)
    dt_max = dt_max_factor * dx / max_speed
    return dt_max


def shallow_water_rhs(theta, h, hu, hv):
    n = len(theta)
    dtheta = theta[1] - theta[0]
    


    sin_theta = np.sin(theta)
    sin_theta = np.where(sin_theta < 1e-10, 1e-10, sin_theta)
    
    f = coriolis_parameter(theta)
    

    h_safe = np.where(h > 1e-6, h, 1e-6)
    u = hu / h_safe
    v = hv / h_safe
    


    flux_mass = hu * sin_theta
    dflux_mass = np.zeros(n)

    dflux_mass[1:-1] = (flux_mass[2:] - flux_mass[:-2]) / (2.0 * dtheta)

    dflux_mass[0] = (flux_mass[1] - flux_mass[0]) / dtheta
    dflux_mass[-1] = (flux_mass[-1] - flux_mass[-2]) / dtheta
    rhs_h = -dflux_mass / (EARTH_RADIUS * sin_theta)
    


    flux_u = (hu**2 / h_safe) * sin_theta
    dflux_u = np.zeros(n)
    dflux_u[1:-1] = (flux_u[2:] - flux_u[:-2]) / (2.0 * dtheta)
    dflux_u[0] = (flux_u[1] - flux_u[0]) / dtheta
    dflux_u[-1] = (flux_u[-1] - flux_u[-2]) / dtheta
    conv_u = -dflux_u / (EARTH_RADIUS * sin_theta)
    

    dh_dtheta = np.zeros(n)
    dh_dtheta[1:-1] = (h[2:] - h[:-2]) / (2.0 * dtheta)
    dh_dtheta[0] = (h[1] - h[0]) / dtheta
    dh_dtheta[-1] = (h[-1] - h[-2]) / dtheta
    pressure_grad = -GRAVITY * h * dh_dtheta / (EARTH_RADIUS * sin_theta)
    

    coriolis_u = -f * hv
    

    speed = np.sqrt(u**2 + v**2)
    drag_u = -DRAG_COEFF * speed * u
    
    rhs_hu = conv_u + pressure_grad + coriolis_u + drag_u * h_safe
    

    coriolis_v = f * hu
    drag_v = -DRAG_COEFF * speed * v
    rhs_hv = coriolis_v + drag_v * h_safe
    
    return rhs_h, rhs_hu, rhs_hv


def midpoint_explicit_step(theta, h, hu, hv, dt):

    rhs_h1, rhs_hu1, rhs_hv1 = shallow_water_rhs(theta, h, hu, hv)
    

    h_mid = h + 0.5 * dt * rhs_h1
    hu_mid = hu + 0.5 * dt * rhs_hu1
    hv_mid = hv + 0.5 * dt * rhs_hv1
    

    h_mid = np.maximum(h_mid, 0.01)
    
    rhs_h2, rhs_hu2, rhs_hv2 = shallow_water_rhs(theta, h_mid, hu_mid, hv_mid)
    

    h_new = h + dt * rhs_h2
    hu_new = hu + dt * rhs_hu2
    hv_new = hv + dt * rhs_hv2
    
    h_new = np.maximum(h_new, 0.01)
    
    return h_new, hu_new, hv_new


def initialize_typhoon_background(theta, h0=100.0, amplitude=5.0, theta_center=np.pi/2, width=0.15):
    h = h0 - amplitude * np.exp(-((theta - theta_center) / width)**2)
    h = np.maximum(h, h0 - 2.0 * amplitude)
    

    hu = np.zeros_like(theta)
    hv = np.zeros_like(theta)
    
    return h, hu, hv


def solve_shallow_water_sphere(n_theta=180, t_span=(0.0, 86400.0), n_steps=8640):
    theta = np.linspace(0.01, np.pi - 0.01, n_theta)
    h, hu, hv = initialize_typhoon_background(theta)
    
    t0, tf = t_span
    dt_fixed = (tf - t0) / n_steps
    
    t_array = np.zeros(n_steps + 1)
    h_history = np.zeros((n_theta, n_steps + 1))
    hu_history = np.zeros((n_theta, n_steps + 1))
    hv_history = np.zeros((n_theta, n_steps + 1))
    
    h_history[:, 0] = h
    hu_history[:, 0] = hu
    hv_history[:, 0] = hv
    t_array[0] = t0
    
    for i in range(n_steps):

        dt_cfl = compute_cfl_condition(theta, h, hu / np.maximum(h, 1e-6))
        dt = min(dt_fixed, dt_cfl)
        
        h, hu, hv = midpoint_explicit_step(theta, h, hu, hv, dt)
        
        t_array[i + 1] = t_array[i] + dt
        h_history[:, i + 1] = h
        hu_history[:, i + 1] = hu
        hv_history[:, i + 1] = hv
    
    return theta, t_array, h_history, hu_history, hv_history

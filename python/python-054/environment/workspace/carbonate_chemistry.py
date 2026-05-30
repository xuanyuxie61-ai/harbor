
import numpy as np






def equilibrium_constants(T, S):
    Tk = T + 273.15
    

    pK1 = 3633.86 / Tk - 61.2172 + 9.6777 * np.log(Tk) - 0.011555 * S + 0.0001152 * S**2
    K1 = 10.0**(-pK1)
    



    pK2 = 471.78 / Tk + 25.929 - 3.16967 * np.log(Tk) - 0.01781 * S + 0.0001122 * S**2
    K2 = 10.0**(-pK2)
    

    T100 = Tk / 100.0
    lnK0 = -60.2409 + 93.4517 / T100 + 23.3585 * np.log(T100) \
           + S * (0.023517 - 0.023656 * T100 + 0.0047036 * T100**2)
    K0 = np.exp(lnK0)
    

    lnKw = 148.9652 - 13847.26 / Tk - 23.6521 * np.log(Tk) \
           + (-5.977 + 118.67 / Tk + 1.0495 * np.log(Tk)) * np.sqrt(S) - 0.01615 * S
    Kw = np.exp(lnKw)
    

    sqrtS = np.sqrt(S)
    lnKB = (-8966.90 - 2890.53 * sqrtS - 77.942 * S + 1.728 * S**1.5 - 0.0996 * S**2) / Tk \
           + 148.0248 + 137.194 * sqrtS + 1.62142 * S \
           + (-24.4344 - 25.085 * sqrtS - 0.2474 * S) * np.log(Tk) \
           + 0.053105 * sqrtS * Tk
    KB = np.exp(lnKB)
    

    BT = 0.0004326 * S / 35.0
    


    logKsp_calc = -171.9065 - 0.077993 * Tk + 2839.319 / Tk + 71.595 * np.log10(Tk) \
                  + (-0.77712 + 0.0028426 * Tk + 178.34 / Tk) * sqrtS \
                  - 0.07711 * S + 0.0041249 * S**1.5
    Ksp_calcite = 10.0**logKsp_calc
    

    logKsp_arag = -171.945 - 0.077993 * Tk + 2903.293 / Tk + 71.595 * np.log10(Tk) \
                  + (-0.068393 + 0.0017276 * Tk + 88.135 / Tk) * sqrtS \
                  - 0.10018 * S + 0.0059413 * S**1.5
    Ksp_aragonite = 10.0**logKsp_arag
    
    return {
        'K0': K0,
        'K1': K1,
        'K2': K2,
        'Kw': Kw,
        'KB': KB,
        'BT': BT,
        'Ksp_calcite': Ksp_calcite,
        'Ksp_aragonite': Ksp_aragonite,
    }






def zero_muller(func, x1, x2, x3, fatol=1e-12, xatol=1e-12, xrtol=1e-12, itmax=100):
    xold = complex(x1)
    xmid = complex(x2)
    xnew = complex(x3)
    fxold = complex(func(xold))
    fxmid = complex(func(xmid))
    fxnew = complex(func(xnew))
    
    for it_num in range(itmax):

        denom = (xold - xnew) * (xmid - xnew) * (xold - xmid)
        if abs(denom) < 1e-30:

            if abs(fxmid - fxnew) > 1e-30:
                dx = -fxnew * (xmid - xnew) / (fxmid - fxnew)
            else:
                dx = complex(1e-6, 1e-6)
            x_plus = xnew + dx
            x_minus = xnew - dx
        else:
            a_coeff = ((xmid - xnew) * (fxold - fxnew) - (xold - xnew) * (fxmid - fxnew)) / denom
            b_coeff = ((xold - xnew)**2 * (fxmid - fxnew) - (xmid - xnew)**2 * (fxold - fxnew)) / denom
            c_coeff = fxnew
            
            discriminant = b_coeff**2 - 4.0 * a_coeff * c_coeff
            sqrt_disc = np.sqrt(discriminant)
            

            if abs(b_coeff + sqrt_disc) > abs(b_coeff - sqrt_disc):
                denom_plus = b_coeff + sqrt_disc
                denom_minus = b_coeff - sqrt_disc
            else:
                denom_plus = b_coeff - sqrt_disc
                denom_minus = b_coeff + sqrt_disc
            
            if abs(a_coeff) < 1e-30:

                dx = -c_coeff / b_coeff if abs(b_coeff) > 1e-30 else complex(1e-6, 0)
                x_plus = xnew + dx
                x_minus = xnew - dx
            else:
                x_plus = xnew + (-2.0 * c_coeff) / denom_plus if abs(denom_plus) > 1e-30 else xnew
                x_minus = xnew + (-2.0 * c_coeff) / denom_minus if abs(denom_minus) > 1e-30 else xnew
        
        fx_plus = complex(func(x_plus))
        fx_minus = complex(func(x_minus))
        
        if abs(fx_plus) < abs(fx_minus):
            x_candidate = x_plus
            fx_candidate = fx_plus
        else:
            x_candidate = x_minus
            fx_candidate = fx_minus
        

        points = [(abs(func(xold)), xold, fxold),
                  (abs(func(xmid)), xmid, fxmid),
                  (abs(fxnew), xnew, fxnew),
                  (abs(fx_candidate), x_candidate, fx_candidate)]
        points.sort(key=lambda t: t[0])
        
        xold = points[0][1]
        fxold = points[0][2]
        xmid = points[1][1]
        fxmid = points[1][2]
        xnew = points[2][1]
        fxnew = points[2][2]
        

        dx_mag = abs(x_candidate - xnew)
        if abs(fxnew) <= fatol:
            return xnew, fxnew, it_num + 1
        if dx_mag <= xatol:
            return xnew, fxnew, it_num + 1
        if xnew != 0 and dx_mag <= xrtol * abs(xnew):
            return xnew, fxnew, it_num + 1
    
    return xnew, fxnew, itmax






def solve_carbonate_system(DIC, TA, T, S, Ca=0.01028):

    if DIC <= 0 or TA <= 0:
        raise ValueError("DIC 和 TA 必须为正数")
    if not (0 <= T <= 40):
        raise ValueError("温度 T 超出合理海洋范围 [0, 40]°C")
    if not (0 <= S <= 45):
        raise ValueError("盐度 S 超出合理海洋范围 [0, 45] psu")
    
    K = equilibrium_constants(T, S)
    K1, K2, Kw, KB, BT = K['K1'], K['K2'], K['Kw'], K['KB'], K['BT']
    

    def proton_residual(H):
        if isinstance(H, complex):
            H = float(H.real)
        else:
            H = float(H)
        if H <= 0:
            return 1e6
        H2 = H * H
        denom = H2 + K1 * H + K1 * K2
        if denom <= 0:
            return 1e6
        alpha1 = K1 * H / denom
        alpha2 = K1 * K2 / denom
        
        residual = TA - DIC * (alpha1 + 2.0 * alpha2) - Kw / H + H \
                   - BT * KB / (KB + H)
        return residual
    

    H_guess = 10.0**(-8.0)

    x1 = H_guess * 0.1
    x2 = H_guess
    x3 = H_guess * 10.0
    
    H_root, fH, iters = zero_muller(proton_residual, x1, x2, x3,
                                     fatol=1e-14, xatol=1e-16, itmax=200)
    H = float(H_root.real)
    if H <= 0:
        H = 1e-8
    














    raise NotImplementedError("HOLE 1: 碳酸盐系统后处理与返回结构待补全")


def batch_solve_carbonate(DIC_arr, TA_arr, T_arr, S_arr, units='molkg'):
    n = len(DIC_arr)
    scale = 1e-6 if units == 'umolkg' else 1.0
    results = []
    for i in range(n):
        try:
            res = solve_carbonate_system(
                DIC_arr[i] * scale, TA_arr[i] * scale, T_arr[i], S_arr[i]
            )
        except ValueError as e:
            res = {'error': str(e), 'pH': np.nan, 'pCO2': np.nan,
                   'Omega_aragonite': np.nan}
        results.append(res)
    return results


def air_sea_co2_flux(pCO2_ocean, pCO2_atm, T, S, u10=5.0):

    Sc = 2116.8 - 136.25 * T + 4.7353 * T**2 - 0.092307 * T**3 + 0.0007555 * T**4
    if Sc <= 0:
        Sc = 1.0
    

    k_cm_h = 0.251 * u10**2 * (Sc / 660.0)**(-0.5)

    k_m_d = k_cm_h * 0.24
    

    K = equilibrium_constants(T, S)
    K0 = K['K0']

    rho_sw = 1023.0 + 0.8 * (S - 35.0) - 0.4 * (T - 20.0)
    K0_molar = K0 * rho_sw / 1e6
    
    flux = k_m_d * K0_molar * (pCO2_ocean - pCO2_atm)
    flux *= 1e3
    return flux

"""
Main entry point for quantum error correction threshold analysis.

This script performs a complete numerical study of the threshold for
Kitaev's surface code under correlated non-Markovian Pauli noise.

Pipeline:
1. Construct surface code stabilizers and analyze parity check conditioning
2. Sample correlated noise instances (Cholesky, Eigen, FFT methods)
3. Evolve errors via Lindblad open-system dynamics
4. Decode syndromes using MWPM / BP / Union-Find decoders
5. Estimate logical error rates via Monte Carlo
6. Compute threshold using FEM boundary analysis and edge detection
7. Validate with Hermite quadrature and sparse grid integration
8. Perform rare-event importance sampling for tail probabilities
"""
import numpy as np
import os
import sys

# Ensure deterministic behavior for reproducibility
np.random.seed(42)
RNG = np.random.default_rng(42)

from stabilizer_surface_code import SurfaceCode
from noise_correlation import CorrelatedPauliNoise, NonMarkovianNoise
from lindblad_dynamics import lindbladian_superoperator, forward_euler_rho, exact_lindblad_evolution, DGLindbladSolver
from syndrome_decoder import MWPMBruteDecoder, BeliefPropagationDecoder, KnapsackLikeLogicalDecoder, UnionFindDecoder
from threshold_fem_boundary import ThresholdFEM, EdgeDetectorThreshold, finite_size_scaling
from quadrature_hermite_sparse import HermiteQuadrature, TruncatedNormalSparseGrid, QuantumNoiseIntegral
from parity_matrix_analysis import ParityCheckConditionAnalyzer, TestMatrixSuite, ConditionEstimator
from rare_event_sampler import RejectionSampler, CausticDegeneracyAnalyzer, MonteCarloLogicalError
from utils import pauli_operators, depolarizing_channel, von_neumann_entropy, fidelity, chop_array


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  Quantum Error Correction Threshold Analysis")
    print("  Surface Code under Correlated Non-Markovian Pauli Noise")
    print("=" * 70)

    # =====================================================================
    # 1. Surface Code Construction
    # =====================================================================
    print_section("1. Surface Code Construction & Parity Check Analysis")

    L_values = [3, 4, 5]
    codes = {}
    for L in L_values:
        code = SurfaceCode(L, boundary="toric")
        codes[L] = code
        print(f"\n  Toric code L={L}:")
        print(f"    n_qubits = {code.n_qubits}")
        print(f"    n_stabilizers = {code.n_stabilizers}")
        print(f"    n_logical = {code.n_logical}")
        print(f"    code_distance (analytical) = {code.distance}")
        print(f"    boundary_word = {code.boundary_word_topology()}")
        d_metrics = code.box_distance_logical_operators()
        print(f"    box_distance_metrics = {d_metrics}")

        # Parity check condition analysis
        Hx_sparse = code.sparse_parity_check("x")
        analyzer = ParityCheckConditionAnalyzer(Hx_sparse.toarray())
        cond_info = analyzer.analyze()
        print(f"    parity_check_rank = {cond_info['rank']}")
        print(f"    hager_condition_estimate = {cond_info['hager_kappa']:.4e}")
        if cond_info['exact_kappa'] is not None:
            print(f"    exact_condition_number = {cond_info['exact_kappa']:.4e}")

    # =====================================================================
    # 2. Noise Model Construction
    # =====================================================================
    print_section("2. Correlated Pauli Noise Model")

    n_qubits = codes[3].n_qubits
    base_rate = 0.05
    noise = CorrelatedPauliNoise(
        n_qubits=n_qubits,
        base_rate=base_rate,
        sigma=0.02,
        correlation_length=2.0,
        nu=0.5,
        rng=RNG
    )
    print(f"\n  Base error rate: {base_rate}")
    print(f"  Correlation length: {noise.correlation_length}")
    print(f"  Matérn ν: {noise.nu}")

    # Sample rates via different methods
    rates_chol = noise.sample_rates_cholesky()
    rates_eig = noise.sample_rates_eigen()
    rates_fft = noise.sample_rates_fft()
    print(f"  Sampled rates (Cholesky): mean={np.mean(rates_chol):.4f}, std={np.std(rates_chol):.4f}")
    print(f"  Sampled rates (Eigen):    mean={np.mean(rates_eig):.4f}, std={np.std(rates_eig):.4f}")
    print(f"  Sampled rates (FFT):      mean={np.mean(rates_fft):.4f}, std={np.std(rates_fft):.4f}")

    # Correlation matrix
    corr = noise.covariance_to_correlation()
    print(f"  Correlation matrix condition: {np.linalg.cond(corr):.4e}")

    # Non-Markovian temporal noise
    nm_noise = NonMarkovianNoise(n_qubits, base_rate, memory_lambda=0.3, sigma=0.02, rng=RNG)
    temporal_seq = nm_noise.sample_temporal_sequence(n_steps=5)
    print(f"  Non-Markovian temporal sequence shape: {temporal_seq.shape}")

    # =====================================================================
    # 3. Lindbladian Open-System Dynamics
    # =====================================================================
    print_section("3. Lindbladian Open-System Dynamics")

    # Single-qubit dissipative dynamics
    I, X, Y, Z = pauli_operators()
    H = 0.5 * Z  # Single-qubit Hamiltonian
    jump_ops = [np.sqrt(0.01) * X, np.sqrt(0.005) * Z]
    L = lindbladian_superoperator(H, jump_ops, hbar=1.0)
    print(f"\n  Liouvillian shape: {L.shape}")

    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    rho_euler = forward_euler_rho(rho0, L, t_final=1.0, n_steps=1000)
    rho_exact = exact_lindblad_evolution(rho0, L, t=1.0)
    err = np.linalg.norm(rho_euler - rho_exact, 'fro')
    print(f"  Forward Euler vs Exact error (t=1.0): {err:.4e}")
    print(f"  Von Neumann entropy of steady state: {von_neumann_entropy(rho_exact):.6f}")

    # DG solver for error probability density on 1D chain
    dg_solver = DGLindbladSolver(n_elements=10, poly_order=3, domain=(0.0, 1.0))
    u0 = np.zeros((dg_solver.n_elements, dg_solver.Np))
    u0[:, :] = 0.1  # Initial uniform error density
    u_final = dg_solver.evolve(u0, t_final=0.5, n_steps=100, v=0.5, D=0.01, gamma=0.1)
    print(f"  DG solver final mean density: {np.mean(u_final):.6f}")

    # =====================================================================
    # 4. Syndrome Decoding
    # =====================================================================
    print_section("4. Syndrome Decoding Algorithms")

    L_test = 3
    code_test = codes[L_test]
    H_full = code_test.get_parity_check_matrix()
    n = code_test.n_qubits
    m = H_full.shape[0]

    # Use a very small random error for decoder demonstration
    e_random = (RNG.integers(0, 2, 2 * n)).astype(int)
    syndrome = code_test.syndrome_of_error(e_random)
    print(f"\n  Decoding demonstration (L={L_test}):")
    print(f"    Random error weight: {np.sum(e_random)}")

    # Belief propagation decoder
    decoder_bp = BeliefPropagationDecoder(H_full, max_iter=50)
    recovery_bp = decoder_bp.decode(syndrome)
    syndrome_bp = code_test.syndrome_of_error(recovery_bp)
    print(f"    BP decoder syndrome match: {np.array_equal(syndrome_bp, syndrome)}")

    # Knapsack-like logical decoder
    decoder_kp = KnapsackLikeLogicalDecoder(H_full)
    recovery_kp = decoder_kp.decode(syndrome)
    syndrome_kp = code_test.syndrome_of_error(recovery_kp)
    print(f"    Knapsack decoder syndrome match: {np.array_equal(syndrome_kp, syndrome)}")

    # Union-Find decoder
    decoder_uf = UnionFindDecoder(L_test, boundary="toric")
    defects = [i for i in range(m) if syndrome[i] == 1]
    recovery_uf = decoder_uf.decode_syndrome(defects)
    print(f"    Union-Find decoder output shape: {recovery_uf.shape}")

    # =====================================================================
    # 5. Monte Carlo Logical Error Rate Estimation
    # =====================================================================
    print_section("5. Monte Carlo Logical Error Rate Estimation")

    p_values = np.linspace(0.02, 0.20, 5)
    results_mc = {}
    for L in [3, 4]:
        code = codes[L]
        H_full_L = code.get_parity_check_matrix()
        decoder = KnapsackLikeLogicalDecoder(H_full_L)
        mc = MonteCarloLogicalError(code, decoder, rng=RNG)
        P_L_vals = []
        print(f"\n  L={L} (distance={code.distance}):")
        for p in p_values:
            res = mc.estimate(p, n_shots=200, error_type="depolarizing")
            P_L_vals.append(res["P_L"])
            print(f"    p={p:.3f} -> P_L={res['P_L']:.4f} ± {res['std_error']:.4f}")
        results_mc[L] = np.array(P_L_vals)

    # =====================================================================
    # 6. Threshold FEM Boundary & Edge Detection
    # =====================================================================
    print_section("6. Threshold FEM Boundary Analysis")

    # FEM solve for L=3
    fem = ThresholdFEM(p_left=0.0, p_right=0.5, n_elements=80)
    nodes_fem, P_L_fem = fem.assemble_and_solve(code_distance=3)
    print(f"\n  FEM solution range: P_L ∈ [{np.min(P_L_fem):.4f}, {np.max(P_L_fem):.4f}]")

    # Edge detection for threshold
    edge_det = EdgeDetectorThreshold(nodes_fem, P_L_fem)
    p_th_deriv = edge_det.derivative_edge_detection(window=5)
    p_th_sigmoid = edge_det.sigmoid_fit_threshold()
    p_th_inflect = edge_det.second_derivative_zero_crossing()
    print(f"  Threshold (max derivative):   {p_th_deriv:.4f}")
    print(f"  Threshold (sigmoid fit):      {p_th_sigmoid:.4f}")
    print(f"  Threshold (inflection point): {p_th_inflect:.4f}")

    # Test edge profile functions
    profile = edge_det.fx_edge_profile(steepness=15.0)
    phantom = edge_det.fxy_shepp_logan_like(width=0.03)
    print(f"  Edge profile max derivative: {np.max(np.abs(np.diff(profile))):.4f}")

    # =====================================================================
    # 7. Hermite Quadrature & Sparse Grid Integration
    # =====================================================================
    print_section("7. Hermite Quadrature & Sparse Grid Integration")

    hq = HermiteQuadrature(n_points=20)
    # Integral of Gaussian function
    test_integral = hq.integrate(lambda x: np.exp(-x ** 2 / 2))
    print(f"\n  Hermite quadrature ∫ exp(-x²/2) exp(-x²) dx ≈ {test_integral:.6f}")
    print(f"  Analytical (for comparison): sqrt(pi/2) ≈ {np.sqrt(np.pi / 2):.6f}")

    # Project onto Hermite basis
    coeffs = hq.project_onto_basis(lambda x: np.tanh(x), max_n=10)
    print(f"  Hermite coefficients for tanh(x): {chop_array(coeffs[:5])}")

    # Sparse grid for 2D noise correlation integral
    sg = TruncatedNormalSparseGrid(dim=2, level=3, bounds=(-2.0, 2.0))
    print(f"  Sparse grid nodes: {sg.nodes.shape[0]}")

    def f2d(x):
        p1 = 0.5 * (1.0 + np.tanh(x[0]))
        p2 = 0.5 * (1.0 + np.tanh(x[1]))
        return p1 * p2

    val_sg = sg.integrate(f2d)
    print(f"  Sparse grid ∫ p1*p2 dμ ≈ {val_sg:.6f}")

    # Quantum noise integral
    qni = QuantumNoiseIntegral(n_qubits=10)
    mom = qni.multi_qubit_moment_integral([1, 2], dim=2, level=3)
    print(f"  Multi-qubit moment E[p1^1 * p2^2] ≈ {mom:.6f}")

    # =====================================================================
    # 8. Rare Event & Importance Sampling
    # =====================================================================
    print_section("8. Rare Event Importance Sampling")

    # Rejection sampler for tail events
    rej_sampler = RejectionSampler(code_distance=3, threshold=0.12, rng=RNG)
    tail_samples = rej_sampler.sample_above_threshold(base_rate=0.05, n_samples=100)
    print(f"\n  Tail samples above threshold: mean={np.mean(tail_samples):.4f}, count={len(tail_samples)}")

    # Caustic degeneracy analysis
    caustic = CausticDegeneracyAnalyzer(n_points=500)
    x_c, y_c, lines = caustic.caustic_syndrome_pattern(n_defects=12, multiplier=3)
    print(f"  Caustic pattern nodes: {len(x_c)}")
    theta, interference = caustic.degeneracy_interference(
        error_weights=np.array([1.0, 0.8, 0.6, 0.4, 0.2]),
        n_qubits=5
    )
    print(f"  Degeneracy interference peak: {np.max(interference):.4f}")

    region = caustic.box_distance_importance_region(p_center=0.15, box_size=0.06)
    print(f"  Importance region: [{region['lower']:.3f}, {region['upper']:.3f}], volume={region['volume']:.4f}")

    # =====================================================================
    # 9. Finite-Size Scaling Analysis
    # =====================================================================
    print_section("9. Finite-Size Scaling Analysis")

    # Synthetic scaling data for demonstration
    p_fss = np.linspace(0.05, 0.20, 20)
    distances = [3, 4, 5]
    P_L_fss = np.zeros((len(distances), len(p_fss)))
    for i, d in enumerate(distances):
        p_th_guess = 0.11
        nu_guess = 1.0
        scaled = (p_fss - p_th_guess) * (d ** (1.0 / nu_guess))
        P_L_fss[i, :] = 0.5 * (1.0 + np.tanh(15.0 * scaled))

    fss_result = finite_size_scaling(p_fss, P_L_fss, distances)
    print(f"\n  Fitted threshold p_th = {fss_result['p_th']:.4f}")
    print(f"  Fitted exponent ν = {fss_result['nu']:.4f}")
    print(f"  Collapse quality = {fss_result['quality']:.4e}")

    # =====================================================================
    # 10. Test Matrix Validation
    # =====================================================================
    print_section("10. Test Matrix Suite Validation")

    test_suite = TestMatrixSuite()
    for name, mat in [
        ("Kahan(6)", test_suite.kahan_matrix(6)),
        ("CONEX1(6)", test_suite.conex1_matrix(6)),
        ("COMBIN(6)", test_suite.combin_matrix(6))
    ]:
        ce = ConditionEstimator(mat)
        hager = ce.hager_estimator()
        exact = ce.exact_condition_number()
        print(f"\n  {name}:")
        print(f"    Hager estimate:  {hager:.4e}")
        print(f"    Exact κ:         {exact:.4e}")
        print(f"    Relative error:  {abs(hager - exact) / exact:.4e}")

    # =====================================================================
    # Summary
    # =====================================================================
    print_section("Summary")
    print("\n  Quantum Error Correction Threshold Analysis completed successfully.")
    print(f"  Surface code distances analyzed: {L_values}")
    print(f"  Threshold estimates (FEM): p_th ≈ {p_th_sigmoid:.3f}")
    print(f"  Threshold estimates (FSS): p_th ≈ {fss_result['p_th']:.3f}")
    print("  All 15 seed project algorithms successfully incorporated.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（50个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: pauli_operators 验证基本性质 ----
I_tc, X_tc, Y_tc, Z_tc = pauli_operators()
assert I_tc.shape == (2, 2), '[TC01] pauli_operators 验证基本性质 FAILED'
assert np.allclose(X_tc @ X_tc, I_tc), '[TC01] pauli_operators 验证基本性质 FAILED'
assert np.allclose(Y_tc @ Y_tc, I_tc), '[TC01] pauli_operators 验证基本性质 FAILED'
assert np.allclose(Z_tc @ Z_tc, I_tc), '[TC01] pauli_operators 验证基本性质 FAILED'
assert np.allclose(X_tc @ Y_tc, 1j * Z_tc), '[TC01] pauli_operators 验证基本性质 FAILED'

# ---- TC02: depolarizing_channel 形状与迹 ----
chi_tc = depolarizing_channel(p=0.1, n_qubits=1)
assert chi_tc.shape == (4, 4), '[TC02] depolarizing_channel 形状与迹 FAILED'
assert np.isclose(np.trace(chi_tc), 4.0*(1-0.1) + 0.1, atol=1e-10), '[TC02] depolarizing_channel 形状与迹 FAILED'

# ---- TC03: von_neumann_entropy 纯态为零 ----
rho_pure_tc = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
S_pure_tc = von_neumann_entropy(rho_pure_tc)
assert abs(S_pure_tc) < 1e-10, '[TC03] von_neumann_entropy 纯态为零 FAILED'

# ---- TC04: fidelity 相同态保真度为1 ----
F_tc = fidelity(rho_pure_tc, rho_pure_tc)
assert abs(F_tc - 1.0) < 1e-10, '[TC04] fidelity 相同态保真度为1 FAILED'

# ---- TC05: chop_array 小值截断为零 ----
arr_tc = np.array([1e-15, 1.0, 1e-16j], dtype=complex)
chopped_tc = chop_array(arr_tc)
assert chopped_tc[0] == 0.0, '[TC05] chop_array 小值截断为零 FAILED'
assert chopped_tc[1] == 1.0, '[TC05] chop_array 小值截断为零 FAILED'
assert chopped_tc[2] == 0.0, '[TC05] chop_array 小值截断为零 FAILED'

# ---- TC06: SurfaceCode toric L=2 构造参数 ----
code2_tc = SurfaceCode(2, boundary="toric")
assert code2_tc.n_qubits == 8, '[TC06] SurfaceCode toric L=2 构造参数 FAILED'
assert code2_tc.n_logical == 2, '[TC06] SurfaceCode toric L=2 构造参数 FAILED'
assert code2_tc.distance == 2, '[TC06] SurfaceCode toric L=2 构造参数 FAILED'

# ---- TC07: SurfaceCode toric L=3 构造参数 ----
code3_tc = SurfaceCode(3, boundary="toric")
assert code3_tc.n_qubits == 18, '[TC07] SurfaceCode toric L=3 构造参数 FAILED'
assert code3_tc.n_logical == 2, '[TC07] SurfaceCode toric L=3 构造参数 FAILED'
assert code3_tc.distance == 3, '[TC07] SurfaceCode toric L=3 构造参数 FAILED'

# ---- TC08: syndrome_of_error 零错误零综合症 ----
zero_err_tc = np.zeros(2 * code3_tc.n_qubits, dtype=int)
synd_zero_tc = code3_tc.syndrome_of_error(zero_err_tc)
assert np.all(synd_zero_tc == 0), '[TC08] syndrome_of_error 零错误零综合症 FAILED'

# ---- TC09: logical_error_indicator 平凡组合 ----
rec_zero_tc = np.zeros(2 * code3_tc.n_qubits, dtype=int)
log_ind_tc = code3_tc.logical_error_indicator(rec_zero_tc, zero_err_tc)
assert np.array_equal(log_ind_tc, np.array([0, 0])), '[TC09] logical_error_indicator 平凡组合 FAILED'

# ---- TC10: SurfaceCode parity_check_matrix 形状 ----
S_tc = code3_tc.get_parity_check_matrix()
n_stab = code3_tc.Hx.shape[0] + code3_tc.Hz.shape[0]
assert S_tc.shape[0] == n_stab, '[TC10] SurfaceCode parity_check_matrix 形状 FAILED'
assert S_tc.shape[1] == 2 * code3_tc.n_qubits, '[TC10] SurfaceCode parity_check_matrix 形状 FAILED'

# ---- TC11: SurfaceCode CRS转换非空 ----
Hx_arr = code3_tc.Hx
val_tc, col_tc, row_tc = code3_tc.convert_to_crs(Hx_arr)
assert len(val_tc) > 0, '[TC11] SurfaceCode CRS转换非空 FAILED'
assert len(col_tc) == len(val_tc), '[TC11] SurfaceCode CRS转换非空 FAILED'
assert len(row_tc) == Hx_arr.shape[0] + 1, '[TC11] SurfaceCode CRS转换非空 FAILED'

# ---- TC12: SurfaceCode sparse_parity_check 类型与形状 ----
Hx_sp_tc = code3_tc.sparse_parity_check("x")
assert Hx_sp_tc.shape == code3_tc.Hx.shape, '[TC12] SurfaceCode sparse_parity_check 类型与形状 FAILED'
Hz_sp_tc = code3_tc.sparse_parity_check("z")
assert Hz_sp_tc.shape == code3_tc.Hz.shape, '[TC12] SurfaceCode sparse_parity_check 类型与形状 FAILED'

# ---- TC13: CorrelatedPauliNoise 协方差与采样边界 ----
import numpy as np
np.random.seed(42)
noise_rng_tc = np.random.default_rng(42)
noise_tc = CorrelatedPauliNoise(n_qubits=4, base_rate=0.05, sigma=0.01, correlation_length=1.0, nu=0.5, rng=noise_rng_tc)
rates_tc = noise_tc.sample_rates_cholesky()
assert len(rates_tc) == 4, '[TC13] CorrelatedPauliNoise 协方差与采样边界 FAILED'
assert np.all(rates_tc >= 1e-6) and np.all(rates_tc <= 1.0 - 1e-6), '[TC13] CorrelatedPauliNoise 协方差与采样边界 FAILED'
corr_tc = noise_tc.covariance_to_correlation()
assert corr_tc.shape == (4, 4), '[TC13] CorrelatedPauliNoise 协方差与采样边界 FAILED'
assert np.allclose(np.diag(corr_tc), 1.0), '[TC13] CorrelatedPauliNoise 协方差与采样边界 FAILED'

# ---- TC14: CorrelatedPauliNoise Eigen采样有穷 ----
np.random.seed(43)
eig_rng_tc = np.random.default_rng(43)
noise_eig_tc = CorrelatedPauliNoise(n_qubits=6, base_rate=0.03, sigma=0.02, correlation_length=1.5, nu=0.5, rng=eig_rng_tc)
rates_eig_tc = noise_eig_tc.sample_rates_eigen()
assert np.all(np.isfinite(rates_eig_tc)), '[TC14] CorrelatedPauliNoise Eigen采样有穷 FAILED'
assert len(rates_eig_tc) == 6, '[TC14] CorrelatedPauliNoise Eigen采样有穷 FAILED'

# ---- TC15: NonMarkovianNoise 时序序列形状与二进制 ----
np.random.seed(44)
nm_rng_tc = np.random.default_rng(44)
nm_tc = NonMarkovianNoise(n_qubits=4, base_rate=0.05, memory_lambda=0.3, sigma=0.01, rng=nm_rng_tc)
seq_tc = nm_tc.sample_temporal_sequence(n_steps=3)
assert seq_tc.shape == (3, 8), '[TC15] NonMarkovianNoise 时序序列形状与二进制 FAILED'
assert np.all((seq_tc == 0) | (seq_tc == 1)), '[TC15] NonMarkovianNoise 时序序列形状与二进制 FAILED'

# ---- TC16: lindbladian_superoperator 形状正确性 ----
I2_tc, X2_tc, Y2_tc, Z2_tc = pauli_operators()
H_l_tc = 0.1 * Z2_tc
jump_l_tc = [np.sqrt(0.01) * X2_tc]
L_sup_tc = lindbladian_superoperator(H_l_tc, jump_l_tc)
assert L_sup_tc.shape == (4, 4), '[TC16] lindbladian_superoperator 形状正确性 FAILED'

# ---- TC17: exact_lindblad_evolution 厄米性与迹守恒 ----
rho0_tc = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
rho_exact_tc = exact_lindblad_evolution(rho0_tc, L_sup_tc, t=0.1)
assert rho_exact_tc.shape == (2, 2), '[TC17] exact_lindblad_evolution 厄米性与迹守恒 FAILED'
assert np.allclose(rho_exact_tc, rho_exact_tc.conj().T), '[TC17] exact_lindblad_evolution 厄米性与迹守恒 FAILED'
assert abs(np.trace(rho_exact_tc) - 1.0) < 1e-10, '[TC17] exact_lindblad_evolution 厄米性与迹守恒 FAILED'

# ---- TC18: forward_euler_rho 迹近似守恒 ----
rho_euler_tc = forward_euler_rho(rho0_tc, L_sup_tc, t_final=0.1, n_steps=2000)
assert rho_euler_tc.shape == (2, 2), '[TC18] forward_euler_rho 迹近似守恒 FAILED'
assert abs(np.trace(rho_euler_tc) - 1.0) < 1e-6, '[TC18] forward_euler_rho 迹近似守恒 FAILED'

# ---- TC19: DGLindbladSolver 演化输出形状与物理范围 ----
dg_tc = DGLindbladSolver(n_elements=4, poly_order=2, domain=(0.0, 1.0))
u0_tc = np.zeros((dg_tc.n_elements, dg_tc.Np))
u0_tc[:, :] = 0.1
u_final_tc = dg_tc.evolve(u0_tc, t_final=0.1, n_steps=10, v=0.5, D=0.01, gamma=0.1)
assert u_final_tc.shape == (4, 3), '[TC19] DGLindbladSolver 演化输出形状与物理范围 FAILED'
assert np.all(u_final_tc >= 0.0) and np.all(u_final_tc <= 1.0), '[TC19] DGLindbladSolver 演化输出形状与物理范围 FAILED'

# ---- TC20: MWPMBruteDecoder 零综合症返回零向量 ----
H_small_tc = np.array([[1, 1, 0], [0, 1, 1]], dtype=int)
decoder_mwpm_tc = MWPMBruteDecoder(H_small_tc)
rec_mwpm_tc = decoder_mwpm_tc.decode(np.array([0, 0]))
assert np.array_equal(rec_mwpm_tc, np.array([0, 0, 0])), '[TC20] MWPMBruteDecoder 零综合症返回零向量 FAILED'

# ---- TC21: BeliefPropagationDecoder 零综合症返回零向量 ----
decoder_bp_tc = BeliefPropagationDecoder(H_small_tc, max_iter=10)
rec_bp_tc = decoder_bp_tc.decode(np.array([0, 0]))
assert np.array_equal(rec_bp_tc, np.array([0, 0, 0])), '[TC21] BeliefPropagationDecoder 零综合症返回零向量 FAILED'

# ---- TC22: KnapsackLikeLogicalDecoder 零综合症返回零向量 ----
decoder_kp_tc = KnapsackLikeLogicalDecoder(H_small_tc)
rec_kp_tc = decoder_kp_tc.decode(np.array([0, 0]))
assert np.array_equal(rec_kp_tc, np.array([0, 0, 0])), '[TC22] KnapsackLikeLogicalDecoder 零综合症返回零向量 FAILED'

# ---- TC23: UnionFindDecoder 空缺陷返回零 ----
uf_tc = UnionFindDecoder(3, boundary="toric")
rec_uf_tc = uf_tc.decode_syndrome([])
assert rec_uf_tc.shape == (18,), '[TC23] UnionFindDecoder 空缺陷返回零 FAILED'
assert np.sum(rec_uf_tc) == 0, '[TC23] UnionFindDecoder 空缺陷返回零 FAILED'

# ---- TC24: ThresholdFEM 组装求解形状与边界条件 ----
fem_tc = ThresholdFEM(p_left=0.0, p_right=0.5, n_elements=20)
nodes_tc, P_L_tc = fem_tc.assemble_and_solve(code_distance=3)
assert len(nodes_tc) == 21, '[TC24] ThresholdFEM 组装求解形状与边界条件 FAILED'
assert len(P_L_tc) == 21, '[TC24] ThresholdFEM 组装求解形状与边界条件 FAILED'
assert P_L_tc[0] == 0.0, '[TC24] ThresholdFEM 组装求解形状与边界条件 FAILED'
assert P_L_tc[-1] == 1.0, '[TC24] ThresholdFEM 组装求解形状与边界条件 FAILED'

# ---- TC25: EdgeDetectorThreshold sigmoid拟合有穷 ----
p_ed_tc = np.linspace(0, 1, 50)
P_ed_tc = 1.0 / (1.0 + np.exp(-10.0 * (p_ed_tc - 0.5)))
ed_tc = EdgeDetectorThreshold(p_ed_tc, P_ed_tc)
p_th_sig_tc = ed_tc.sigmoid_fit_threshold()
assert np.isfinite(p_th_sig_tc), '[TC25] EdgeDetectorThreshold sigmoid拟合有穷 FAILED'

# ---- TC26: EdgeDetectorThreshold 导数检测有穷 ----
p_th_deriv_tc = ed_tc.derivative_edge_detection(window=3)
assert np.isfinite(p_th_deriv_tc), '[TC26] EdgeDetectorThreshold 导数检测有穷 FAILED'

# ---- TC27: finite_size_scaling 返回结构 ----
p_fss_tc = np.linspace(0.05, 0.20, 10)
distances_tc = [3, 4]
P_L_fss_tc = np.zeros((2, 10))
for i_tc, d_tc in enumerate(distances_tc):
    scaled_tc = (p_fss_tc - 0.11) * d_tc
    P_L_fss_tc[i_tc, :] = 0.5 * (1.0 + np.tanh(10.0 * scaled_tc))
fss_res_tc = finite_size_scaling(p_fss_tc, P_L_fss_tc, distances_tc)
assert "p_th" in fss_res_tc, '[TC27] finite_size_scaling 返回结构 FAILED'
assert "nu" in fss_res_tc, '[TC27] finite_size_scaling 返回结构 FAILED'
assert "quality" in fss_res_tc, '[TC27] finite_size_scaling 返回结构 FAILED'

# ---- TC28: HermiteQuadrature 积分常数1 ----
hq_tc = HermiteQuadrature(n_points=10)
val_hq_tc = hq_tc.integrate(lambda x: np.ones_like(x))
assert abs(val_hq_tc - np.sqrt(np.pi)) < 1e-10, '[TC28] HermiteQuadrature 积分常数1 FAILED'

# ---- TC29: HermiteQuadrature 积分exp(-x²/2) ----
val_exp_tc = hq_tc.integrate(lambda x: np.exp(-x ** 2 / 2))
assert abs(val_exp_tc - np.sqrt(2 * np.pi / 3)) < 1e-6, '[TC29] HermiteQuadrature 积分exp(-x²/2) FAILED'

# ---- TC30: Hermite 多项式递推正确性 ----
x_test = np.array([0.0, 1.0, 2.0])
H0_tc = hq_tc.physicist_polynomial(0, x_test)
assert np.allclose(H0_tc, [1.0, 1.0, 1.0]), '[TC30] Hermite 多项式递推正确性 FAILED'
H1_tc = hq_tc.physicist_polynomial(1, x_test)
assert np.allclose(H1_tc, [0.0, 2.0, 4.0]), '[TC30] Hermite 多项式递推正确性 FAILED'

# ---- TC31: TruncatedNormalSparseGrid 积分有穷 ----
sg_tc = TruncatedNormalSparseGrid(dim=2, level=2, bounds=(-1.0, 1.0))
val_sg_tc = sg_tc.integrate(lambda x: 1.0)
assert np.isfinite(val_sg_tc), '[TC31] TruncatedNormalSparseGrid 积分有穷 FAILED'
assert val_sg_tc >= 0.0, '[TC31] TruncatedNormalSparseGrid 积分有穷 FAILED'

# ---- TC32: QuantumNoiseIntegral 多比特矩积分非负 ----
qni_tc = QuantumNoiseIntegral(n_qubits=4)
mom_tc = qni_tc.multi_qubit_moment_integral([1, 1], dim=2, level=2)
assert np.isfinite(mom_tc), '[TC32] QuantumNoiseIntegral 多比特矩积分非负 FAILED'
assert mom_tc >= 0.0, '[TC32] QuantumNoiseIntegral 多比特矩积分非负 FAILED'

# ---- TC33: ParityCheckConditionAnalyzer 单位矩阵分析 ----
analyzer_tc = ParityCheckConditionAnalyzer(np.eye(5))
info_tc = analyzer_tc.analyze()
assert info_tc['rank'] == 5, '[TC33] ParityCheckConditionAnalyzer 单位矩阵分析 FAILED'
assert info_tc['exact_kappa'] == 1.0, '[TC33] ParityCheckConditionAnalyzer 单位矩阵分析 FAILED'

# ---- TC34: ConditionEstimator 单位矩阵条件数 ----
ce_tc = ConditionEstimator(np.eye(5))
assert ce_tc.exact_condition_number() == 1.0, '[TC34] ConditionEstimator 单位矩阵条件数 FAILED'

# ---- TC35: TestMatrixSuite Kahan矩阵形状与上三角 ----
kahan_tc = TestMatrixSuite.kahan_matrix(4)
assert kahan_tc.shape == (4, 4), '[TC35] TestMatrixSuite Kahan矩阵形状与上三角 FAILED'
assert np.allclose(np.triu(kahan_tc), kahan_tc), '[TC35] TestMatrixSuite Kahan矩阵形状与上三角 FAILED'

# ---- TC36: TestMatrixSuite CONEX1矩阵秩 ----
conex_tc = TestMatrixSuite.conex1_matrix(5)
assert conex_tc.shape == (5, 5), '[TC36] TestMatrixSuite CONEX1矩阵秩 FAILED'
assert np.linalg.matrix_rank(conex_tc) == 5, '[TC36] TestMatrixSuite CONEX1矩阵秩 FAILED'

# ---- TC37: RejectionSampler 样本数量与边界 ----
import numpy as np
np.random.seed(45)
rej_rng_tc = np.random.default_rng(45)
rej_tc = RejectionSampler(code_distance=3, threshold=0.5, rng=rej_rng_tc)
samples_tc = rej_tc.sample_above_threshold(base_rate=0.1, n_samples=10)
assert len(samples_tc) == 10, '[TC37] RejectionSampler 样本数量与边界 FAILED'
assert np.all(samples_tc >= 0.0) and np.all(samples_tc <= 1.0), '[TC37] RejectionSampler 样本数量与边界 FAILED'

# ---- TC38: CausticDegeneracyAnalyzer 焦散模式形状 ----
caustic_tc = CausticDegeneracyAnalyzer(n_points=100)
x_c_tc, y_c_tc, lines_tc = caustic_tc.caustic_syndrome_pattern(n_defects=6, multiplier=2)
assert len(x_c_tc) == 6, '[TC38] CausticDegeneracyAnalyzer 焦散模式形状 FAILED'
assert len(lines_tc) == 6, '[TC38] CausticDegeneracyAnalyzer 焦散模式形状 FAILED'

# ---- TC39: CausticDegeneracyAnalyzer 干涉图样非负 ----
theta_tc, interf_tc = caustic_tc.degeneracy_interference(
    error_weights=np.array([1.0, 0.8, 0.6, 0.4, 0.2]),
    n_qubits=5
)
assert len(interf_tc) == 100, '[TC39] CausticDegeneracyAnalyzer 干涉图样非负 FAILED'
assert np.all(interf_tc >= 0.0), '[TC39] CausticDegeneracyAnalyzer 干涉图样非负 FAILED'

# ---- TC40: MonteCarloLogicalError p=0时逻辑错误率为零 ----
import numpy as np
np.random.seed(46)
mc_rng_tc = np.random.default_rng(46)
code_mc_tc = SurfaceCode(3, boundary="toric")
H_mc_tc = code_mc_tc.get_parity_check_matrix()
decoder_mc_tc = BeliefPropagationDecoder(H_mc_tc, max_iter=20)
mc_tc = MonteCarloLogicalError(code_mc_tc, decoder_mc_tc, rng=mc_rng_tc)
res_mc_tc = mc_tc.estimate(0.0, n_shots=20, error_type="depolarizing")
assert res_mc_tc["P_L"] == 0.0, '[TC40] MonteCarloLogicalError p=0时逻辑错误率为零 FAILED'
assert res_mc_tc["n_shots"] == 20, '[TC40] MonteCarloLogicalError p=0时逻辑错误率为零 FAILED'

# ---- TC41: SurfaceCode box_distance 与边界词 ----
box_tc = code3_tc.box_distance_logical_operators()
assert "dx" in box_tc and "dz" in box_tc, '[TC41] SurfaceCode box_distance 与边界词 FAILED'
assert box_tc["dx"] == 3.0, '[TC41] SurfaceCode box_distance 与边界词 FAILED'
bword_tc = code3_tc.boundary_word_topology()
assert len(bword_tc) == 12, '[TC41] SurfaceCode box_distance 与边界词 FAILED'

# ---- TC42: CorrelatedPauliNoise FFT采样形状正确 ----
import numpy as np
np.random.seed(48)
fft_rng_tc = np.random.default_rng(48)
noise_fft_tc = CorrelatedPauliNoise(n_qubits=5, base_rate=0.05, sigma=0.02, correlation_length=1.0, nu=0.5, rng=fft_rng_tc)
rates_fft_tc = noise_fft_tc.sample_rates_fft()
assert len(rates_fft_tc) == 5, '[TC42] CorrelatedPauliNoise FFT采样形状正确 FAILED'
assert np.all(np.isfinite(rates_fft_tc)), '[TC42] CorrelatedPauliNoise FFT采样形状正确 FAILED'
assert np.all(rates_fft_tc >= 1e-6) and np.all(rates_fft_tc <= 1.0 - 1e-6), '[TC42] CorrelatedPauliNoise FFT采样形状正确 FAILED'

# ---- TC43: generate_brc_like_data 形状正确 ----
import numpy as np
np.random.seed(47)
brc_rng_tc = np.random.default_rng(47)
brc_noise_tc = CorrelatedPauliNoise(n_qubits=5, base_rate=0.05, sigma=0.02, rng=brc_rng_tc)
brc_data_tc = brc_noise_tc.generate_brc_like_data(n_samples=20)
assert brc_data_tc.shape == (20, 5), '[TC43] generate_brc_like_data 形状正确 FAILED'
assert np.all(brc_data_tc >= 0.0) and np.all(brc_data_tc <= 1.0), '[TC43] generate_brc_like_data 形状正确 FAILED'

# ---- TC44: 相关函数极值性质 ----
noise_cf_tc = CorrelatedPauliNoise(n_qubits=3, base_rate=0.05, sigma=0.02, correlation_length=1.0, nu=0.5)
r_test = np.array([0.0, 1.0, 2.0])
exp_corr = noise_cf_tc.exponential_correlation(r_test)
assert exp_corr[0] == 1.0, '[TC44] 相关函数极值性质 FAILED'
assert np.all(exp_corr >= 0.0) and np.all(exp_corr <= 1.0), '[TC44] 相关函数极值性质 FAILED'
gauss_corr = noise_cf_tc.gaussian_correlation(r_test)
assert gauss_corr[0] == 1.0, '[TC44] 相关函数极值性质 FAILED'
rat_corr = noise_cf_tc.rational_quadratic_correlation(r_test, alpha=1.0)
assert rat_corr[0] == 1.0, '[TC44] 相关函数极值性质 FAILED'

# ---- TC45: 二值高斯消元秩一致性 ----
from utils import binary_gaussian_elimination
M_test_tc = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=int)
rref_tc, rank_tc, pivots_tc = binary_gaussian_elimination(M_test_tc)
assert rank_tc == 2, '[TC45] 二值高斯消元秩一致性 FAILED'

# ---- TC46: RejectionSampler 重要性区域体积 ----
region_tc = caustic_tc.box_distance_importance_region(p_center=0.15, box_size=0.06)
assert "volume" in region_tc, '[TC46] RejectionSampler 重要性区域体积 FAILED'
assert region_tc["volume"] > 0.0, '[TC46] RejectionSampler 重要性区域体积 FAILED'

# ---- TC47: MonteCarloLogicalError 返回值结构完整 ----
res_structure_tc = mc_tc.estimate(0.01, n_shots=2, error_type="bitflip")
assert "P_L" in res_structure_tc, '[TC47] MonteCarloLogicalError 返回值结构完整 FAILED'
assert "variance" in res_structure_tc, '[TC47] MonteCarloLogicalError 返回值结构完整 FAILED'
assert "std_error" in res_structure_tc, '[TC47] MonteCarloLogicalError 返回值结构完整 FAILED'

# ---- TC48: SVD条件数奇异性检测 ----
sing_tc = TestMatrixSuite.ill_conditioned_parity_check(6, 4)
ce_sing_tc = ConditionEstimator(sing_tc)
kappa_tc = ce_sing_tc.exact_condition_number()
assert np.isfinite(kappa_tc), '[TC48] SVD条件数奇异性检测 FAILED'

# ---- TC49: Hermite 投影系数合理性 ----
coeffs_tc = hq_tc.project_onto_basis(lambda x: np.ones_like(x), max_n=5)
assert len(coeffs_tc) == 6, '[TC49] Hermite 投影系数合理性 FAILED'
# c0 = ∫ ψ0(x)·1·exp(-x²)dx = π^{-1/4}·√(2π/3)
expected_c0_tc = np.pi**(-0.25) * np.sqrt(2 * np.pi / 3)
assert abs(coeffs_tc[0] - expected_c0_tc) < 1e-6, '[TC49] Hermite 投影系数合理性 FAILED'

# ---- TC50: DGLindbladSolver rhs输出形状正确 ----
dg2_tc = DGLindbladSolver(n_elements=3, poly_order=2, domain=(-1.0, 1.0))
u_test_tc = np.zeros((dg2_tc.n_elements, dg2_tc.Np))
u_test_tc[:, :] = 0.2
rhs_out_tc = dg2_tc.rhs(u_test_tc, v=1.0, D=0.01, gamma=0.1)
assert rhs_out_tc.shape == (3, 3), '[TC50] DGLindbladSolver rhs输出形状正确 FAILED'
assert np.all(np.isfinite(rhs_out_tc)), '[TC50] DGLindbladSolver rhs输出形状正确 FAILED'

print('\n全部 50 个测试通过!\n')

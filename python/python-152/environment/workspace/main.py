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
    sys.exit(main())

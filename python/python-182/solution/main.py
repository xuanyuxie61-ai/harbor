"""
main.py

Unified zero-parameter entry point for the Bayesian Hierarchical Calibration
of a Spatially-Coupled Reaction-Diffusion System on an Annular Domain.

Running this script executes the complete pipeline:
  1. Synthetic data generation on an annular grid
  2. FEM basis spatial discretization and GMRF prior construction
  3. Periodic tridiagonal covariance computation via r83p
  4. Polynomial surrogate construction for FHN forward model
  5. Latin-hypercube-initialized adaptive MCMC with unicycle rotation proposals
  6. Bayesian quadrature for model evidence (line, square, triangle domains)
  7. Posterior summary output
"""
from inference_engine import run_bayesian_inference


def main():
    print("=" * 70)
    print("Bayesian Inference for Spatially-Coupled Reaction-Diffusion on Annulus")
    print("Domain: Bayesian Inference & MCMC Sampling")
    print("=" * 70)

    results = run_bayesian_inference()

    print("\n" + "=" * 70)
    print("FINAL POSTERIOR SUMMARIES")
    print("=" * 70)
    names = ['a', 'b', 'gamma', 'd0', 'c0', 'c1', 'c2', 'c3', 'log_sigma']
    for i, name in enumerate(names):
        print(f"  {name:12s}: mean = {results['posterior_mean'][i]:.5f},  "
              f"std = {results['posterior_std'][i]:.5f}")

    true = results['data']['true_params']
    print(f"\n  True a       = {true['a']}")
    print(f"  True b       = {true['b']}")
    print(f"  True gamma   = {true['gamma']}")
    print(f"  True d0      = {true['d0']}")
    print(f"  True sigma   = {true['sigma']}")

    print(f"\n  Acceptance rate        = {results['accept_rate']:.3f}")
    print(f"  Evidence slice (gamma) = {results['evidence_gamma']:.4e}")
    print(f"  Evidence slice (a,b)   = {results['evidence_ab']:.4e}")
    print(f"  Evidence slice (simplex)= {results['evidence_triangle']:.4e}")
    print("=" * 70)
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()

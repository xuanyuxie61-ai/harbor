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

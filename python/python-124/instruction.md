# 任务：复现缺失模块

## 目标
你面对的是一个 Python 科研代码项目。部分源码文件已缺失，导致程序无法完整运行。你需要根据保留的入口代码和项目描述，补全缺失模块，使项目恢复预期功能。

## 工作目录
代码仓库位于 `/app` 目录下。

## 项目描述

# Bone Remodeling Simulation Project Description

## Project Overview

This project implements a multi‑scale finite‑element simulation of bone remodeling under mechanical loading. It combines a 2D finite‑element solver (quadratic T6 triangles) for linear elasticity, a bone density field represented by Chebyshev expansions, microstructural models of trabecular bone, biochemical ODE models for bone density evolution, parameter identification via nonlinear least‑squares, and numerical diagnostics for accuracy assessment. The main entry point is `main.py`, which orchestrates these modules. All other `.py` files must be re‑implemented by the agent based on this description.

## File Inventory and Responsibilities

| File | Primary Responsibility |
|------|------------------------|
| `bone_geometry.py` | Generates a 2D T6 quadratic triangular mesh for a bone cross‑section. Provides node coordinates, element connectivity, element areas, triangle neighbour relations, and distance‑based classification of cortical vs. trabecular nodes. Also contains a standalone signed point‑to‑line distance function. |
| `microstructure_model.py` | Models trabecular bone as a 0/1 matrix (pentomino‑like shapes). Computes porosity, specific surface, and effective Young’s modulus of a representative volume element (RVE). Provides a function to build a density field over the whole mesh. |
| `density_field.py` | Represents a 2D bone density field via Chebyshev series. Evaluates density at reference and physical coordinates and maps density to elastic modulus via a power‑law. Includes helper functions for Chebyshev evaluation (Clenshaw recurrence), coefficient truncation, and L2 inner products. |
| `quadrature_engine.py` | High‑order quadrature rules: Gegenbauer‑Gauss (1D), triangle Gauss rules (2D, orders 1–7), unit tetrahedron monomial integrals, combination generators, and L2/H1 error estimators. |
| `fem_core.py` | 2D linear‑elasticity FEM solver using T6 quadratic triangles. Assembles stiffness matrix and load vector, applies Dirichlet/Neumann boundary conditions, solves the linear system, and computes strain energy density and nodal stresses. Also includes a banded lower‑triangular solver and shape‑function routines. |
| `bone_remodeling_ode.py` | Time‑dependent bone remodeling ODE models: a mechanostat‑based single‑species model and a coupled multi‑species model. Provides analytical solutions for a simplified linear ODE, ODE integration, and mass‑conservation checks. |
| `parameter_optimization.py` | Nonlinear least‑squares parameter identification using Levenberg‑Marquardt (wrapping `scipy.optimize.least_squares`). Also includes golden‑section search and gradient‑descent optimizers. Contains a simplified forward model for bone remodeling. |
| `numerical_diagnostics.py` | Tools for assessing numerical accuracy: matrix exponential (Padé and Taylor), stable vs. naive polynomial evaluation and quadratic root finding, matrix condition number analysis, and a report generator. |

## Module Boundaries and Interactions

- `main.py` imports and uses all other modules. It does not call any internal functions that are not exported by the module’s public API.
- `bone_geometry.py` produces the mesh and node coordinates. The result is used by `fem_core.py` (via `ElasticFEM2D`) and by

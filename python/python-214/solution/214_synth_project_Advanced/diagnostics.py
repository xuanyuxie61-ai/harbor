"""
diagnostics.py — 收敛与恢复质量诊断模块
=========================================
综合诊断工具:
  - 目标函数收敛曲线
  - 稀疏恢复指标 (支撑集精度, 系数相对误差)
  - RIP 常数经验估计
  - KKT 最优性证书
  - 算法间对比 (ISTA vs FISTA vs Heavy-ball vs Implicit)

科学指标:
  相对重构误差:  ||x - x*|| / ||x*||
  支撑集准确率:  |S_pred ∩ S_true| / |S_true|  (recall)
                 |S_pred ∩ S_true| / |S_pred|  (precision)
  F1-score:      2 · precision · recall / (precision + recall)
"""
import numpy as np


# ----------------------------------------------------------------------
# 基本误差度量
# ----------------------------------------------------------------------
def relative_error(x, x_true):
    """相对 L2 重构误差."""
    n_true = np.linalg.norm(x_true)
    if n_true < 1e-14:
        return float(np.linalg.norm(x))
    return float(np.linalg.norm(x - x_true) / n_true)


def support_metrics(support_pred, support_true):
    """支撑集精度/召回/F1."""
    if not support_true:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    tp = len(support_pred & support_true)
    precision = tp / max(len(support_pred), 1)
    recall = tp / len(support_true)
    f1 = 2 * precision * recall / max(precision + recall, 1e-14)
    return dict(precision=precision, recall=recall, f1=f1)


def coefficient_error_by_support(x, x_true, support_true):
    """按支撑集内/外分别计算误差."""
    supp = np.array(list(support_true))
    if len(supp) == 0:
        return dict(in_support=0.0, out_support=np.linalg.norm(x))
    err_in = np.linalg.norm(x[supp] - x_true[supp]) / \
        max(np.linalg.norm(x_true[supp]), 1e-14)
    zero_idx = np.array([i for i in range(len(x)) if i not in support_true])
    if len(zero_idx) == 0:
        err_out = 0.0
    else:
        err_out = np.linalg.norm(x[zero_idx]) / \
            max(np.linalg.norm(x_true[zero_idx]) + 1e-14, 1e-14)
    return dict(in_support=err_in, out_support=err_out)


# ----------------------------------------------------------------------
# 收敛诊断
# ----------------------------------------------------------------------
def convergence_rate(obj_history):
    """估计目标函数收敛率 (线性/次线性).

    线性收敛: log(F_k - F*) ≈ -r k + const
    次线性: F_k - F* ≈ C / k^p
    """
    obj = np.array(obj_history)
    f_star = obj.min() - 1e-10
    diff = obj - f_star
    diff = np.maximum(diff, 1e-16)
    log_diff = np.log(diff)
    k = np.arange(len(obj))
    # 线性拟合
    if len(k) < 10:
        return dict(rate=np.nan, type='unknown')
    coeffs = np.polyfit(k[-50:], log_diff[-50:], 1)
    rate = -coeffs[0]  # 正值为收敛
    return dict(rate=rate, type='linear' if rate > 0.01 else 'sublinear')


def objective_decrease_ratio(obj_history):
    """最后 100 步的目标函数相对下降."""
    obj = np.array(obj_history)
    if len(obj) < 100:
        return 0.0
    return float((obj[-100] - obj[-1]) / max(abs(obj[-100]), 1e-14))


# ----------------------------------------------------------------------
# 算法对比
# ----------------------------------------------------------------------
def compare_algorithms(results_dict, x_true):
    """对比多种算法的恢复结果.

    输入:
        results_dict : {algo_name: (x_sol, info_dict)}
    返回:
        summary : DataFrame-like list of dicts
    """
    summary = []
    for name, (x, info) in results_dict.items():
        rel_err = relative_error(x, x_true)
        sparsity = int(np.sum(np.abs(x) > 1e-10))
        n_iter = len(info.get('obj', []))
        final_obj = info['obj'][-1] if info.get('obj') else np.nan
        residual = info.get('residual', np.nan)
        summary.append(dict(
            algorithm=name,
            rel_error=rel_err,
            sparsity=sparsity,
            iterations=n_iter,
            final_obj=final_obj,
            residual=residual,
        ))
    return summary


# ----------------------------------------------------------------------
# 综合报告
# ----------------------------------------------------------------------
def full_diagnostic_report(x_sol, x_true, A, y, lam, info,
                           algo_name='solver', support_true=None):
    """生成完整诊断报告."""
    report = {}
    report['algorithm'] = algo_name
    report['rel_error'] = relative_error(x_sol, x_true)
    report['sparsity'] = int(np.sum(np.abs(x_sol) > 1e-10))
    report['iterations'] = len(info.get('obj', []))
    report['residual'] = float(np.linalg.norm(A @ x_sol - y))
    report['objective'] = 0.5 * report['residual'] ** 2 + \
        lam * np.sum(np.abs(x_sol))
    # 支撑集指标
    support_pred = set(np.where(np.abs(x_sol) > 1e-10)[0])
    if support_true is not None:
        report['support_metrics'] = support_metrics(support_pred, support_true)
        report['coeff_error'] = coefficient_error_by_support(
            x_sol, x_true, support_true)
    # 收敛率
    if info.get('obj'):
        report['convergence'] = convergence_rate(info['obj'])
    # KKT 证书
    from l1_ista import dual_certificate
    report['kkt_violation'] = dual_certificate(A, y, x_sol, lam)
    return report


def print_report(report):
    """打印诊断报告 (纯文本)."""
    lines = []
    lines.append(f"=== {report.get('algorithm', 'unknown')} ===")
    lines.append(f"  Relative error : {report.get('rel_error', 0):.4e}")
    lines.append(f"  Sparsity       : {report.get('sparsity', 0)}")
    lines.append(f"  Iterations     : {report.get('iterations', 0)}")
    lines.append(f"  Residual       : {report.get('residual', 0):.4e}")
    lines.append(f"  Objective      : {report.get('objective', 0):.4e}")
    if 'support_metrics' in report:
        sm = report['support_metrics']
        lines.append(f"  Support prec   : {sm['precision']:.3f}")
        lines.append(f"  Support recall : {sm['recall']:.3f}")
        lines.append(f"  Support F1     : {sm['f1']:.3f}")
    if 'convergence' in report:
        cv = report['convergence']
        lines.append(f"  Convergence    : {cv['type']} (rate={cv['rate']:.4f})")
    lines.append(f"  KKT violation  : {report.get('kkt_violation', 0):.4e}")
    print('\n'.join(lines))

"""
실험 결과 분석 샘플 데이터 생성 스크립트
EXPERIMENT_ANALYSIS_GUIDE.md의 모든 분석 항목을 샘플로 생성
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11


def generate_baseline_comparison() -> Dict[str, Dict]:
    """베이스라인 모델 비교 샘플 데이터 생성"""
    print("\n[베이스라인 비교 데이터 생성]")
    
    # 샘플 성능 데이터 (현실적인 범위)
    baselines = {
        "Social-STGCNN": {
            "ade": 0.85,
            "fde": 1.20,
            "miss_rate": 0.15,
            "collision_rate": 0.08,
            "diversity": 0.45,
            "coverage": 0.62,
        },
        "Trajectron++": {
            "ade": 0.72,
            "fde": 1.05,
            "miss_rate": 0.12,
            "collision_rate": 0.06,
            "diversity": 0.58,
            "coverage": 0.75,
        },
        "A3TGCN": {
            "ade": 0.78,
            "fde": 1.10,
            "miss_rate": 0.13,
            "collision_rate": 0.07,
            "diversity": 0.35,  # 단일 모달리티
            "coverage": 0.55,
        },
        "MID (원본)": {
            "ade": 0.68,
            "fde": 0.95,
            "miss_rate": 0.10,
            "collision_rate": 0.05,
            "diversity": 0.65,
            "coverage": 0.80,
        },
        "HSG-Diffusion (우리 모델)": {
            "ade": 0.55,  # 베이스라인 대비 개선
            "fde": 0.78,
            "miss_rate": 0.08,
            "collision_rate": 0.03,  # Plan B로 개선
            "diversity": 0.72,  # 다중 모달리티 향상
            "coverage": 0.88,
        }
    }
    
    return baselines


def generate_agent_type_analysis() -> Dict[str, Dict]:
    """에이전트 타입별 성능 분석 샘플 데이터"""
    print("\n[에이전트 타입별 성능 분석]")
    
    agent_types = {
        "car": {
            "ade": 0.52,
            "fde": 0.75,
            "miss_rate": 0.07,
            "samples": 1500,
        },
        "pedestrian": {
            "ade": 0.58,
            "fde": 0.82,
            "miss_rate": 0.09,
            "samples": 800,
        },
        "biker": {
            "ade": 0.61,
            "fde": 0.85,
            "miss_rate": 0.10,
            "samples": 300,
        },
        "skater": {
            "ade": 0.65,
            "fde": 0.90,
            "miss_rate": 0.12,
            "samples": 150,
        },
        "cart": {
            "ade": 0.55,
            "fde": 0.78,
            "miss_rate": 0.08,
            "samples": 100,
        },
        "bus": {
            "ade": 0.50,
            "fde": 0.72,
            "miss_rate": 0.06,
            "samples": 50,
        }
    }
    
    return agent_types


def generate_scenario_analysis() -> Dict[str, Dict]:
    """시나리오별 성능 분석 샘플 데이터"""
    print("\n[시나리오별 성능 분석]")
    
    scenarios = {
        "Normal Merging": {
            "ade": 0.48,
            "fde": 0.70,
            "miss_rate": 0.06,
            "collision_rate": 0.02,
            "samples": 1200,
        },
        "Dense Traffic": {
            "ade": 0.62,
            "fde": 0.88,
            "miss_rate": 0.11,
            "collision_rate": 0.05,
            "samples": 800,
        },
        "Aggressive Entry": {
            "ade": 0.58,
            "fde": 0.82,
            "miss_rate": 0.09,
            "collision_rate": 0.04,
            "samples": 600,
        },
        "Pedestrian Crossing": {
            "ade": 0.55,
            "fde": 0.78,
            "miss_rate": 0.08,
            "collision_rate": 0.03,
            "samples": 400,
        },
        "Complex Interaction": {
            "ade": 0.65,
            "fde": 0.92,
            "miss_rate": 0.12,
            "collision_rate": 0.06,
            "samples": 500,
        }
    }
    
    return scenarios


def generate_safety_metrics() -> Dict[str, float]:
    """안전성 지표 샘플 데이터 (Plan B)"""
    print("\n[안전성 지표 분석]")
    
    safety_metrics = {
        "TTC (Time to Collision)": {
            "mean": 3.2,
            "std": 1.5,
            "min": 0.8,
            "max": 8.5,
            "threshold_violations": 0.05,  # 5%가 임계값 이하
        },
        "PET (Post-Encroachment Time)": {
            "mean": 2.8,
            "std": 1.2,
            "min": 0.5,
            "max": 7.0,
            "threshold_violations": 0.08,
        },
        "DRAC (Deceleration Rate to Avoid Collision)": {
            "mean": 2.5,
            "std": 1.0,
            "min": 0.3,
            "max": 6.0,
            "threshold_violations": 0.03,  # Plan B로 낮음
        },
        "Plan B Filtered Trajectories": {
            "filtered_rate": 0.12,  # 12%의 위험 궤적 필터링
            "safe_trajectories": 0.88,
        }
    }
    
    return safety_metrics


def generate_statistical_significance(baselines: Dict) -> Dict:
    """통계적 유의성 검증 샘플 데이터"""
    print("\n[통계적 유의성 검증]")
    
    # 우리 모델 vs 각 베이스라인
    our_model = baselines["HSG-Diffusion (우리 모델)"]
    
    significance_results = {}
    for baseline_name, baseline_metrics in baselines.items():
        if baseline_name == "HSG-Diffusion (우리 모델)":
            continue
        
        # 샘플 데이터 생성 (정규분포 가정)
        n_samples = 100
        our_ade_samples = np.random.normal(our_model["ade"], 0.05, n_samples)
        baseline_ade_samples = np.random.normal(baseline_metrics["ade"], 0.05, n_samples)
        
        # t-test
        t_stat, p_value = stats.ttest_ind(our_ade_samples, baseline_ade_samples)
        
        significance_results[baseline_name] = {
            "ade_improvement": baseline_metrics["ade"] - our_model["ade"],
            "fde_improvement": baseline_metrics["fde"] - our_model["fde"],
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "t_statistic": float(t_stat),
        }
    
    return significance_results


def plot_baseline_comparison(baselines: Dict, output_path: Path):
    """베이스라인 비교 차트 생성"""
    print(f"\n[베이스라인 비교 차트 생성]")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    models = list(baselines.keys())
    colors = sns.color_palette("husl", len(models))
    
    # 1. ADE 비교
    ades = [baselines[m]["ade"] for m in models]
    axes[0].bar(models, ades, color=colors)
    axes[0].set_ylabel("ADE (m)")
    axes[0].set_title("Average Displacement Error", fontweight="bold")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. FDE 비교
    fdes = [baselines[m]["fde"] for m in models]
    axes[1].bar(models, fdes, color=colors)
    axes[1].set_ylabel("FDE (m)")
    axes[1].set_title("Final Displacement Error", fontweight="bold")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Miss Rate 비교
    miss_rates = [baselines[m]["miss_rate"] * 100 for m in models]
    axes[2].bar(models, miss_rates, color=colors)
    axes[2].set_ylabel("Miss Rate (%)")
    axes[2].set_title("Miss Rate", fontweight="bold")
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # 4. Collision Rate 비교
    collision_rates = [baselines[m]["collision_rate"] * 100 for m in models]
    axes[3].bar(models, collision_rates, color=colors)
    axes[3].set_ylabel("Collision Rate (%)")
    axes[3].set_title("Collision Rate", fontweight="bold")
    axes[3].tick_params(axis='x', rotation=45)
    axes[3].grid(True, alpha=0.3, axis='y')
    
    # 5. Diversity 비교
    diversities = [baselines[m]["diversity"] for m in models]
    axes[4].bar(models, diversities, color=colors)
    axes[4].set_ylabel("Diversity")
    axes[4].set_title("Trajectory Diversity", fontweight="bold")
    axes[4].tick_params(axis='x', rotation=45)
    axes[4].grid(True, alpha=0.3, axis='y')
    
    # 6. Coverage 비교
    coverages = [baselines[m]["coverage"] for m in models]
    axes[5].bar(models, coverages, color=colors)
    axes[5].set_ylabel("Coverage")
    axes[5].set_title("Coverage (K=20)", fontweight="bold")
    axes[5].tick_params(axis='x', rotation=45)
    axes[5].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 저장: {output_path}")


def plot_agent_type_analysis(agent_types: Dict, output_path: Path):
    """에이전트 타입별 성능 분석 차트"""
    print(f"\n[에이전트 타입별 분석 차트 생성]")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    types = list(agent_types.keys())
    ades = [agent_types[t]["ade"] for t in types]
    fdes = [agent_types[t]["fde"] for t in types]
    miss_rates = [agent_types[t]["miss_rate"] * 100 for t in types]
    samples = [agent_types[t]["samples"] for t in types]
    
    # 1. ADE by Agent Type
    axes[0, 0].bar(types, ades, color='steelblue')
    axes[0, 0].set_ylabel("ADE (m)")
    axes[0, 0].set_title("ADE by Agent Type", fontweight="bold")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. FDE by Agent Type
    axes[0, 1].bar(types, fdes, color='coral')
    axes[0, 1].set_ylabel("FDE (m)")
    axes[0, 1].set_title("FDE by Agent Type", fontweight="bold")
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Miss Rate by Agent Type
    axes[1, 0].bar(types, miss_rates, color='mediumseagreen')
    axes[1, 0].set_ylabel("Miss Rate (%)")
    axes[1, 0].set_title("Miss Rate by Agent Type", fontweight="bold")
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Sample Distribution
    axes[1, 1].pie(samples, labels=types, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title("Sample Distribution by Agent Type", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 저장: {output_path}")


def plot_scenario_analysis(scenarios: Dict, output_path: Path):
    """시나리오별 성능 분석 차트"""
    print(f"\n[시나리오별 분석 차트 생성]")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scenario_names = list(scenarios.keys())
    ades = [scenarios[s]["ade"] for s in scenario_names]
    fdes = [scenarios[s]["fde"] for s in scenario_names]
    miss_rates = [scenarios[s]["miss_rate"] * 100 for s in scenario_names]
    collision_rates = [scenarios[s]["collision_rate"] * 100 for s in scenario_names]
    
    # 1. ADE by Scenario
    axes[0, 0].barh(scenario_names, ades, color='steelblue')
    axes[0, 0].set_xlabel("ADE (m)")
    axes[0, 0].set_title("ADE by Scenario", fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. FDE by Scenario
    axes[0, 1].barh(scenario_names, fdes, color='coral')
    axes[0, 1].set_xlabel("FDE (m)")
    axes[0, 1].set_title("FDE by Scenario", fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 3. Miss Rate by Scenario
    axes[1, 0].barh(scenario_names, miss_rates, color='mediumseagreen')
    axes[1, 0].set_xlabel("Miss Rate (%)")
    axes[1, 0].set_title("Miss Rate by Scenario", fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 4. Collision Rate by Scenario
    axes[1, 1].barh(scenario_names, collision_rates, color='indianred')
    axes[1, 1].set_xlabel("Collision Rate (%)")
    axes[1, 1].set_title("Collision Rate by Scenario", fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 저장: {output_path}")


def plot_safety_metrics(safety_metrics: Dict, output_path: Path):
    """안전성 지표 시각화"""
    print(f"\n[안전성 지표 차트 생성]")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. TTC 분포
    ttc_data = safety_metrics["TTC (Time to Collision)"]
    ttc_samples = np.random.normal(ttc_data["mean"], ttc_data["std"], 1000)
    ttc_samples = np.clip(ttc_samples, 0, 10)
    axes[0, 0].hist(ttc_samples, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(ttc_data["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {ttc_data["mean"]:.2f}s')
    axes[0, 0].set_xlabel("TTC (seconds)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Time to Collision Distribution", fontweight="bold")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. PET 분포
    pet_data = safety_metrics["PET (Post-Encroachment Time)"]
    pet_samples = np.random.normal(pet_data["mean"], pet_data["std"], 1000)
    pet_samples = np.clip(pet_samples, 0, 8)
    axes[0, 1].hist(pet_samples, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(pet_data["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {pet_data["mean"]:.2f}s')
    axes[0, 1].set_xlabel("PET (seconds)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Post-Encroachment Time Distribution", fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. DRAC 분포
    drac_data = safety_metrics["DRAC (Deceleration Rate to Avoid Collision)"]
    drac_samples = np.random.normal(drac_data["mean"], drac_data["std"], 1000)
    drac_samples = np.clip(drac_samples, 0, 7)
    axes[1, 0].hist(drac_samples, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(drac_data["mean"], color='red', linestyle='--', linewidth=2, label=f'Mean: {drac_data["mean"]:.2f} m/s²')
    axes[1, 0].set_xlabel("DRAC (m/s²)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Deceleration Rate to Avoid Collision", fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Plan B 필터링 효과
    planb_data = safety_metrics["Plan B Filtered Trajectories"]
    categories = ['Safe', 'Filtered']
    values = [planb_data["safe_trajectories"] * 100, planb_data["filtered_rate"] * 100]
    colors_pie = ['green', 'red']
    axes[1, 1].pie(values, labels=categories, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    axes[1, 1].set_title("Plan B Safety Filtering", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 저장: {output_path}")


def generate_comparison_table(baselines: Dict, output_path: Path):
    """베이스라인 비교 표 생성 (CSV, LaTeX)"""
    print(f"\n[비교 표 생성]")
    
    # CSV 표
    data = []
    for model_name, metrics in baselines.items():
        data.append({
            "Model": model_name,
            "ADE (m)": f"{metrics['ade']:.4f}",
            "FDE (m)": f"{metrics['fde']:.4f}",
            "Miss Rate (%)": f"{metrics['miss_rate']*100:.2f}",
            "Collision Rate (%)": f"{metrics['collision_rate']*100:.2f}",
            "Diversity": f"{metrics['diversity']:.4f}",
            "Coverage": f"{metrics['coverage']:.4f}",
        })
    
    df = pd.DataFrame(data)
    csv_path = output_path / "baseline_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV 저장: {csv_path}")
    
    # LaTeX 표
    latex_path = output_path / "baseline_comparison.tex"
    latex_table = df.to_latex(index=False, float_format="%.4f", escape=False)
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ LaTeX 저장: {latex_path}")


def generate_summary_report(
    baselines: Dict,
    agent_types: Dict,
    scenarios: Dict,
    safety_metrics: Dict,
    significance: Dict,
    output_path: Path
):
    """종합 분석 리포트 생성"""
    print(f"\n[종합 분석 리포트 생성]")
    
    report_lines = [
        "# 실험 결과 종합 분석 리포트",
        "",
        "## 📊 1. 최종 데이터 도출",
        "",
        "### 주요 평가 지표 (HSG-Diffusion)",
        "",
        f"- **ADE**: {baselines['HSG-Diffusion (우리 모델)']['ade']:.4f} m",
        f"- **FDE**: {baselines['HSG-Diffusion (우리 모델)']['fde']:.4f} m",
        f"- **Miss Rate**: {baselines['HSG-Diffusion (우리 모델)']['miss_rate']*100:.2f}%",
        f"- **Collision Rate**: {baselines['HSG-Diffusion (우리 모델)']['collision_rate']*100:.2f}%",
        "",
        "### 다중 모달리티 평가",
        "",
        f"- **Diversity**: {baselines['HSG-Diffusion (우리 모델)']['diversity']:.4f}",
        f"- **Coverage (K=20)**: {baselines['HSG-Diffusion (우리 모델)']['coverage']:.4f}",
        "",
        "### 안전성 지표 (Plan B)",
        "",
    ]
    
    # 안전성 지표 추가
    for metric_name, metric_data in safety_metrics.items():
        if "Plan B" not in metric_name:
            report_lines.append(f"- **{metric_name}**:")
            report_lines.append(f"  - 평균: {metric_data['mean']:.2f} ± {metric_data['std']:.2f}")
            report_lines.append(f"  - 임계값 위반률: {metric_data['threshold_violations']*100:.2f}%")
            report_lines.append("")
        else:
            report_lines.append(f"- **{metric_name}**:")
            report_lines.append(f"  - 안전 궤적: {metric_data['safe_trajectories']*100:.1f}%")
            report_lines.append(f"  - 필터링된 궤적: {metric_data['filtered_rate']*100:.1f}%")
            report_lines.append("")
    
    report_lines.extend([
        "## 📈 2. 비교 대상 (Baseline)",
        "",
        "### 베이스라인 모델 성능",
        "",
        "| 모델 | ADE (m) | FDE (m) | Miss Rate (%) | Collision Rate (%) | Diversity | Coverage |",
        "|------|---------|---------|---------------|---------------------|-----------|----------|",
    ])
    
    for model_name, metrics in baselines.items():
        report_lines.append(
            f"| {model_name} | {metrics['ade']:.4f} | {metrics['fde']:.4f} | "
            f"{metrics['miss_rate']*100:.2f} | {metrics['collision_rate']*100:.2f} | "
            f"{metrics['diversity']:.4f} | {metrics['coverage']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "### 성능 개선 요약",
        "",
    ])
    
    our_model = baselines["HSG-Diffusion (우리 모델)"]
    for baseline_name, baseline_metrics in baselines.items():
        if baseline_name == "HSG-Diffusion (우리 모델)":
            continue
        ade_improvement = ((baseline_metrics['ade'] - our_model['ade']) / baseline_metrics['ade']) * 100
        fde_improvement = ((baseline_metrics['fde'] - our_model['fde']) / baseline_metrics['fde']) * 100
        report_lines.append(f"- **vs {baseline_name}**:")
        report_lines.append(f"  - ADE 개선: {ade_improvement:.1f}% ({baseline_metrics['ade']:.4f} → {our_model['ade']:.4f} m)")
        report_lines.append(f"  - FDE 개선: {fde_improvement:.1f}% ({baseline_metrics['fde']:.4f} → {our_model['fde']:.4f} m)")
        report_lines.append("")
    
    report_lines.extend([
        "### 에이전트 타입별 성능",
        "",
        "| 에이전트 타입 | ADE (m) | FDE (m) | Miss Rate (%) | 샘플 수 |",
        "|---------------|---------|---------|---------------|---------|",
    ])
    
    for agent_type, metrics in agent_types.items():
        report_lines.append(
            f"| {agent_type} | {metrics['ade']:.4f} | {metrics['fde']:.4f} | "
            f"{metrics['miss_rate']*100:.2f} | {metrics['samples']} |"
        )
    
    report_lines.extend([
        "",
        "### 시나리오별 성능",
        "",
        "| 시나리오 | ADE (m) | FDE (m) | Miss Rate (%) | Collision Rate (%) | 샘플 수 |",
        "|----------|---------|---------|---------------|---------------------|---------|",
    ])
    
    for scenario, metrics in scenarios.items():
        report_lines.append(
            f"| {scenario} | {metrics['ade']:.4f} | {metrics['fde']:.4f} | "
            f"{metrics['miss_rate']*100:.2f} | {metrics['collision_rate']*100:.2f} | {metrics['samples']} |"
        )
    
    report_lines.extend([
        "",
        "## 🔬 3. 통계적 유의성 검증",
        "",
        "### t-test 결과 (우리 모델 vs 베이스라인)",
        "",
        "| 베이스라인 | ADE 개선 (m) | FDE 개선 (m) | p-value | 유의성 (p<0.05) |",
        "|------------|--------------|--------------|---------|-----------------|",
    ])
    
    for baseline_name, sig_data in significance.items():
        significant = "✅ Yes" if sig_data['significant'] else "❌ No"
        report_lines.append(
            f"| {baseline_name} | {sig_data['ade_improvement']:.4f} | "
            f"{sig_data['fde_improvement']:.4f} | {sig_data['p_value']:.4f} | {significant} |"
        )
    
    report_lines.extend([
        "",
        "## 🎯 4. 최종 결론",
        "",
        "### 1. 성능 향상",
        f"- 우리 모델이 모든 베이스라인 대비 ADE/FDE 개선",
        f"- 평균 ADE 개선: {np.mean([s['ade_improvement'] for s in significance.values()]):.1f}%",
        f"- 평균 FDE 개선: {np.mean([s['fde_improvement'] for s in significance.values()]):.1f}%",
        "",
        "### 2. 다중 모달리티",
        f"- Diversity: {our_model['diversity']:.4f} (베이스라인 평균 대비 {((our_model['diversity'] / np.mean([b['diversity'] for k, b in baselines.items() if k != 'HSG-Diffusion (우리 모델)'])) - 1) * 100:.1f}% 향상)",
        f"- Coverage: {our_model['coverage']:.4f} (베이스라인 평균 대비 {((our_model['coverage'] / np.mean([b['coverage'] for k, b in baselines.items() if k != 'HSG-Diffusion (우리 모델)'])) - 1) * 100:.1f}% 향상)",
        "",
        "### 3. 이기종 처리",
        "- 모든 에이전트 타입에서 일관된 성능",
        "- 차량, 보행자, 자전거 등 다양한 에이전트 타입 처리 가능",
        "",
        "### 4. 안전성",
        f"- Plan B 필터링으로 {safety_metrics['Plan B Filtered Trajectories']['filtered_rate']*100:.1f}%의 위험 궤적 제거",
        f"- Collision Rate: {our_model['collision_rate']*100:.2f}% (베이스라인 평균 대비 {((our_model['collision_rate'] / np.mean([b['collision_rate'] for k, b in baselines.items() if k != 'HSG-Diffusion (우리 모델)'])) - 1) * 100:.1f}% 감소)",
        "",
        "## 📁 생성된 파일",
        "",
        "- `baseline_comparison.png`: 베이스라인 비교 차트",
        "- `agent_type_analysis.png`: 에이전트 타입별 성능 분석",
        "- `scenario_analysis.png`: 시나리오별 성능 분석",
        "- `safety_metrics.png`: 안전성 지표 시각화",
        "- `baseline_comparison.csv`: 베이스라인 비교 표 (CSV)",
        "- `baseline_comparison.tex`: 베이스라인 비교 표 (LaTeX)",
        "- `analysis_results.json`: 전체 분석 결과 (JSON)",
        "",
    ])
    
    report_path = output_path / "comprehensive_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ 리포트 저장: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="실험 결과 분석 샘플 데이터 생성")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/analysis",
        help="결과 저장 디렉토리"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("실험 결과 분석 샘플 데이터 생성")
    print("=" * 80)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 베이스라인 비교 데이터 생성
    baselines = generate_baseline_comparison()
    
    # 2. 에이전트 타입별 분석
    agent_types = generate_agent_type_analysis()
    
    # 3. 시나리오별 분석
    scenarios = generate_scenario_analysis()
    
    # 4. 안전성 지표
    safety_metrics = generate_safety_metrics()
    
    # 5. 통계적 유의성 검증
    significance = generate_statistical_significance(baselines)
    
    # 6. 시각화 생성
    plot_baseline_comparison(baselines, output_dir / "baseline_comparison.png")
    plot_agent_type_analysis(agent_types, output_dir / "agent_type_analysis.png")
    plot_scenario_analysis(scenarios, output_dir / "scenario_analysis.png")
    plot_safety_metrics(safety_metrics, output_dir / "safety_metrics.png")
    
    # 7. 비교 표 생성
    generate_comparison_table(baselines, output_dir)
    
    # 8. 전체 결과 JSON 저장
    all_results = {
        "baselines": baselines,
        "agent_types": agent_types,
        "scenarios": scenarios,
        "safety_metrics": safety_metrics,
        "statistical_significance": significance,
    }
    
    json_path = output_dir / "analysis_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ JSON 저장: {json_path}")
    
    # 9. 종합 리포트 생성
    generate_summary_report(
        baselines, agent_types, scenarios, safety_metrics, significance, output_dir
    )
    
    print("\n" + "=" * 80)
    print("✓ 모든 분석 샘플 데이터 생성 완료!")
    print("=" * 80)
    print(f"\n📁 결과 위치: {output_dir}")
    print("\n생성된 파일:")
    print("  - baseline_comparison.png: 베이스라인 비교 차트")
    print("  - agent_type_analysis.png: 에이전트 타입별 분석")
    print("  - scenario_analysis.png: 시나리오별 분석")
    print("  - safety_metrics.png: 안전성 지표")
    print("  - baseline_comparison.csv/tex: 비교 표")
    print("  - analysis_results.json: 전체 결과")
    print("  - comprehensive_analysis_report.md: 종합 리포트")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()


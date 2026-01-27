"""
F1 Data Analysis & Visualization Examples
Các ví dụ phân tích và trực quan hóa dữ liệu F1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Thiết lập style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== LOAD DATA ====================

def load_data():
    """Load dữ liệu đã crawl"""
    try:
        results = pd.read_csv('f1_data_output/f1_session_results_2023_2025.csv')
        laps = pd.read_csv('f1_data_output/f1_lap_times_2023_2025.csv')
        return results, laps
    except FileNotFoundError:
        print("❌ Không tìm thấy file dữ liệu. Hãy chạy crawler trước!")
        return None, None

# ==================== ANALYSIS 1: DRIVER PERFORMANCE ====================

def analyze_driver_performance(results):
    """Phân tích hiệu suất tay đua"""
    print("📊 PHÂN TÍCH HIỆU SUẤT TAY ĐUA")
    print("="*60)
    
    # Chỉ lấy Race results
    race_results = results[results['SessionType'] == 'R'].copy()
    
    # 1. Tổng điểm
    driver_points = race_results.groupby('BroadcastName')['Points'].sum().sort_values(ascending=False)
    
    print("\n🏆 TOP 10 TAY ĐUA THEO ĐIỂM:")
    for i, (driver, points) in enumerate(driver_points.head(10).items(), 1):
        print(f"  {i:2d}. {driver:20s} - {points:.0f} điểm")
    
    # 2. Số lần podium
    podium_count = race_results[race_results['Position'] <= 3].groupby('BroadcastName').size().sort_values(ascending=False)
    
    print("\n🥇 TOP 10 TAY ĐUA THEO PODIUM:")
    for i, (driver, count) in enumerate(podium_count.head(10).items(), 1):
        print(f"  {i:2d}. {driver:20s} - {count} lần")
    
    # 3. Tỷ lệ hoàn thành
    finish_rate = race_results.groupby('BroadcastName').apply(
        lambda x: (x['Status'] == 'Finished').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    print("\n✅ TOP 10 TAY ĐUA THEO TỶ LỆ HOÀN THÀNH:")
    for i, (driver, rate) in enumerate(finish_rate.head(10).items(), 1):
        print(f"  {i:2d}. {driver:20s} - {rate:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Top 10 điểm
    driver_points.head(10).plot(kind='barh', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Top 10 Tay Đua - Tổng Điểm', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Điểm')
    axes[0, 0].invert_yaxis()
    
    # Plot 2: Top 10 podium
    podium_count.head(10).plot(kind='barh', ax=axes[0, 1], color='gold')
    axes[0, 1].set_title('Top 10 Tay Đua - Số Lần Podium', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Số lần Podium')
    axes[0, 1].invert_yaxis()
    
    # Plot 3: Tỷ lệ hoàn thành
    finish_rate.head(10).plot(kind='barh', ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Top 10 Tay Đua - Tỷ Lệ Hoàn Thành', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Tỷ lệ (%)')
    axes[1, 0].invert_yaxis()
    
    # Plot 4: Điểm trung bình mỗi race
    avg_points = race_results.groupby('BroadcastName')['Points'].mean().sort_values(ascending=False).head(10)
    avg_points.plot(kind='barh', ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('Top 10 Tay Đua - Điểm TB/Race', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Điểm trung bình')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('f1_analysis/driver_performance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Đã lưu biểu đồ: f1_analysis/driver_performance.png")
    plt.show()

# ==================== ANALYSIS 2: TEAM COMPARISON ====================

def analyze_team_performance(results):
    """So sánh hiệu suất các đội"""
    print("\n📊 PHÂN TÍCH HIỆU SUẤT ĐỘI ĐUA")
    print("="*60)
    
    race_results = results[results['SessionType'] == 'R'].copy()
    
    # 1. Tổng điểm theo đội
    team_points = race_results.groupby('TeamName')['Points'].sum().sort_values(ascending=False)
    
    print("\n🏆 BẢN XẾP HẠNG ĐỘI ĐUA:")
    for i, (team, points) in enumerate(team_points.items(), 1):
        print(f"  {i:2d}. {team:30s} - {points:.0f} điểm")
    
    # 2. Tỷ lệ hoàn thành
    team_reliability = race_results.groupby('TeamName').apply(
        lambda x: (x['Status'] == 'Finished').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Điểm số đội
    team_points.plot(kind='barh', ax=axes[0], color='navy')
    axes[0].set_title('Tổng Điểm Theo Đội', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Điểm')
    axes[0].invert_yaxis()
    
    # Plot 2: Độ tin cậy
    team_reliability.plot(kind='barh', ax=axes[1], color='darkgreen')
    axes[1].set_title('Độ Tin Cậy Theo Đội (% Hoàn Thành)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Tỷ lệ (%)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('f1_analysis/team_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Đã lưu biểu đồ: f1_analysis/team_comparison.png")
    plt.show()

# ==================== ANALYSIS 3: QUALIFYING VS RACE ====================

def analyze_qualifying_vs_race(results):
    """Phân tích mối quan hệ Qualifying và Race"""
    print("\n📊 PHÂN TÍCH QUALIFYING VS RACE")
    print("="*60)
    
    # Lấy qualifying và race
    quali = results[results['SessionType'] == 'Q'][['Year', 'Round', 'BroadcastName', 'Position']].copy()
    race = results[results['SessionType'] == 'R'][['Year', 'Round', 'BroadcastName', 'Position', 'GridPosition']].copy()
    
    # Merge
    comparison = race.merge(quali, on=['Year', 'Round', 'BroadcastName'], 
                           suffixes=('_Race', '_Quali'))
    comparison['Position_Change'] = comparison['Position_Quali'] - comparison['Position_Race']
    
    # Top overtakers
    top_overtakers = comparison.groupby('BroadcastName')['Position_Change'].mean().sort_values(ascending=False).head(10)
    
    print("\n🚀 TOP 10 TAY ĐUA VỰC NHIỀU NHẤT:")
    for i, (driver, change) in enumerate(top_overtakers.items(), 1):
        print(f"  {i:2d}. {driver:20s} - {change:+.2f} vị trí")
    
    # Correlation
    correlation = comparison[['Position_Quali', 'Position_Race']].corr().iloc[0, 1]
    print(f"\n📈 Correlation Quali-Race: {correlation:.3f}")
    print("   (Càng gần 1 = Vị trí quali càng quan trọng)")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Overtaking ability
    top_overtakers.plot(kind='barh', ax=axes[0], color='red')
    axes[0].set_title('Khả Năng Vượt Trong Race', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Số vị trí vượt trung bình')
    axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
    axes[0].invert_yaxis()
    
    # Plot 2: Scatter Quali vs Race
    axes[1].scatter(comparison['Position_Quali'], comparison['Position_Race'], alpha=0.5)
    axes[1].plot([1, 20], [1, 20], 'r--', alpha=0.5, label='Perfect correlation')
    axes[1].set_xlabel('Vị trí Qualifying')
    axes[1].set_ylabel('Vị trí Race')
    axes[1].set_title(f'Qualifying vs Race Position (r={correlation:.3f})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('f1_analysis/quali_vs_race.png', dpi=300, bbox_inches='tight')
    print("\n✅ Đã lưu biểu đồ: f1_analysis/quali_vs_race.png")
    plt.show()

# ==================== ANALYSIS 4: LAP TIME ANALYSIS ====================

def analyze_lap_times(laps):
    """Phân tích lap times"""
    print("\n📊 PHÂN TÍCH LAP TIMES")
    print("="*60)
    
    if laps is None or len(laps) == 0:
        print("❌ Không có dữ liệu lap times")
        return
    
    # Convert lap time to seconds
    def laptime_to_seconds(laptime_str):
        if pd.isna(laptime_str):
            return np.nan
        try:
            if isinstance(laptime_str, str):
                if ':' in laptime_str:
                    parts = laptime_str.split(':')
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + float(parts[1])
                return float(laptime_str)
        except:
            return np.nan
    
    laps['LapTime_Seconds'] = laps['LapTime'].apply(laptime_to_seconds)
    laps_clean = laps[laps['LapTime_Seconds'].notna()].copy()
    
    if len(laps_clean) == 0:
        print("❌ Không có dữ liệu lap time hợp lệ")
        return
    
    # Độ ổn định của tay đua (cần ít nhất 5 laps để tính)
    driver_lap_count = laps_clean.groupby('Driver').size()
    drivers_with_enough_laps = driver_lap_count[driver_lap_count >= 5].index
    
    laps_filtered = laps_clean[laps_clean['Driver'].isin(drivers_with_enough_laps)]
    
    if len(laps_filtered) == 0:
        print("❌ Không có tay đua nào có đủ dữ liệu (cần ít nhất 5 laps)")
        return
    
    driver_consistency = laps_filtered.groupby('Driver')['LapTime_Seconds'].agg(['mean', 'std', 'count']).dropna()
    driver_consistency = driver_consistency[driver_consistency['mean'] > 0]
    driver_consistency = driver_consistency[driver_consistency['std'] > 0]  # Phải có variance
    driver_consistency['CV'] = driver_consistency['std'] / driver_consistency['mean']  # Coefficient of variation
    driver_consistency = driver_consistency.sort_values('CV')
    
    if len(driver_consistency) == 0:
        print("❌ Không có dữ liệu để phân tích consistency")
        return
    
    print("\n🎯 TOP 10 TAY ĐUA ỔN ĐỊNH NHẤT (Thấp = Ổn định):")
    top_n = min(10, len(driver_consistency))
    for i, (driver, row) in enumerate(driver_consistency.head(top_n).iterrows(), 1):
        print(f"  {i:2d}. {driver:5s} - CV: {row['CV']:.4f} (Mean: {row['mean']:.2f}s, Std: {row['std']:.2f}s, Laps: {int(row['count'])})")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Consistency
    top_drivers = driver_consistency.head(top_n)
    if len(top_drivers) > 0:
        top_drivers['CV'].plot(kind='barh', ax=axes[0], color='purple')
        axes[0].set_title('Top Tay Đua Ổn Định Nhất', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Coefficient of Variation (thấp hơn = ổn định hơn)')
        axes[0].invert_yaxis()
    
    # Plot 2: Speed vs Consistency scatter
    if len(driver_consistency) > 0:
        axes[1].scatter(driver_consistency['mean'], driver_consistency['std'], alpha=0.6, s=100)
        
        # Annotate top 5
        top_5_annotate = min(5, len(driver_consistency))
        for driver in driver_consistency.head(top_5_annotate).index:
            axes[1].annotate(driver, 
                            (driver_consistency.loc[driver, 'mean'], 
                             driver_consistency.loc[driver, 'std']),
                            fontsize=9, alpha=0.7)
        
        axes[1].set_xlabel('Thời gian vòng đua trung bình (s)')
        axes[1].set_ylabel('Độ lệch chuẩn (s)')
        axes[1].set_title('Tốc Độ vs Độ Ổn Định', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('f1_analysis/lap_time_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✅ Đã lưu biểu đồ: f1_analysis/lap_time_analysis.png")
    plt.show()

# ==================== ANALYSIS 5: SEASON TRENDS ====================

def analyze_season_trends(results):
    """Phân tích xu hướng theo mùa giải"""
    print("\n📊 PHÂN TÍCH XU HƯỚNG MÙA GIẢI")
    print("="*60)
    
    race_results = results[results['SessionType'] == 'R'].copy()
    
    # Điểm theo round cho top drivers
    points_by_round = race_results.pivot_table(
        index='Round', 
        columns='BroadcastName', 
        values='Points', 
        aggfunc='sum'
    ).fillna(0)
    
    # Cumulative points
    cumulative_points = points_by_round.cumsum()
    
    # Top 5 drivers
    final_standings = cumulative_points.iloc[-1].sort_values(ascending=False).head(5)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    for driver in final_standings.index:
        plt.plot(cumulative_points.index, cumulative_points[driver], 
                marker='o', label=driver, linewidth=2)
    
    plt.xlabel('Chặng đua', fontsize=12)
    plt.ylabel('Tổng điểm tích lũy', fontsize=12)
    plt.title('Cuộc Đua Vô Địch - Điểm Số Tích Lũy Top 5 Tay Đua', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_analysis/season_trends.png', dpi=300, bbox_inches='tight')
    print("\n✅ Đã lưu biểu đồ: f1_analysis/season_trends.png")
    plt.show()

# ==================== MAIN ====================

def main():
    """Chạy tất cả các phân tích"""
    
    # Tạo folder output
    import os
    if not os.path.exists('f1_analysis'):
        os.makedirs('f1_analysis')
    
    print("🏎️  F1 DATA ANALYSIS")
    print("="*60)
    print("\n📁 Đang load dữ liệu...")
    
    results, laps = load_data()
    
    if results is None:
        return
    
    print(f"✅ Đã load:")
    print(f"   - Session results: {len(results)} records")
    if laps is not None:
        print(f"   - Lap times: {len(laps)} records")
    
    # Chạy các phân tích
    analyze_driver_performance(results)
    analyze_team_performance(results)
    analyze_qualifying_vs_race(results)
    
    if laps is not None:
        analyze_lap_times(laps)
    
    analyze_season_trends(results)
    
    print("\n" + "="*60)
    print("✅ HOÀN TẤT TẤT CẢ PHÂN TÍCH!")
    print("📊 Các biểu đồ đã lưu tại: f1_analysis/")
    print("="*60)

if __name__ == "__main__":
    main()
"""
F1 Lap-by-Lap Analysis
Phân tích chi tiết dữ liệu vòng đua: sector times, speeds, tyres, positions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
os.makedirs('f1_lapbylap_analysis', exist_ok=True)

# ==================== LOAD DATA ====================

def load_lapbylap_data(filepath='f1_lapbylap_data/lapbylap_detailed_2024_R1.csv'):
    """Load lap-by-lap data"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded: {len(df)} lap records")
        print(f"   Drivers: {df['Driver'].nunique()}")
        print(f"   Event: {df['EventName'].iloc[0]}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        print("   Hãy chạy f1_lapbylap_crawler.py trước!")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# ==================== ANALYSIS 1: LAP TIME EVOLUTION ====================

def analyze_lap_time_evolution(df, top_n=5):
    """Phân tích sự thay đổi lap time qua các vòng đua"""
    
    print("\n" + "="*60)
    print("📊 PHÂN TÍCH LAP TIME EVOLUTION")
    print("="*60)
    
    # Get top N drivers by final position
    final_positions = df.groupby('Driver')['Position'].last().sort_values()
    top_drivers = final_positions.head(top_n).index
    
    # Filter data
    df_top = df[df['Driver'].isin(top_drivers)].copy()
    
    # Remove outliers (pit laps usually much slower)
    df_top = df_top[df_top['IsPitLap'] == False]
    
    # Plot
    plt.figure(figsize=(16, 8))
    
    for driver in top_drivers:
        driver_data = df_top[df_top['Driver'] == driver]
        plt.plot(driver_data['LapNumber'], driver_data['LapTime'], 
                marker='o', label=driver, linewidth=2, alpha=0.8)
    
    plt.xlabel('Lap Number', fontsize=12, fontweight='bold')
    plt.ylabel('Lap Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Lap Time Evolution - Top 5 Drivers', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_lapbylap_analysis/lap_time_evolution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: lap_time_evolution.png")
    plt.close()

# ==================== ANALYSIS 2: SECTOR ANALYSIS ====================

def analyze_sector_performance(df):
    """Phân tích hiệu suất từng sector"""
    
    print("\n" + "="*60)
    print("📊 PHÂN TÍCH SECTOR PERFORMANCE")
    print("="*60)
    
    # Calculate average sector times by driver
    sector_avg = df.groupby('Driver')[['Sector1Time', 'Sector2Time', 'Sector3Time']].mean()
    
    # Remove drivers with NaN
    sector_avg = sector_avg.dropna()
    
    if len(sector_avg) == 0:
        print("❌ No sector data available")
        return
    
    # Find fastest in each sector
    print("\n🏆 FASTEST IN EACH SECTOR:")
    for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        fastest_driver = sector_avg[sector].idxmin()
        fastest_time = sector_avg[sector].min()
        print(f"   {sector}: {fastest_driver} - {fastest_time:.3f}s")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, sector in enumerate(['Sector1Time', 'Sector2Time', 'Sector3Time']):
        top_10 = sector_avg[sector].nsmallest(10).sort_values()
        top_10.plot(kind='barh', ax=axes[i], color=f'C{i}')
        axes[i].set_title(f'{sector.replace("Time", "")} - Top 10', 
                         fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Time (s)')
        axes[i].invert_yaxis()
        axes[i].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('f1_lapbylap_analysis/sector_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: sector_performance.png")
    plt.close()

# ==================== ANALYSIS 3: SPEED ANALYSIS ====================

def analyze_speed_metrics(df):
    """Phân tích tốc độ tại các điểm đo"""
    
    print("\n" + "="*60)
    print("📊 PHÂN TÍCH TỐC ĐỘ")
    print("="*60)
    
    speed_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
    available_speeds = [col for col in speed_cols if col in df.columns and df[col].notna().any()]
    
    if len(available_speeds) == 0:
        print("❌ No speed data available")
        return
    
    # Calculate max speed by driver
    speed_stats = df.groupby('Driver')[available_speeds].max()
    
    # Overall max speeds
    print("\n🚀 MAX SPEEDS:")
    for col in available_speeds:
        max_speed = df[col].max()
        driver_max = df[df[col] == max_speed]['Driver'].iloc[0] if not pd.isna(max_speed) else 'N/A'
        print(f"   {col}: {max_speed:.1f} km/h ({driver_max})")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(available_speeds[:4]):
        top_10 = speed_stats[col].nlargest(10).sort_values()
        top_10.plot(kind='barh', ax=axes[i], color='red')
        axes[i].set_title(f'{col} - Top 10 Fastest', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Speed (km/h)')
        axes[i].invert_yaxis()
        axes[i].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('f1_lapbylap_analysis/speed_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: speed_analysis.png")
    plt.close()

# ==================== ANALYSIS 4: TYRE STRATEGY ====================

def analyze_tyre_strategy(df):
    """Phân tích chiến thuật lốp"""
    
    print("\n" + "="*60)
    print("📊 PHÂN TÍCH CHIẾN THUẬT LỐP")
    print("="*60)
    
    # Number of pit stops by driver
    pit_stops = df[df['IsPitLap']].groupby('Driver').size().sort_values(ascending=False)
    
    print(f"\n🔧 SỐ LẦN PIT:")
    for driver, count in pit_stops.items():
        print(f"   {driver}: {count} lần")
    
    # Tyre compound usage by driver
    print(f"\n🏎️  CHIẾN THUẬT LỐP:")
    for driver in df['Driver'].unique():
        driver_data = df[df['Driver'] == driver]
        compounds = driver_data['Compound'].value_counts()
        compound_str = " → ".join([f"{comp}({count})" for comp, count in compounds.items()])
        print(f"   {driver}: {compound_str}")
    
    # Visualize tyre life vs lap time
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Lap time vs tyre age
    compounds = df['Compound'].unique()
    for compound in compounds:
        if pd.notna(compound):
            compound_data = df[df['Compound'] == compound]
            axes[0].scatter(compound_data['TyreLife'], compound_data['LapTime'], 
                          label=compound, alpha=0.5, s=50)
    
    axes[0].set_xlabel('Tyre Life (laps)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Lap Time (s)', fontsize=11, fontweight='bold')
    axes[0].set_title('Lap Time vs Tyre Age', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Compound usage distribution
    compound_usage = df['Compound'].value_counts()
    compound_usage.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Tyre Compound Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('f1_lapbylap_analysis/tyre_strategy.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: tyre_strategy.png")
    plt.close()

# ==================== ANALYSIS 5: POSITION CHANGES ====================

def analyze_position_changes(df):
    """Phân tích thay đổi vị trí qua các vòng đua"""
    
    print("\n" + "="*60)
    print("📊 PHÂN TÍCH THAY ĐỔI VỊ TRÍ")
    print("="*60)
    
    # Get position changes by driver
    position_changes = df.groupby('Driver').agg({
        'Position': ['first', 'last'],
        'CumulativePositionChange': 'last'
    })
    position_changes.columns = ['StartPos', 'EndPos', 'TotalChange']
    position_changes['NetChange'] = position_changes['StartPos'] - position_changes['EndPos']
    position_changes = position_changes.sort_values('TotalChange', ascending=False)
    
    print("\n🚀 TOP OVERTAKERS (tổng số vị trí vượt):")
    for i, (driver, row) in enumerate(position_changes.head(5).iterrows(), 1):
        print(f"   {i}. {driver}: {row['TotalChange']:+.0f} vị trí "
              f"(P{row['StartPos']:.0f} → P{row['EndPos']:.0f})")
    
    # Visualize position evolution
    plt.figure(figsize=(16, 8))
    
    for driver in df['Driver'].unique():
        driver_data = df[df['Driver'] == driver].sort_values('LapNumber')
        plt.plot(driver_data['LapNumber'], driver_data['Position'], 
                label=driver, linewidth=1.5, marker='o', markersize=3, alpha=0.7)
    
    plt.xlabel('Lap Number', fontsize=12, fontweight='bold')
    plt.ylabel('Position', fontsize=12, fontweight='bold')
    plt.title('Position Changes Throughout Race', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Lower position number is better
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('f1_lapbylap_analysis/position_changes.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: position_changes.png")
    plt.close()

# ==================== ANALYSIS 6: PIT STOP ANALYSIS ====================

def analyze_pit_stops(df):
    """Phân tích chi tiết pit stops"""
    
    print("\n" + "="*60)
    print("📊 PHÂN TÍCH PIT STOPS")
    print("="*60)
    
    pit_data = df[df['IsPitLap'] == True].copy()
    
    if len(pit_data) == 0:
        print("❌ No pit stop data")
        return
    
    print(f"\n📊 Tổng số pit stops: {len(pit_data)}")
    
    # Pit stop timing
    print(f"\n⏱️  TIMING:")
    print(f"   - Pit sớm nhất: Lap {pit_data['LapNumber'].min()}")
    print(f"   - Pit muộn nhất: Lap {pit_data['LapNumber'].max()}")
    
    # Pit stop duration analysis
    if 'PitDuration' in pit_data.columns:
        pit_with_duration = pit_data[pit_data['PitDuration'].notna()]
        if len(pit_with_duration) > 0:
            avg_duration = pit_with_duration['PitDuration'].mean()
            min_duration = pit_with_duration['PitDuration'].min()
            max_duration = pit_with_duration['PitDuration'].max()
            
            print(f"\n🔧 PIT DURATION:")
            print(f"   - Trung bình: {avg_duration:.2f}s")
            print(f"   - Nhanh nhất: {min_duration:.2f}s")
            print(f"   - Chậm nhất: {max_duration:.2f}s")
            
            # Find fastest pit stop
            fastest_pit = pit_with_duration[pit_with_duration['PitDuration'] == min_duration]
            print(f"   - Pit nhanh nhất: {fastest_pit['Driver'].iloc[0]} (Lap {fastest_pit['LapNumber'].iloc[0]})")
    
    # Visualize pit stop laps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Pit stop timing distribution
    pit_data.groupby('LapNumber').size().plot(kind='bar', ax=axes[0], color='orange')
    axes[0].set_title('Number of Pit Stops by Lap', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lap Number')
    axes[0].set_ylabel('Number of Pit Stops')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Pit duration by driver (if available)
    if 'PitDuration' in pit_data.columns:
        pit_duration_by_driver = pit_data.groupby('Driver')['PitDuration'].mean().sort_values()
        pit_duration_by_driver.plot(kind='barh', ax=axes[1], color='red')
        axes[1].set_title('Average Pit Stop Duration by Driver', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Duration (seconds)')
        axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('f1_lapbylap_analysis/pit_stop_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: pit_stop_analysis.png")
    plt.close()

# ==================== ANALYSIS 7: RACE PACE COMPARISON ====================

def analyze_race_pace(df, top_n=5):
    """So sánh race pace giữa các tay đua"""
    
    print("\n" + "="*60)
    print("📊 PHÂN TÍCH RACE PACE")
    print("="*60)
    
    # Remove pit laps and outliers
    df_clean = df[(df['IsPitLap'] == False) & (df['LapTime'].notna())].copy()
    
    # Calculate average pace by driver
    avg_pace = df_clean.groupby('Driver')['LapTime'].mean().sort_values()
    
    print(f"\n🏎️  AVERAGE RACE PACE (excl. pit laps):")
    for i, (driver, pace) in enumerate(avg_pace.head(10).items(), 1):
        print(f"   {i:2d}. {driver}: {pace:.3f}s")
    
    # Consistency (std deviation)
    pace_consistency = df_clean.groupby('Driver')['LapTime'].std().sort_values()
    
    print(f"\n🎯 MOST CONSISTENT (lowest std dev):")
    for i, (driver, std) in enumerate(pace_consistency.head(5).items(), 1):
        print(f"   {i}. {driver}: ±{std:.3f}s")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average pace
    avg_pace.head(10).plot(kind='barh', ax=axes[0], color='blue')
    axes[0].set_title('Top 10 - Average Race Pace', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Average Lap Time (s)')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Consistency
    pace_consistency.head(10).plot(kind='barh', ax=axes[1], color='green')
    axes[1].set_title('Top 10 - Most Consistent', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Std Deviation (s) - Lower is better')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('f1_lapbylap_analysis/race_pace.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: race_pace.png")
    plt.close()

# ==================== MAIN ====================

def main():
    """Run all analyses"""
    
    print("="*60)
    print("🏎️  F1 LAP-BY-LAP ANALYSIS")
    print("="*60)
    
    # Load data
    print("\n📁 Loading data...")
    df = load_lapbylap_data()
    
    if df is None:
        return
    
    # Run analyses
    print("\n" + "="*60)
    print("🚀 STARTING ANALYSES")
    print("="*60)
    
    try:
        analyze_lap_time_evolution(df)
    except Exception as e:
        print(f"❌ Lap time evolution error: {e}")
    
    try:
        analyze_sector_performance(df)
    except Exception as e:
        print(f"❌ Sector performance error: {e}")
    
    try:
        analyze_speed_metrics(df)
    except Exception as e:
        print(f"❌ Speed metrics error: {e}")
    
    try:
        analyze_tyre_strategy(df)
    except Exception as e:
        print(f"❌ Tyre strategy error: {e}")
    
    try:
        analyze_position_changes(df)
    except Exception as e:
        print(f"❌ Position changes error: {e}")
    
    try:
        analyze_pit_stops(df)
    except Exception as e:
        print(f"❌ Pit stops error: {e}")
    
    try:
        analyze_race_pace(df)
    except Exception as e:
        print(f"❌ Race pace error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ ANALYSES COMPLETE!")
    print("📊 All visualizations saved to: f1_lapbylap_analysis/")
    print("="*60)
    
    # List created files
    print("\n📋 FILES CREATED:")
    analysis_dir = 'f1_lapbylap_analysis'
    if os.path.exists(analysis_dir):
        for file in sorted(os.listdir(analysis_dir)):
            print(f"   - {file}")

if __name__ == "__main__":
    main()
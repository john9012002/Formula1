"""
F1 Car-Specific Analysis - Complete Implementation
Phân tích chi tiết hiệu suất từng xe đua của mỗi đội
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
os.makedirs('f1_car_analysis', exist_ok=True)

# ==================== CAR ANALYZER CLASS ====================

class F1CarAnalyzer:
    """Main class for car-specific analysis"""
    
    def __init__(self, results_path, laps_path=None):
        """Initialize with data paths"""
        print("📁 Loading data...")
        self.results = pd.read_csv(results_path)
        
        if laps_path and os.path.exists(laps_path):
            self.laps = pd.read_csv(laps_path)
            print(f"✅ Loaded: {len(self.results)} results, {len(self.laps)} laps")
        else:
            self.laps = None
            print(f"✅ Loaded: {len(self.results)} results")
        
        self.teams = self.results['TeamName'].unique()
        print(f"🏎️  Teams: {len(self.teams)}")
    
    # ==================== 1. PERFORMANCE COMPARISON ====================
    
    def compare_performance(self, team_name):
        """So sánh hiệu suất 2 xe trong đội"""
        
        print(f"\n{'='*60}")
        print(f"📊 PERFORMANCE COMPARISON - {team_name}")
        print(f"{'='*60}")
        
        team_data = self.results[self.results['TeamName'] == team_name]
        drivers = team_data['BroadcastName'].unique()
        
        if len(drivers) != 2:
            print(f"⚠️ Team has {len(drivers)} drivers (expected 2)")
            return None
        
        d1, d2 = drivers[0], drivers[1]
        
        # A. Qualifying Performance
        print(f"\n🏁 QUALIFYING PERFORMANCE:")
        quali_data = team_data[team_data['SessionType'] == 'Q']
        
        if len(quali_data) > 0:
            d1_quali = quali_data[quali_data['BroadcastName'] == d1]
            d2_quali = quali_data[quali_data['BroadcastName'] == d2]
            
            d1_avg_pos = d1_quali['Position'].mean()
            d2_avg_pos = d2_quali['Position'].mean()
            
            print(f"   {d1}: Avg P{d1_avg_pos:.1f}")
            print(f"   {d2}: Avg P{d2_avg_pos:.1f}")
            print(f"   Gap: {abs(d1_avg_pos - d2_avg_pos):.2f} positions")
        
        # B. Race Performance
        print(f"\n🏎️  RACE PERFORMANCE:")
        race_data = team_data[team_data['SessionType'] == 'R']
        
        if len(race_data) > 0:
            d1_race = race_data[race_data['BroadcastName'] == d1]
            d2_race = race_data[race_data['BroadcastName'] == d2]
            
            # Average finish position
            d1_avg_finish = d1_race['Position'].mean()
            d2_avg_finish = d2_race['Position'].mean()
            
            # Points
            d1_points = d1_race['Points'].sum() if 'Points' in d1_race.columns else 0
            d2_points = d2_race['Points'].sum() if 'Points' in d2_race.columns else 0
            
            print(f"   {d1}:")
            print(f"      Avg finish: P{d1_avg_finish:.1f}")
            print(f"      Total points: {d1_points:.0f}")
            print(f"   {d2}:")
            print(f"      Avg finish: P{d2_avg_finish:.1f}")
            print(f"      Total points: {d2_points:.0f}")
            print(f"   Points gap: {abs(d1_points - d2_points):.0f}")
        
        # C. Pace (if lap data available)
        if self.laps is not None:
            print(f"\n⏱️  RACE PACE (avg lap time):")
            team_laps = self.laps[self.laps['Team'] == team_name]
            
            # Remove pit laps
            clean_laps = team_laps[team_laps.get('IsPitLap', False) == False]
            
            if len(clean_laps) > 0:
                # Convert lap time to seconds if needed
                pace_data = clean_laps.groupby('Driver')['LapTime'].apply(
                    lambda x: pd.to_numeric(x, errors='coerce').mean()
                )
                
                if d1 in pace_data.index and d2 in pace_data.index:
                    d1_pace = pace_data[d1]
                    d2_pace = pace_data[d2]
                    gap = abs(d1_pace - d2_pace)
                    
                    faster = d1 if d1_pace < d2_pace else d2
                    
                    print(f"   {d1}: {d1_pace:.3f}s")
                    print(f"   {d2}: {d2_pace:.3f}s")
                    print(f"   {faster} faster by {gap:.3f}s/lap")
        
        return {'team': team_name, 'driver1': d1, 'driver2': d2}
    
    # ==================== 2. RELIABILITY ANALYSIS ====================
    
    def compare_reliability(self, team_name):
        """Phân tích độ tin cậy"""
        
        print(f"\n{'='*60}")
        print(f"🔧 RELIABILITY ANALYSIS - {team_name}")
        print(f"{'='*60}")
        
        team_data = self.results[
            (self.results['TeamName'] == team_name) & 
            (self.results['SessionType'] == 'R')
        ]
        
        if len(team_data) == 0:
            print("⚠️ No race data")
            return
        
        drivers = team_data['BroadcastName'].unique()
        
        print(f"\n📊 FINISH STATISTICS:")
        
        for driver in drivers:
            driver_data = team_data[team_data['BroadcastName'] == driver]
            total_races = len(driver_data)
            
            if 'Status' in driver_data.columns:
                finished = (driver_data['Status'] == 'Finished').sum()
                finish_rate = finished / total_races * 100
                
                # DNF analysis
                mechanical_dnf = driver_data['Status'].str.contains(
                    'Engine|Gearbox|Mechanical|Electrical|Hydraulics|Suspension',
                    case=False, na=False
                ).sum()
                
                accident_dnf = driver_data['Status'].str.contains(
                    'Collision|Damage|Accident|Crash',
                    case=False, na=False
                ).sum()
                
                print(f"\n   {driver}:")
                print(f"      Total races: {total_races}")
                print(f"      Finished: {finished} ({finish_rate:.1f}%)")
                print(f"      DNF - Mechanical: {mechanical_dnf}")
                print(f"      DNF - Accident: {accident_dnf}")
                
                if mechanical_dnf > 0:
                    print(f"      ⚠️ Reliability concerns!")
    
    # ==================== 3. SETUP ANALYSIS ====================
    
    def compare_setup(self, team_name):
        """So sánh setup giữa 2 xe"""
        
        print(f"\n{'='*60}")
        print(f"⚙️  SETUP ANALYSIS - {team_name}")
        print(f"{'='*60}")
        
        if self.laps is None:
            print("⚠️ Lap data required for setup analysis")
            return
        
        team_laps = self.laps[self.laps['Team'] == team_name]
        drivers = team_laps['Driver'].unique()
        
        if len(drivers) != 2:
            return
        
        # A. Speed analysis (downforce indicator)
        print(f"\n🚀 TOP SPEED (Downforce indicator):")
        
        speed_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
        available_speeds = [col for col in speed_cols if col in team_laps.columns]
        
        if available_speeds:
            speed_stats = team_laps.groupby('Driver')[available_speeds].max()
            
            for col in available_speeds:
                if col in speed_stats.columns:
                    print(f"\n   {col}:")
                    for driver in drivers:
                        if driver in speed_stats.index:
                            speed = speed_stats.loc[driver, col]
                            if not pd.isna(speed):
                                print(f"      {driver}: {speed:.1f} km/h")
                    
                    if len(drivers) == 2:
                        speeds = [speed_stats.loc[d, col] for d in drivers if d in speed_stats.index]
                        if len(speeds) == 2 and not any(pd.isna(speeds)):
                            gap = abs(speeds[0] - speeds[1])
                            if gap > 3:
                                print(f"      ⚠️ Significant gap: {gap:.1f} km/h (different setup)")
        
        # B. Tyre usage pattern
        print(f"\n🏎️  TYRE STRATEGY:")
        
        for driver in drivers:
            driver_laps = team_laps[team_laps['Driver'] == driver]
            if 'Compound' in driver_laps.columns:
                compounds = driver_laps['Compound'].value_counts()
                compound_str = ", ".join([f"{c}({n})" for c, n in compounds.items()])
                print(f"   {driver}: {compound_str}")
    
    # ==================== 4. TEAMMATE HEAD-TO-HEAD ====================
    
    def teammate_head_to_head(self, team_name):
        """Head-to-head battle analysis"""
        
        print(f"\n{'='*60}")
        print(f"🏆 TEAMMATE HEAD-TO-HEAD - {team_name}")
        print(f"{'='*60}")
        
        team_data = self.results[self.results['TeamName'] == team_name]
        drivers = team_data['BroadcastName'].unique()
        
        if len(drivers) != 2:
            return
        
        d1, d2 = drivers[0], drivers[1]
        
        # Qualifying H2H
        quali_data = team_data[team_data['SessionType'] == 'Q']
        quali_h2h = {'d1': 0, 'd2': 0, 'tie': 0}
        
        for round_num in quali_data['Round'].unique():
            round_data = quali_data[quali_data['Round'] == round_num]
            d1_pos = round_data[round_data['BroadcastName'] == d1]['Position'].values
            d2_pos = round_data[round_data['BroadcastName'] == d2]['Position'].values
            
            if len(d1_pos) > 0 and len(d2_pos) > 0:
                if d1_pos[0] < d2_pos[0]:
                    quali_h2h['d1'] += 1
                elif d2_pos[0] < d1_pos[0]:
                    quali_h2h['d2'] += 1
                else:
                    quali_h2h['tie'] += 1
        
        # Race H2H
        race_data = team_data[team_data['SessionType'] == 'R']
        race_h2h = {'d1': 0, 'd2': 0, 'tie': 0}
        
        for round_num in race_data['Round'].unique():
            round_data = race_data[race_data['Round'] == round_num]
            d1_pos = round_data[round_data['BroadcastName'] == d1]['Position'].values
            d2_pos = round_data[round_data['BroadcastName'] == d2]['Position'].values
            
            if len(d1_pos) > 0 and len(d2_pos) > 0:
                if d1_pos[0] < d2_pos[0]:
                    race_h2h['d1'] += 1
                elif d2_pos[0] < d1_pos[0]:
                    race_h2h['d2'] += 1
                else:
                    race_h2h['tie'] += 1
        
        print(f"\n📊 QUALIFYING H2H: {d1} {quali_h2h['d1']}-{quali_h2h['d2']} {d2}")
        print(f"📊 RACE H2H: {d1} {race_h2h['d1']}-{race_h2h['d2']} {d2}")
        
        # Points battle
        if 'Points' in race_data.columns:
            d1_points = race_data[race_data['BroadcastName'] == d1]['Points'].sum()
            d2_points = race_data[race_data['BroadcastName'] == d2]['Points'].sum()
            
            print(f"\n💰 POINTS: {d1} {d1_points:.0f}-{d2_points:.0f} {d2}")
            
            leader = d1 if d1_points > d2_points else d2
            gap = abs(d1_points - d2_points)
            print(f"   {leader} leads by {gap:.0f} points")
    
    # ==================== 5. DEVELOPMENT TREND ====================
    
    def development_trend(self, team_name, save_plot=True):
        """Track performance evolution over season"""
        
        print(f"\n{'='*60}")
        print(f"📈 DEVELOPMENT TREND - {team_name}")
        print(f"{'='*60}")
        
        team_data = self.results[
            (self.results['TeamName'] == team_name) & 
            (self.results['SessionType'] == 'R')
        ]
        
        if len(team_data) == 0:
            print("⚠️ No race data")
            return
        
        drivers = team_data['BroadcastName'].unique()
        
        # Evolution by round
        evolution = team_data.groupby(['Round', 'BroadcastName']).agg({
            'Position': 'mean',
            'Points': 'sum'
        }).reset_index()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Position evolution
        for driver in drivers:
            driver_data = evolution[evolution['BroadcastName'] == driver]
            axes[0].plot(driver_data['Round'], driver_data['Position'], 
                        marker='o', label=driver, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Round', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Position', fontsize=11, fontweight='bold')
        axes[0].set_title(f'{team_name} - Position Evolution', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Points per race
        for driver in drivers:
            driver_data = evolution[evolution['BroadcastName'] == driver]
            axes[1].plot(driver_data['Round'], driver_data['Points'], 
                        marker='s', label=driver, linewidth=2, markersize=8)
        
        axes[1].set_xlabel('Round', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Points', fontsize=11, fontweight='bold')
        axes[1].set_title(f'{team_name} - Points per Race', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'f1_car_analysis/{team_name.replace(" ", "_")}_development.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {filename}")
        
        plt.close()
    
    # ==================== 6. COMPLETE REPORT ====================
    
    def full_car_report(self, team_name):
        """Generate complete car analysis report for a team"""
        
        print(f"\n{'#'*60}")
        print(f"# {team_name.upper()} - COMPLETE CAR ANALYSIS REPORT")
        print(f"{'#'*60}")
        
        # Run all analyses
        self.compare_performance(team_name)
        self.compare_reliability(team_name)
        self.compare_setup(team_name)
        self.teammate_head_to_head(team_name)
        self.development_trend(team_name)
        
        print(f"\n{'#'*60}")
        print(f"# END OF REPORT - {team_name}")
        print(f"{'#'*60}\n")
    
    # ==================== 7. ALL TEAMS OVERVIEW ====================
    
    def analyze_all_teams(self):
        """Run analysis for all teams"""
        
        print("="*60)
        print("🏎️  F1 CAR ANALYSIS - ALL TEAMS")
        print("="*60)
        
        for team in self.teams[:5]:  # Limit to top 5 teams for demo
            self.full_car_report(team)
            print("\n" + "="*60 + "\n")

# ==================== MAIN ====================

def main():
    """Main execution"""
    
    print("="*60)
    print("🏎️  F1 CAR-SPECIFIC ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    results_path = 'f1_data_output/f1_session_results_2023_2025.csv'
    laps_path = 'f1_data_output/f1_lap_times_2023_2025.csv'
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        print("   Please run the crawler first!")
        return
    
    analyzer = F1CarAnalyzer(results_path, laps_path)
    
    # Option 1: Analyze specific team
    print("\n🔍 Choose analysis mode:")
    print("1. Analyze specific team")
    print("2. Analyze all teams (top 5)")
    
    # For demo, analyze top 3 teams
    top_teams = analyzer.results.groupby('TeamName')['Points'].sum().nlargest(3).index
    
    print(f"\n📊 Analyzing top 3 teams by points...")
    
    for team in top_teams:
        analyzer.full_car_report(team)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📊 Charts saved to: f1_car_analysis/")
    print("="*60)

if __name__ == "__main__":
    main()
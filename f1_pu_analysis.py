"""
F1 Power Unit Manufacturer Analysis - IMPROVED VERSION
Phiên bản cải tiến với error handling tốt hơn
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
sns.set_palette("tab10")
os.makedirs('f1_pu_analysis', exist_ok=True)

# ==================== PU MAPPING ====================

PU_MAPPING = {
    'Mercedes': 'Mercedes',
    'Mercedes-AMG Petronas': 'Mercedes',
    'Aston Martin': 'Mercedes',
    'Williams': 'Mercedes',
    'McLaren': 'Mercedes',
    'Ferrari': 'Ferrari',
    'Haas F1 Team': 'Ferrari',
    'Kick Sauber': 'Ferrari',
    'Alfa Romeo': 'Ferrari',
    'Red Bull Racing': 'Honda',
    'Oracle Red Bull Racing': 'Honda',
    'RB': 'Honda',
    'AlphaTauri': 'Honda',
    'Racing Bulls': 'Honda',
    'Visa Cash App RB': 'Honda',
    'Alpine': 'Renault',
    'BWT Alpine F1 Team': 'Renault'
}

WORKS_TEAMS = {
    'Mercedes': ['Mercedes', 'Mercedes-AMG Petronas'],
    'Ferrari': ['Ferrari'],
    'Honda': ['Red Bull Racing', 'Oracle Red Bull Racing'],
    'Renault': ['Alpine', 'BWT Alpine F1 Team']
}

POWER_CIRCUITS = ['Italian', 'Belgian', 'Saudi', 'Azerbaijan', 'Bahrain']

# ==================== POWER UNIT ANALYZER CLASS ====================

class PowerUnitAnalyzer:
    """Main class for Power Unit analysis"""
    
    def __init__(self, results_path, laps_path=None):
        """Initialize with data"""
        print("="*60)
        print("🔧 F1 POWER UNIT ANALYSIS")
        print("="*60)
        print("\n📁 Loading data...")
        
        try:
            self.results = pd.read_csv(results_path)
            print(f"✅ Results: {len(self.results)} records")
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            self.results = None
            return
        
        if laps_path and os.path.exists(laps_path):
            try:
                self.laps = pd.read_csv(laps_path)
                print(f"✅ Laps: {len(self.laps)} records")
            except Exception as e:
                print(f"⚠️ Could not load laps: {e}")
                self.laps = None
        else:
            self.laps = None
            print("⚠️ No lap data available")
        
        # Add PU column
        self._add_pu_manufacturer()
        
    def _add_pu_manufacturer(self):
        """Add Power Unit manufacturer column with error handling"""
        
        try:
            # Map team to PU
            self.results['PU_Manufacturer'] = self.results['TeamName'].map(PU_MAPPING)
            
            # Handle unknown teams
            unknown_teams = self.results[self.results['PU_Manufacturer'].isna()]['TeamName'].unique()
            if len(unknown_teams) > 0:
                print(f"\n⚠️ Unknown teams (will be excluded): {unknown_teams}")
                self.results = self.results[self.results['PU_Manufacturer'].notna()]
            
            # Handle McLaren switch (Mercedes→Honda in 2025)
            if 'Year' in self.results.columns:
                mclaren_2025_mask = (
                    (self.results['TeamName'] == 'McLaren') & 
                    (self.results['Year'] >= 2025)
                )
                self.results.loc[mclaren_2025_mask, 'PU_Manufacturer'] = 'Honda'
            
            # Add to laps if available
            if self.laps is not None:
                if 'Team' in self.laps.columns:
                    self.laps['PU_Manufacturer'] = self.laps['Team'].map(PU_MAPPING)
                    
                    # Handle McLaren switch
                    if 'Year' in self.laps.columns:
                        mclaren_2025_mask = (
                            (self.laps['Team'] == 'McLaren') & 
                            (self.laps['Year'] >= 2025)
                        )
                        self.laps.loc[mclaren_2025_mask, 'PU_Manufacturer'] = 'Honda'
            
            # Verify mapping
            pu_counts = self.results.groupby('PU_Manufacturer')['TeamName'].nunique()
            
            print(f"\n🔧 Power Unit Distribution:")
            for pu, count in pu_counts.items():
                teams = self.results[self.results['PU_Manufacturer'] == pu]['TeamName'].unique()
                teams_str = ', '.join(teams[:3])
                if len(teams) > 3:
                    teams_str += f" (+{len(teams)-3} more)"
                print(f"   {pu}: {count} teams - {teams_str}")
                
        except Exception as e:
            print(f"❌ Error adding PU manufacturers: {e}")
    
    # ==================== 1. POWER ANALYSIS ====================
    
    def analyze_power(self):
        """Analyze peak power via top speed"""
        
        print(f"\n{'='*60}")
        print("🚀 POWER ANALYSIS (Top Speed)")
        print(f"{'='*60}")
        
        if self.laps is None:
            print("⚠️ Lap data required for speed analysis")
            return
        
        try:
            # Filter power circuits
            power_laps = self.laps.copy()
            
            if 'EventName' in self.laps.columns:
                power_mask = self.laps['EventName'].str.contains(
                    '|'.join(POWER_CIRCUITS), case=False, na=False
                )
                power_laps = self.laps[power_mask].copy()
                
                if len(power_laps) == 0:
                    print("⚠️ No power circuit data found, using all laps")
                    power_laps = self.laps.copy()
                else:
                    print(f"\n📊 Using {len(power_laps)} laps from power circuits")
            
            # Analyze speed metrics
            speed_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
            available_speeds = [col for col in speed_cols if col in power_laps.columns]
            
            if not available_speeds:
                print("⚠️ No speed data available")
                return
            
            print(f"📈 Available speed metrics: {', '.join(available_speeds)}")
            
            for speed_col in available_speeds:
                if power_laps[speed_col].notna().any():
                    print(f"\n🏁 {speed_col} Analysis:")
                    
                    speed_stats = power_laps.groupby('PU_Manufacturer')[speed_col].agg([
                        ('Max', 'max'),
                        ('Avg', 'mean'),
                        ('Count', 'count')
                    ]).round(1)
                    
                    # Sort by max
                    speed_stats = speed_stats.sort_values('Max', ascending=False)
                    
                    for i, (pu, row) in enumerate(speed_stats.iterrows(), 1):
                        if not pd.isna(row['Max']):
                            print(f"   {i}. {pu:10s}: Max {row['Max']:.1f} km/h, "
                                  f"Avg {row['Avg']:.1f} km/h (n={int(row['Count'])})")
            
            # Visualize
            self._plot_speed_comparison(power_laps, available_speeds)
            
        except Exception as e:
            print(f"❌ Error in power analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_speed_comparison(self, power_laps, speed_cols):
        """Plot speed comparison with error handling"""
        
        try:
            n_plots = len(speed_cols)
            if n_plots == 0:
                return
            
            fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
            
            if n_plots == 1:
                axes = [axes]
            
            for idx, speed_col in enumerate(speed_cols):
                valid_data = power_laps[power_laps[speed_col].notna()]
                
                if len(valid_data) > 0:
                    speed_data = valid_data.groupby('PU_Manufacturer')[speed_col].max().sort_values()
                    
                    if len(speed_data) > 0:
                        speed_data.plot(kind='barh', ax=axes[idx], color='darkblue')
                        axes[idx].set_title(f'{speed_col} - Max Speed', fontsize=12, fontweight='bold')
                        axes[idx].set_xlabel('Speed (km/h)')
                        axes[idx].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig('f1_pu_analysis/power_comparison.png', dpi=300, bbox_inches='tight')
            print("\n✅ Saved: power_comparison.png")
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not create speed plot: {e}")
    
    # ==================== 2. RELIABILITY ANALYSIS ====================
    
    def analyze_reliability(self):
        """Analyze PU reliability with error handling"""
        
        print(f"\n{'='*60}")
        print("🔧 RELIABILITY ANALYSIS")
        print(f"{'='*60}")
        
        try:
            race_data = self.results[self.results['SessionType'] == 'R'].copy()
            
            if len(race_data) == 0:
                print("⚠️ No race data available")
                return
            
            if 'Status' not in race_data.columns:
                print("⚠️ Status column not available")
                return
            
            reliability_stats = race_data.groupby('PU_Manufacturer').apply(
                lambda x: pd.Series({
                    'TotalRaces': len(x),
                    'Finished': (x['Status'] == 'Finished').sum(),
                    'FinishRate': (x['Status'] == 'Finished').sum() / len(x) * 100,
                    'PU_Failures': x['Status'].str.contains(
                        'Engine|Power Unit|Turbo|MGU|ERS|Electrical|Gearbox|Hydraulics',
                        case=False, na=False
                    ).sum(),
                    'Teams': x['TeamName'].nunique()
                })
            ).round(2)
            
            reliability_stats['FailureRate'] = (
                reliability_stats['PU_Failures'] / reliability_stats['TotalRaces'] * 100
            ).round(2)
            
            print(f"\n📊 RELIABILITY STATISTICS:")
            print(reliability_stats[['TotalRaces', 'Finished', 'FinishRate', 'PU_Failures', 'FailureRate']])
            
            # Ranking
            print(f"\n🏆 RELIABILITY RANKING (by finish rate):")
            ranking = reliability_stats.sort_values('FinishRate', ascending=False)
            for i, (pu, row) in enumerate(ranking.iterrows(), 1):
                print(f"   {i}. {pu:10s}: {row['FinishRate']:.1f}% "
                      f"({int(row['Finished'])}/{int(row['TotalRaces'])} races, "
                      f"{int(row['PU_Failures'])} PU failures)")
            
            # Visualize
            self._plot_reliability(reliability_stats)
            
        except Exception as e:
            print(f"❌ Error in reliability analysis: {e}")
    
    def _plot_reliability(self, reliability_stats):
        """Plot reliability comparison"""
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Finish rate
            reliability_stats['FinishRate'].sort_values().plot(
                kind='barh', ax=axes[0], color='green'
            )
            axes[0].set_title('Finish Rate by PU', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Finish Rate (%)')
            axes[0].grid(True, alpha=0.3, axis='x')
            
            # Failure rate
            reliability_stats['FailureRate'].sort_values(ascending=False).plot(
                kind='barh', ax=axes[1], color='red'
            )
            axes[1].set_title('PU Failure Rate', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Failure Rate (%)')
            axes[1].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig('f1_pu_analysis/reliability_comparison.png', dpi=300, bbox_inches='tight')
            print("\n✅ Saved: reliability_comparison.png")
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not create reliability plot: {e}")
    
    # ==================== 3. CHAMPIONSHIP IMPACT ====================
    
    def analyze_championship_impact(self):
        """Analyze PU impact on championship"""
        
        print(f"\n{'='*60}")
        print("🏆 CHAMPIONSHIP IMPACT")
        print(f"{'='*60}")
        
        try:
            race_data = self.results[self.results['SessionType'] == 'R'].copy()
            
            if len(race_data) == 0:
                print("⚠️ No race data available")
                return
            
            if 'Points' not in race_data.columns:
                print("⚠️ Points column not available")
                return
            
            championship_stats = race_data.groupby('PU_Manufacturer').agg({
                'Points': 'sum',
                'TeamName': 'nunique',
                'Position': lambda x: (x == 1).sum()
            })
            
            championship_stats.columns = ['TotalPoints', 'NumTeams', 'Wins']
            championship_stats['PointsPerTeam'] = (
                championship_stats['TotalPoints'] / championship_stats['NumTeams']
            ).round(1)
            
            # Add podiums
            podiums = race_data[race_data['Position'] <= 3].groupby('PU_Manufacturer').size()
            championship_stats['Podiums'] = podiums
            championship_stats['Podiums'] = championship_stats['Podiums'].fillna(0)
            
            championship_stats = championship_stats.sort_values('TotalPoints', ascending=False)
            
            print(f"\n📊 CHAMPIONSHIP STATISTICS:")
            print(championship_stats)
            
            print(f"\n🏆 CHAMPIONSHIP RANKING:")
            for i, (pu, row) in enumerate(championship_stats.iterrows(), 1):
                print(f"   {i}. {pu:10s}: {row['TotalPoints']:.0f} points "
                      f"({row['Wins']:.0f} wins, {row['Podiums']:.0f} podiums, "
                      f"{row['PointsPerTeam']:.1f} pts/team)")
            
            # Visualize
            self._plot_championship(championship_stats)
            
        except Exception as e:
            print(f"❌ Error in championship analysis: {e}")
    
    def _plot_championship(self, championship_stats):
        """Plot championship comparison"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            championship_stats['TotalPoints'].plot(kind='bar', ax=axes[0, 0], color='gold')
            axes[0, 0].set_title('Total Championship Points', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel('Points')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            axes[0, 0].tick_params(axis='x', rotation=0)
            
            championship_stats['PointsPerTeam'].plot(kind='bar', ax=axes[0, 1], color='silver')
            axes[0, 1].set_title('Average Points per Team', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel('Points/Team')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            axes[0, 1].tick_params(axis='x', rotation=0)
            
            championship_stats['Wins'].plot(kind='bar', ax=axes[1, 0], color='darkred')
            axes[1, 0].set_title('Total Wins', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Wins')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].tick_params(axis='x', rotation=0)
            
            championship_stats['Podiums'].plot(kind='bar', ax=axes[1, 1], color='orange')
            axes[1, 1].set_title('Total Podiums', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Podiums')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].tick_params(axis='x', rotation=0)
            
            plt.tight_layout()
            plt.savefig('f1_pu_analysis/championship_impact.png', dpi=300, bbox_inches='tight')
            print("\n✅ Saved: championship_impact.png")
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Could not create championship plot: {e}")
    
    # ==================== 4. COMPLETE REPORT ====================
    
    def full_pu_report(self):
        """Generate complete PU analysis report"""
        
        print("\n" + "#"*60)
        print("# F1 POWER UNIT ANALYSIS - COMPLETE REPORT")
        print("#"*60)
        
        self.analyze_power()
        self.analyze_reliability()
        self.analyze_championship_impact()
        
        print("\n" + "#"*60)
        print("# END OF REPORT")
        print("#"*60)

# ==================== MAIN ====================

def main():
    """Main execution"""
    
    results_path = 'f1_data_output/f1_session_results_2023_2025.csv'
    laps_path = 'f1_data_output/f1_lap_times_2023_2025.csv'
    
    if not os.path.exists(results_path):
        print(f"❌ File not found: {results_path}")
        print("   Please run crawler first!")
        return
    
    analyzer = PowerUnitAnalyzer(results_path, laps_path)
    
    if analyzer.results is not None:
        analyzer.full_pu_report()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print(f"📊 Charts saved to: f1_pu_analysis/")
        print("="*60)
    else:
        print("❌ Could not initialize analyzer")

if __name__ == "__main__":
    main()
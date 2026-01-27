"""
F1 Power Unit Manufacturer Analysis - Complete Implementation
Phân tích 4 nhà sản xuất động cơ: Mercedes, Ferrari, Honda, Renault
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

# Power Unit supplier mapping (2023-2025)
PU_MAPPING = {
    # Mercedes PU
    'Mercedes': 'Mercedes',
    'Mercedes-AMG Petronas': 'Mercedes',
    'Aston Martin': 'Mercedes',
    'Williams': 'Mercedes',
    'McLaren': 'Mercedes',  # Will handle year-specific change to Honda
    
    # Ferrari PU
    'Ferrari': 'Ferrari',
    'Haas F1 Team': 'Ferrari',
    'Kick Sauber': 'Ferrari',
    'Alfa Romeo': 'Ferrari',
    
    # Honda PU
    'Red Bull Racing': 'Honda',
    'Oracle Red Bull Racing': 'Honda',
    'RB': 'Honda',
    'AlphaTauri': 'Honda',
    'Racing Bulls': 'Honda',
    'Visa Cash App RB': 'Honda',
    
    # Renault PU
    'Alpine': 'Renault',
    'BWT Alpine F1 Team': 'Renault'
}

# Works teams (factory teams with in-house PU)
WORKS_TEAMS = {
    'Mercedes': ['Mercedes', 'Mercedes-AMG Petronas'],
    'Ferrari': ['Ferrari'],
    'Honda': ['Red Bull Racing', 'Oracle Red Bull Racing'],
    'Renault': ['Alpine', 'BWT Alpine F1 Team']
}

# Power circuits (to measure PU power)
POWER_CIRCUITS = [
    'Italian',  # Monza
    'Belgian',  # Spa
    'Saudi',    # Jeddah
    'Azerbaijan'  # Baku
]

# ==================== POWER UNIT ANALYZER CLASS ====================

class PowerUnitAnalyzer:
    """Main class for Power Unit analysis"""
    
    def __init__(self, results_path, laps_path=None):
        """Initialize with data"""
        print("="*60)
        print("🔧 F1 POWER UNIT ANALYSIS")
        print("="*60)
        print("\n📁 Loading data...")
        
        self.results = pd.read_csv(results_path)
        
        if laps_path and os.path.exists(laps_path):
            self.laps = pd.read_csv(laps_path)
            print(f"✅ Loaded: {len(self.results)} results, {len(self.laps)} laps")
        else:
            self.laps = None
            print(f"✅ Loaded: {len(self.results)} results")
        
        # Add PU column
        self._add_pu_manufacturer()
        
    def _add_pu_manufacturer(self):
        """Add Power Unit manufacturer column"""
        
        # Map team to PU
        self.results['PU_Manufacturer'] = self.results['TeamName'].map(PU_MAPPING)
        
        # Handle McLaren switch (Mercedes→Honda in 2025)
        if 'Year' in self.results.columns:
            mclaren_2025_mask = (
                (self.results['TeamName'] == 'McLaren') & 
                (self.results['Year'] >= 2025)
            )
            self.results.loc[mclaren_2025_mask, 'PU_Manufacturer'] = 'Honda'
        
        # Add to laps if available
        if self.laps is not None and 'Team' in self.laps.columns:
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
            print(f"   {pu}: {count} teams - {', '.join(teams[:3])}")
    
    # ==================== 1. POWER ANALYSIS ====================
    
    def analyze_power(self):
        """Analyze peak power via top speed on power circuits"""
        
        print(f"\n{'='*60}")
        print("🚀 POWER ANALYSIS (Top Speed)")
        print(f"{'='*60}")
        
        if self.laps is None or self.laps.empty:
            print("⚠️ No lap data available → skipping power analysis")
            return
        
        # Debug info
        print(f"   Lap data shape: {self.laps.shape}")
        print(f"   Available columns: {', '.join(self.laps.columns.tolist())}")
        
        # Power circuit filter
        power_circuit_pattern = '|'.join(POWER_CIRCUITS)
        
        if 'EventName' in self.laps.columns:
            mask = self.laps['EventName'].str.contains(power_circuit_pattern, case=False, na=False)
            power_laps = self.laps[mask].copy()
            print(f"   → Filtered {len(power_laps)} laps from power circuits "
                  f"({power_circuit_pattern})")
        else:
            print("⚠️ Column 'EventName' not found → using ALL laps for analysis")
            print("   (You may want to rename or add a circuit/grand prix column)")
            power_laps = self.laps.copy()
        
        if len(power_laps) == 0:
            print("⚠️ No laps after filtering → skipping")
            return
        
        # Speed columns - linh hoạt hơn
        speed_cols_possible = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'Speed']
        available_speeds = [col for col in speed_cols_possible if col in power_laps.columns]
        
        if not available_speeds:
            print("⚠️ No speed columns found. Checked: " + ", ".join(speed_cols_possible))
            return
        
        print(f"\n📊 Analysis based on {len(power_laps)} laps")
        print(f"   Using speed columns: {', '.join(available_speeds)}")
        
        for speed_col in available_speeds:
            if power_laps[speed_col].notna().any():
                print(f"\n🏁 {speed_col} Analysis:")
                
                speed_stats = power_laps.groupby('PU_Manufacturer')[speed_col].agg([
                    ('Max', 'max'),
                    ('Avg', 'mean'),
                    ('Std', 'std'),
                    ('Count', 'count')
                ]).round(1)
                
                speed_stats = speed_stats.sort_values('Avg', ascending=False)
                
                for i, (pu, row) in enumerate(speed_stats.iterrows(), 1):
                    print(f"   {i}. {pu:12s}: Max {row['Max']:>6.1f} | "
                          f"Avg {row['Avg']:>6.1f} (±{row['Std']:>5.1f}) | n={int(row['Count'])}")
        
        # Visualize
        self._plot_speed_comparison(power_laps, available_speeds)

    def _plot_speed_comparison(self, power_laps, speed_cols):
        """Plot speed comparison - improved version"""
        
        if not speed_cols:
            return
        
        n = len(speed_cols)
        fig, axes = plt.subplots(1, n, figsize=(7*n, 6), squeeze=False)
        axes = axes.flatten()  # đảm bảo luôn là list
        
        for i, col in enumerate(speed_cols):
            ax = axes[i]
            if power_laps[col].notna().any():
                data = power_laps.groupby('PU_Manufacturer')[col].max().sort_values(ascending=True)
                
                if not data.empty:
                    bars = data.plot(kind='barh', ax=ax, color='darkblue', edgecolor='black')
                    ax.set_title(f'{col} - Max Speed', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Speed (km/h)')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Thêm giá trị lên bar
                    for bar in bars.patches:
                        width = bar.get_width()
                        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                                f'{width:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path = 'f1_pu_analysis/power_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved power comparison chart: {save_path}")
        plt.close(fig)
    
    # ==================== 2. RELIABILITY ANALYSIS ====================
    
    def analyze_reliability(self):
        """Analyze PU reliability"""
        
        print(f"\n{'='*60}")
        print("🔧 RELIABILITY ANALYSIS")
        print(f"{'='*60}")
        
        race_data = self.results[self.results['SessionType'] == 'R']
        
        if len(race_data) == 0:
            print("⚠️ No race data")
            return
        
        reliability_stats = race_data.groupby('PU_Manufacturer').apply(
            lambda x: pd.Series({
                'TotalRaces': len(x),
                'Finished': (x['Status'] == 'Finished').sum() if 'Status' in x.columns else 0,
                'FinishRate': (x['Status'] == 'Finished').sum() / len(x) * 100 if 'Status' in x.columns else 0,
                'PU_Failures': x.get('Status', pd.Series()).str.contains(
                    'Engine|Power Unit|Turbo|MGU|ERS|Electrical|Gearbox|Hydraulics',
                    case=False, na=False
                ).sum() if 'Status' in x.columns else 0,
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
    
    def _plot_reliability(self, reliability_stats):
        """Plot reliability comparison"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Finish rate
        reliability_stats['FinishRate'].sort_values().plot(
            kind='barh', ax=axes[0], color='green'
        )
        axes[0].set_title('Finish Rate by PU', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Finish Rate (%)')
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Failure rate
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
    
    # ==================== 3. CHAMPIONSHIP IMPACT ====================
    
    def analyze_championship_impact(self):
        """Analyze PU impact on championship"""
        
        print(f"\n{'='*60}")
        print("🏆 CHAMPIONSHIP IMPACT")
        print(f"{'='*60}")
        
        race_data = self.results[self.results['SessionType'] == 'R']
        
        if 'Points' not in race_data.columns:
            print("⚠️ No points data")
            return
        
        championship_stats = race_data.groupby('PU_Manufacturer').agg({
            'Points': 'sum',
            'TeamName': 'nunique',
            'Position': lambda x: (x == 1).sum()  # Wins
        })
        
        championship_stats.columns = ['TotalPoints', 'NumTeams', 'Wins']
        championship_stats['PointsPerTeam'] = (
            championship_stats['TotalPoints'] / championship_stats['NumTeams']
        ).round(1)
        
        # Add podiums
        podiums = race_data[race_data['Position'] <= 3].groupby('PU_Manufacturer').size()
        championship_stats['Podiums'] = podiums
        
        championship_stats = championship_stats.sort_values('TotalPoints', ascending=False)
        
        print(f"\n📊 CHAMPIONSHIP STATISTICS:")
        print(championship_stats)
        
        print(f"\n🏆 CHAMPIONSHIP RANKING:")
        for i, (pu, row) in enumerate(championship_stats.iterrows(), 1):
            print(f"   {i}. {pu:10s}: {row['TotalPoints']:.0f} points "
                  f"({row['Wins']:.0f} wins, {row.get('Podiums', 0):.0f} podiums, "
                  f"{row['PointsPerTeam']:.1f} pts/team)")
        
        # Visualize
        self._plot_championship(championship_stats)
    
    def _plot_championship(self, championship_stats):
        """Plot championship comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total points
        championship_stats['TotalPoints'].plot(
            kind='bar', ax=axes[0, 0], color='gold'
        )
        axes[0, 0].set_title('Total Championship Points', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Points')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Points per team
        championship_stats['PointsPerTeam'].plot(
            kind='bar', ax=axes[0, 1], color='silver'
        )
        axes[0, 1].set_title('Average Points per Team', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Points/Team')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Wins
        championship_stats['Wins'].plot(
            kind='bar', ax=axes[1, 0], color='darkred'
        )
        axes[1, 0].set_title('Total Wins', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Wins')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Podiums
        if 'Podiums' in championship_stats.columns:
            championship_stats['Podiums'].plot(
                kind='bar', ax=axes[1, 1], color='orange'
            )
            axes[1, 1].set_title('Total Podiums', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Podiums')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('f1_pu_analysis/championship_impact.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved: championship_impact.png")
        plt.close()
    
    # ==================== 4. WORKS VS CUSTOMER ====================
    
    def analyze_works_vs_customer(self):
        """Compare works teams vs customer teams"""
        
        print(f"\n{'='*60}")
        print("🏭 WORKS vs CUSTOMER TEAMS")
        print(f"{'='*60}")
        
        results = []
        
        for pu, works_team_names in WORKS_TEAMS.items():
            pu_data = self.results[self.results['PU_Manufacturer'] == pu]
            
            if len(pu_data) == 0:
                continue
            
            # Separate works and customer
            works_data = pu_data[pu_data['TeamName'].isin(works_team_names)]
            customer_data = pu_data[~pu_data['TeamName'].isin(works_team_names)]
            
            # Race data only
            works_race = works_data[works_data['SessionType'] == 'R']
            customer_race = customer_data[customer_data['SessionType'] == 'R']
            
            if len(works_race) > 0 and len(customer_race) > 0:
                works_pos = works_race['Position'].mean()
                customer_pos = customer_race['Position'].mean()
                
                works_pts = works_race['Points'].sum() if 'Points' in works_race.columns else 0
                customer_pts = customer_race['Points'].sum() if 'Points' in customer_race.columns else 0
                
                results.append({
                    'PU': pu,
                    'Works_AvgPos': works_pos,
                    'Customer_AvgPos': customer_pos,
                    'Position_Gap': customer_pos - works_pos,
                    'Works_Points': works_pts,
                    'Customer_Points': customer_pts
                })
        
        if results:
            results_df = pd.DataFrame(results)
            
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(results_df.round(2))
            
            print(f"\n🔍 ANALYSIS:")
            for _, row in results_df.iterrows():
                gap = row['Position_Gap']
                print(f"   {row['PU']:10s}: Customer teams "
                      f"{'behind' if gap > 0 else 'ahead'} by {abs(gap):.2f} positions")
        else:
            print("⚠️ Not enough data for comparison")
    
    # ==================== 5. DEVELOPMENT TREND ====================
    
    def analyze_development_trend(self):
        """Track PU performance evolution"""
        
        print(f"\n{'='*60}")
        print("📈 DEVELOPMENT TREND")
        print(f"{'='*60}")
        
        race_data = self.results[self.results['SessionType'] == 'R']
        
        if 'Round' not in race_data.columns:
            print("⚠️ No round data available")
            return
        
        # Evolution by round
        evolution = race_data.groupby(['Round', 'PU_Manufacturer']).agg({
            'Position': 'mean',
            'Points': 'sum'
        }).reset_index()
        
        print(f"\n📊 Performance evolution tracked across {race_data['Round'].nunique()} rounds")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Position evolution
        for pu in evolution['PU_Manufacturer'].unique():
            pu_data = evolution[evolution['PU_Manufacturer'] == pu]
            axes[0].plot(pu_data['Round'], pu_data['Position'], 
                        marker='o', label=pu, linewidth=2.5, markersize=6)
        
        axes[0].set_xlabel('Round', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Average Position', fontsize=11, fontweight='bold')
        axes[0].set_title('Position Evolution by PU', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3)
        
        # Points evolution
        for pu in evolution['PU_Manufacturer'].unique():
            pu_data = evolution[evolution['PU_Manufacturer'] == pu]
            axes[1].plot(pu_data['Round'], pu_data['Points'], 
                        marker='s', label=pu, linewidth=2.5, markersize=6)
        
        axes[1].set_xlabel('Round', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Points per Round', fontsize=11, fontweight='bold')
        axes[1].set_title('Points Evolution by PU', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('f1_pu_analysis/development_trend.png', dpi=300, bbox_inches='tight')
        print("\n✅ Saved: development_trend.png")
        plt.close()
    
    # ==================== 6. COMPLETE REPORT ====================
    
    def full_pu_report(self):
        """Generate complete PU analysis report"""
        
        print("\n" + "#"*60)
        print("# F1 POWER UNIT ANALYSIS - COMPLETE REPORT")
        print("#"*60)
        
        self.analyze_power()
        self.analyze_reliability()
        self.analyze_championship_impact()
        self.analyze_works_vs_customer()
        self.analyze_development_trend()
        
        print("\n" + "#"*60)
        print("# END OF REPORT")
        print("#"*60)

# ==================== MAIN ====================

def main():
    """Main execution"""
    
    # Paths
    results_path = 'f1_data_output/f1_session_results_2023_2025.csv'
    laps_path = 'f1_data_output/f1_lap_times_2023_2025.csv'
    
    if not os.path.exists(results_path):
        print(f"❌ File not found: {results_path}")
        print("   Please run crawler first!")
        return
    
    # Initialize analyzer
    analyzer = PowerUnitAnalyzer(results_path, laps_path)
    
    # Run complete analysis
    analyzer.full_pu_report()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print(f"📊 All charts saved to: f1_pu_analysis/")
    print("="*60)

if __name__ == "__main__":
    main()
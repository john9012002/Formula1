"""
F1 Google Trends - ADVANCED ANALYSIS
Với folder organization tự động
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
import shutil

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output folders
OUTPUT_DIR = 'f1_trends_analysis'
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

def setup_folders():
    """Tạo folder structure"""
    
    print("="*60)
    print("📁 SETTING UP FOLDERS")
    print("="*60)
    
    # Create main folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print(f"\n✅ Created folder structure:")
    print(f"   📂 {OUTPUT_DIR}/")
    print(f"      📊 charts/     - All visualizations")
    print(f"      📈 data/       - Processed data files")
    print(f"      📋 reports/    - Analysis reports")

def load_demo_data():
    """Load demo data đã tạo"""
    
    print("\n" + "="*60)
    print("📊 LOADING DATA")
    print("="*60)
    
    try:
        # Load trends data
        df_trends = pd.read_csv('f1_google_trends_demo.csv', index_col=0, parse_dates=True)
        print("\n✅ Loaded trends data:")
        print(f"   Date range: {df_trends.index.min().date()} to {df_trends.index.max().date()}")
        print(f"   Data points: {len(df_trends)}")
        
        # Load regional data
        df_regional = pd.read_csv('f1_regional_interest_demo.csv', index_col=0)
        print(f"\n✅ Loaded regional data:")
        print(f"   Countries: {len(df_regional)}")
        
        # Load queries
        df_top_queries = pd.read_csv('f1_top_queries_demo.csv')
        df_rising_queries = pd.read_csv('f1_rising_queries_demo.csv')
        print(f"\n✅ Loaded query data:")
        print(f"   Top queries: {len(df_top_queries)}")
        print(f"   Rising queries: {len(df_rising_queries)}")
        
        return df_trends, df_regional, df_top_queries, df_rising_queries
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Please run Google_demo.py first!")
        return None, None, None, None

def analyze_growth_trends(df_trends):
    """Phân tích xu hướng tăng trưởng"""
    
    print("\n" + "="*60)
    print("📈 GROWTH TREND ANALYSIS")
    print("="*60)
    
    # Year-over-year growth
    df_yearly = df_trends.resample('Y').mean()
    
    print("\n📊 Annual Average Search Interest:")
    for year, row in df_yearly.iterrows():
        print(f"   {year.year}: Formula 1 = {row['Formula 1']:.1f}, "
              f"F1 = {row['F1']:.1f}, DTS = {row['Drive to Survive']:.1f}")
    
    # Calculate YoY growth
    print("\n📈 Year-over-Year Growth:")
    growth_data = []
    
    for i in range(1, len(df_yearly)):
        year = df_yearly.index[i].year
        prev_year = df_yearly.index[i-1].year
        
        for col in df_yearly.columns:
            growth = ((df_yearly[col].iloc[i] - df_yearly[col].iloc[i-1]) / 
                     df_yearly[col].iloc[i-1] * 100)
            print(f"   {year} vs {prev_year} - {col}: {growth:+.1f}%")
            
            growth_data.append({
                'Year': year,
                'Metric': col,
                'Growth': growth
            })
    
    # Save growth data
    pd.DataFrame(growth_data).to_csv(
        os.path.join(DATA_DIR, 'yearly_growth.csv'), 
        index=False
    )
    
    # Best year
    best_year = df_yearly['Formula 1'].idxmax().year
    print(f"\n🏆 Peak Year: {best_year}")
    
    return df_yearly

def analyze_seasonality(df_trends):
    """Phân tích tính thời vụ"""
    
    print("\n" + "="*60)
    print("📅 SEASONALITY ANALYSIS")
    print("="*60)
    
    # Monthly average
    df_trends['Month'] = df_trends.index.month
    monthly_avg = df_trends.groupby('Month')[['Formula 1', 'F1']].mean()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print("\n📊 Average Interest by Month:")
    for month, row in monthly_avg.iterrows():
        print(f"   {month_names[month-1]:3s}: Formula 1 = {row['Formula 1']:.1f}, "
              f"F1 = {row['F1']:.1f}")
    
    # Save monthly data
    monthly_avg.to_csv(os.path.join(DATA_DIR, 'monthly_average.csv'))
    
    # Identify peak season
    peak_months = monthly_avg['Formula 1'].nlargest(3).index.tolist()
    peak_month_names = [month_names[m-1] for m in peak_months]
    
    print(f"\n🔥 Peak Season Months: {', '.join(peak_month_names)}")
    print(f"   (Race season: March - November)")
    
    # Visualize seasonality
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Monthly pattern
    monthly_avg.plot(kind='bar', ax=axes[0], color=['steelblue', 'orange'])
    axes[0].set_title('Average Search Interest by Month', fontweight='bold')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Search Interest')
    axes[0].set_xticklabels(month_names, rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend(['Formula 1', 'F1'])
    
    # Heatmap
    df_heatmap = df_trends.copy()
    df_heatmap['Year'] = df_heatmap.index.year
    heatmap_data = df_heatmap.pivot_table(
        values='Formula 1',
        index='Month',
        columns='Year',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                ax=axes[1], cbar_kws={'label': 'Search Interest'})
    axes[1].set_title('F1 Interest Heatmap (Month vs Year)', fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Month')
    axes[1].set_yticklabels(month_names, rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'seasonality_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print("\n✅ Saved: charts/seasonality_analysis.png")
    
    plt.close()
    
    return monthly_avg

def analyze_dts_impact(df_trends):
    """Phân tích impact của Drive to Survive"""
    
    print("\n" + "="*60)
    print("🎬 DRIVE TO SURVIVE IMPACT")
    print("="*60)
    
    # Compare pre and post DTS
    pre_dts = df_trends[df_trends.index.year < 2019]['Formula 1'].mean()
    post_dts = df_trends[df_trends.index.year >= 2019]['Formula 1'].mean()
    
    impact = ((post_dts - pre_dts) / pre_dts * 100)
    
    print(f"\n📊 Before DTS (2017-2018): Avg interest = {pre_dts:.1f}")
    print(f"📊 After DTS (2019-2025): Avg interest = {post_dts:.1f}")
    print(f"📈 Impact: +{impact:.1f}% increase")
    
    # Save impact data
    impact_data = pd.DataFrame({
        'Period': ['Pre-DTS', 'Post-DTS'],
        'Average_Interest': [pre_dts, post_dts],
        'Impact_%': [0, impact]
    })
    impact_data.to_csv(os.path.join(DATA_DIR, 'dts_impact.csv'), index=False)
    
    # DTS correlation
    corr = df_trends['Formula 1'].corr(df_trends['Drive to Survive'])
    print(f"\n🔗 Correlation between F1 & DTS search: {corr:.3f}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot both
    ax.plot(df_trends.index, df_trends['Formula 1'], 
           label='Formula 1', linewidth=2.5, color='red', alpha=0.8)
    ax.plot(df_trends.index, df_trends['Drive to Survive'], 
           label='Drive to Survive', linewidth=2.5, color='blue', alpha=0.8)
    
    # Mark DTS launch
    ax.axvline(pd.Timestamp('2019-03-08'), color='green', linestyle='--', 
              linewidth=2, alpha=0.7, label='DTS S1 Launch')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Search Interest', fontsize=12, fontweight='bold')
    ax.set_title('Drive to Survive Impact on F1 Search Interest', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'dts_impact.png'), 
                dpi=300, bbox_inches='tight')
    print("\n✅ Saved: charts/dts_impact.png")
    
    plt.close()

def analyze_regional_patterns(df_regional):
    """Phân tích patterns theo khu vực"""
    
    print("\n" + "="*60)
    print("🌍 REGIONAL PATTERNS ANALYSIS")
    print("="*60)
    
    # Group by continent (simplified)
    regions = {
        'Europe': ['Netherlands', 'Monaco', 'Belgium', 'United Kingdom', 
                  'Italy', 'Spain', 'Germany', 'France', 'Austria', 'Hungary', 'Poland'],
        'Americas': ['United States', 'Mexico', 'Brazil', 'Canada'],
        'Asia-Pacific': ['Singapore', 'Australia', 'Japan'],
        'Middle East': ['Saudi Arabia', 'United Arab Emirates']
    }
    
    regional_avg = {}
    regional_details = []
    
    for region, countries in regions.items():
        values = [df_regional.loc[c, 'Formula 1'] for c in countries 
                 if c in df_regional.index]
        if values:
            regional_avg[region] = np.mean(values)
            for country in countries:
                if country in df_regional.index:
                    regional_details.append({
                        'Region': region,
                        'Country': country,
                        'Interest': df_regional.loc[country, 'Formula 1']
                    })
    
    # Save regional data
    pd.DataFrame(regional_details).to_csv(
        os.path.join(DATA_DIR, 'regional_breakdown.csv'),
        index=False
    )
    
    print("\n📊 Average Interest by Region:")
    for region, avg in sorted(regional_avg.items(), key=lambda x: x[1], reverse=True):
        print(f"   {region:15s}: {avg:.1f}")
    
    # Top 3 per region
    print("\n🏆 Top Countries per Region:")
    for region, countries in regions.items():
        region_data = df_regional[df_regional.index.isin(countries)]
        if len(region_data) > 0:
            top_3 = region_data.nlargest(3, 'Formula 1')
            print(f"\n{region}:")
            for i, (country, row) in enumerate(top_3.iterrows(), 1):
                print(f"   {i}. {country}: {row['Formula 1']}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Regional averages
    pd.Series(regional_avg).sort_values().plot(kind='barh', ax=axes[0], color='teal')
    axes[0].set_title('Average Interest by Region', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Search Interest')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Top 15 countries
    df_regional.nlargest(15, 'Formula 1').sort_values('Formula 1')['Formula 1'].plot(
        kind='barh', ax=axes[1], color='crimson'
    )
    axes[1].set_title('Top 15 Countries', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Search Interest')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'regional_patterns.png'), 
                dpi=300, bbox_inches='tight')
    print("\n✅ Saved: charts/regional_patterns.png")
    
    plt.close()

def analyze_query_insights(df_top_queries, df_rising_queries):
    """Phân tích insights từ queries"""
    
    print("\n" + "="*60)
    print("🔍 SEARCH QUERY INSIGHTS")
    print("="*60)
    
    # Categorize queries
    categories = {
        'Schedules/Results': ['schedule', 'results', 'live', 'standings'],
        'Drivers': ['verstappen', 'hamilton', 'norris', 'leclerc', 'piastri', 'russell', 'alonso'],
        'Teams': ['ferrari', 'red bull', 'mercedes', 'mclaren'],
        'Races/Tracks': ['monaco', 'vegas', 'las vegas', 'miami', 'singapore'],
        'General Info': ['news', 'f1 2024', 'academy']
    }
    
    # Categorize each query
    categorized = []
    for idx, row in df_top_queries.iterrows():
        query = row['query']
        category = 'Other'
        for cat, keywords in categories.items():
            if any(kw in query.lower() for kw in keywords):
                category = cat
                break
        categorized.append({
            'Query': query,
            'Category': category,
            'Value': row['value']
        })
    
    # Save categorized queries
    pd.DataFrame(categorized).to_csv(
        os.path.join(DATA_DIR, 'queries_categorized.csv'),
        index=False
    )
    
    print("\n📊 Query Categories:")
    for category, keywords in categories.items():
        count = sum(1 for q in df_top_queries['query'] 
                   if any(kw in q.lower() for kw in keywords))
        print(f"   {category:20s}: {count} queries")
    
    # Rising trends analysis
    print("\n📈 Rising Trends Analysis:")
    
    # Extract numeric growth rates
    numeric_rising = []
    for idx, row in df_rising_queries.iterrows():
        value = row['value']
        if isinstance(value, str) and '%' in value:
            try:
                growth = int(value.replace('%', '').replace('+', ''))
                numeric_rising.append((row['query'], growth))
            except:
                pass
    
    if numeric_rising:
        numeric_rising.sort(key=lambda x: x[1], reverse=True)
        print("\n   Top 5 Fastest Growing:")
        for i, (query, growth) in enumerate(numeric_rising[:5], 1):
            print(f"   {i}. {query}: +{growth}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top queries
    df_top_queries.set_index('query').head(10).sort_values('value')['value'].plot(
        kind='barh', ax=axes[0], color='navy'
    )
    axes[0].set_title('Top 10 Related Queries', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Search Volume')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Category distribution (pie)
    category_counts = {cat: sum(1 for q in df_top_queries['query'] 
                               if any(kw in q.lower() for kw in keywords))
                      for cat, keywords in categories.items()}
    
    non_zero = {k: v for k, v in category_counts.items() if v > 0}
    axes[1].pie(non_zero.values(), labels=non_zero.keys(), autopct='%1.1f%%',
               startangle=90)
    axes[1].set_title('Query Category Distribution', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'query_insights.png'), 
                dpi=300, bbox_inches='tight')
    print("\n✅ Saved: charts/query_insights.png")
    
    plt.close()

def generate_executive_summary(df_trends, df_regional, df_yearly):
    """Tạo executive summary"""
    
    print("\n" + "="*60)
    print("📋 EXECUTIVE SUMMARY")
    print("="*60)
    
    # Key metrics
    latest_interest = df_trends['Formula 1'].iloc[-1]
    avg_interest = df_trends['Formula 1'].mean()
    peak_interest = df_trends['Formula 1'].max()
    
    # Growth
    five_year_growth = ((df_trends['Formula 1'].iloc[-1] - df_trends['Formula 1'].iloc[0]) / 
                       df_trends['Formula 1'].iloc[0] * 100)
    
    # Regional
    top_country = df_regional['Formula 1'].idxmax()
    top_country_value = df_regional['Formula 1'].max()
    
    summary_text = f"""
📊 KEY METRICS:
   Current Interest Level:     {latest_interest:.0f}/100
   Average (5 years):          {avg_interest:.1f}/100
   Peak Interest:              {peak_interest:.0f}/100
   5-Year Growth:              {five_year_growth:+.1f}%

🌍 GEOGRAPHIC:
   Top Country:                {top_country} ({top_country_value})
   Total Countries Tracked:    {len(df_regional)}

📈 TRENDS:
   Strongest Growth Period:    {df_yearly['Formula 1'].diff().idxmax().year}
   Most Consistent Year:       {df_yearly.index[df_yearly['Formula 1'].values.argmax()].year}

💡 KEY INSIGHTS:
   1. F1 interest has grown {five_year_growth:.1f}% over 5 years
   2. Peak season is March-November (race calendar)
   3. Drive to Survive significantly boosted search interest
   4. {top_country} shows highest regional interest (driver effect)
   5. Rising interest in new markets (USA, Middle East)
    """
    
    print(summary_text)
    
    # Save summary
    report_path = os.path.join(REPORTS_DIR, 'executive_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("F1 GOOGLE TRENDS - EXECUTIVE SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(summary_text)
    
    print(f"\n✅ Saved: reports/executive_summary.txt")

def create_readme():
    """Tạo README file"""
    
    readme_content = """# F1 Google Trends Analysis

## 📁 Folder Structure

```
f1_trends_analysis/
├── charts/              # All visualizations
│   ├── seasonality_analysis.png
│   ├── dts_impact.png
│   ├── regional_patterns.png
│   └── query_insights.png
│
├── data/                # Processed data
│   ├── yearly_growth.csv
│   ├── monthly_average.csv
│   ├── dts_impact.csv
│   ├── regional_breakdown.csv
│   └── queries_categorized.csv
│
└── reports/             # Analysis reports
    └── executive_summary.txt

```

## 📊 Charts

1. **seasonality_analysis.png** - Monthly patterns and yearly heatmap
2. **dts_impact.png** - Drive to Survive effect timeline
3. **regional_patterns.png** - Geographic distribution
4. **query_insights.png** - Search behavior analysis

## 📈 Data Files

All processed data in CSV format for further analysis.

## 📋 Reports

Executive summary with key findings and recommendations.

---

Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    readme_path = os.path.join(OUTPUT_DIR, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n✅ Created: README.md")

def main():
    """Main analysis function"""
    
    # Setup folders
    setup_folders()
    
    # Load data
    df_trends, df_regional, df_top_queries, df_rising_queries = load_demo_data()
    
    if df_trends is None:
        return
    
    # Run analyses
    print("\n" + "="*60)
    print("🚀 RUNNING ADVANCED ANALYSES")
    print("="*60)
    
    # 1. Growth trends
    df_yearly = analyze_growth_trends(df_trends)
    
    # 2. Seasonality
    analyze_seasonality(df_trends)
    
    # 3. DTS impact
    analyze_dts_impact(df_trends)
    
    # 4. Regional patterns
    analyze_regional_patterns(df_regional)
    
    # 5. Query insights
    analyze_query_insights(df_top_queries, df_rising_queries)
    
    # 6. Executive summary
    generate_executive_summary(df_trends, df_regional, df_yearly)
    
    # 7. Create README
    create_readme()
    
    # Final summary
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
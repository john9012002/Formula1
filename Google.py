"""
F1 Google Trends - DEMO DATA
Dùng data mẫu khi bị rate limited
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def generate_demo_trends_data():
    """
    Tạo demo data dựa trên patterns thực tế của F1
    """
    
    print("="*60)
    print("🏎️  F1 GOOGLE TRENDS - DEMO MODE")
    print("   (Using realistic sample data)")
    print("="*60)
    
    # Generate dates (last 5 years, weekly)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Create realistic F1 search patterns
    n = len(date_range)
    
    # Base trend (growing over time)
    base_trend = np.linspace(40, 70, n)
    
    # Seasonal pattern (race season March-November)
    months = np.array([d.month for d in date_range])
    seasonal = np.where((months >= 3) & (months <= 11), 15, -10)
    
    # Random noise
    noise = np.random.normal(0, 5, n)
    
    # Formula 1 interest
    formula1_interest = base_trend + seasonal + noise
    formula1_interest = np.clip(formula1_interest, 0, 100)
    
    # F1 interest (slightly different pattern)
    f1_interest = formula1_interest * 0.85 + np.random.normal(0, 3, n)
    f1_interest = np.clip(f1_interest, 0, 100)
    
    # Drive to Survive boost (started 2019, big impact)
    dts_boost = np.zeros(n)
    for i, date in enumerate(date_range):
        if date.year >= 2019:
            # March releases
            if date.month == 3:
                dts_boost[i] = 20
            elif date.month in [2, 4]:
                dts_boost[i] = 10
    
    dts_interest = formula1_interest * 0.7 + dts_boost + np.random.normal(0, 5, n)
    dts_interest = np.clip(dts_interest, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Formula 1': formula1_interest.astype(int),
        'F1': f1_interest.astype(int),
        'Drive to Survive': dts_interest.astype(int)
    }, index=date_range)
    
    return df

def analyze_demo_data(df):
    """Analyze demo trends data"""
    
    print("\n📊 F1 Search Interest (0-100 scale):")
    print(df.tail(10))
    
    print("\n📈 Statistics:")
    for keyword in df.columns:
        avg = df[keyword].mean()
        max_val = df[keyword].max()
        min_val = df[keyword].min()
        latest = df[keyword].iloc[-1]
        max_date = df[keyword].idxmax()
        
        print(f"\n{keyword}:")
        print(f"  Average: {avg:.1f}")
        print(f"  Max: {max_val} (on {max_date.date()})")
        print(f"  Min: {min_val}")
        print(f"  Latest: {latest}")
    
    # Save
    df.to_csv('f1_google_trends_demo.csv')
    print("\n✅ Saved: f1_google_trends_demo.csv")
    
    # Plot
    plt.figure(figsize=(14, 7))
    
    for keyword in df.columns:
        plt.plot(df.index, df[keyword], 
                label=keyword, linewidth=2.5, marker='o', markersize=2, alpha=0.8)
    
    # Highlight key events
    plt.axvline(pd.Timestamp('2019-03-01'), color='red', linestyle='--', 
               alpha=0.5, label='DTS S1 Release')
    plt.axvline(pd.Timestamp('2021-12-01'), color='orange', linestyle='--', 
               alpha=0.5, label='2021 Title Fight')
    
    plt.xlabel('Date', fontsize=12, fontweight='bold')
    plt.ylabel('Search Interest (0-100)', fontsize=12, fontweight='bold')
    plt.title('F1 Search Interest Over Time - Demo Data (Realistic Pattern)', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig('f1_google_trends_demo.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: f1_google_trends_demo.png")
    plt.show()
    
    return df

def generate_regional_demo_data():
    """Generate demo regional interest data"""
    
    print("\n🌍 REGIONAL INTEREST - DEMO DATA:")
    print("-"*60)
    
    # Top countries for F1 (realistic)
    countries = {
        'Netherlands': 100,  # Max Verstappen effect
        'Monaco': 95,
        'Belgium': 88,
        'United Kingdom': 85,
        'Singapore': 82,
        'Italy': 80,
        'Australia': 78,
        'United States': 75,  # Growing
        'Mexico': 72,
        'Brazil': 70,
        'Spain': 68,
        'Germany': 65,
        'France': 63,
        'Canada': 60,
        'Japan': 58,
        'Saudi Arabia': 55,
        'United Arab Emirates': 52,
        'Austria': 50,
        'Hungary': 48,
        'Poland': 45
    }
    
    df_regional = pd.DataFrame({
        'Country': countries.keys(),
        'Formula 1': countries.values()
    }).set_index('Country')
    
    print("\n🏆 Top 20 Countries - Search Interest:")
    for idx, (country, interest) in enumerate(countries.items(), 1):
        print(f"   {idx:2d}. {country:25s}: {interest}")
    
    # Save
    df_regional.to_csv('f1_regional_interest_demo.csv')
    print("\n✅ Saved: f1_regional_interest_demo.csv")
    
    # Plot
    plt.figure(figsize=(10, 8))
    df_regional.head(10).sort_values('Formula 1')['Formula 1'].plot(
        kind='barh', color='steelblue'
    )
    plt.xlabel('Search Interest', fontsize=11, fontweight='bold')
    plt.title('Top 10 Countries - F1 Interest (Demo)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('f1_regional_interest_demo.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: f1_regional_interest_demo.png")
    
    return df_regional

def generate_related_queries_demo():
    """Generate demo related queries"""
    
    print("\n🔍 RELATED QUERIES - DEMO DATA:")
    print("-"*60)
    
    top_queries = {
        'f1 schedule': 100,
        'f1 results': 95,
        'formula 1 standings': 90,
        'f1 live': 85,
        'max verstappen': 82,
        'lewis hamilton': 78,
        'ferrari f1': 75,
        'red bull racing': 72,
        'monaco grand prix': 70,
        'f1 news': 68
    }
    
    rising_queries = {
        'las vegas gp': 'Breakout',
        'lando norris': '+500%',
        'oscar piastri': '+450%',
        'f1 academy': '+320%',
        'george russell': '+280%',
        'vegas f1': '+250%',
        'charles leclerc': '+200%',
        'f1 2024': '+180%',
        'miami gp': '+150%',
        'fernando alonso': '+120%'
    }
    
    print("\n📊 Top Related Queries:")
    for query, value in top_queries.items():
        print(f"   {query:25s}: {value}")
    
    print("\n📈 Rising Queries:")
    for query, growth in rising_queries.items():
        print(f"   {query:25s}: {growth}")
    
    # Save
    pd.DataFrame({
        'query': top_queries.keys(),
        'value': top_queries.values()
    }).to_csv('f1_top_queries_demo.csv', index=False)
    
    pd.DataFrame({
        'query': rising_queries.keys(),
        'value': rising_queries.values()
    }).to_csv('f1_rising_queries_demo.csv', index=False)
    
    print("\n✅ Saved: f1_top_queries_demo.csv")
    print("✅ Saved: f1_rising_queries_demo.csv")

def main():
    """Main demo function"""
    
    print("\n⚠️  NOTE: This is DEMO data with realistic patterns")
    print("    For real data, wait 1-2 hours or use VPN")
    print()
    
    # Generate and analyze trends data
    df = generate_demo_trends_data()
    analyze_demo_data(df)
    
    # Regional data
    df_regional = generate_regional_demo_data()
    
    # Related queries
    generate_related_queries_demo()
    
    print("\n" + "="*60)
    print("✅ DEMO ANALYSIS COMPLETE")
    print("="*60)
    
    print("\n📁 Files created:")
    print("   - f1_google_trends_demo.csv")
    print("   - f1_google_trends_demo.png")
    print("   - f1_regional_interest_demo.csv")
    print("   - f1_regional_interest_demo.png")
    print("   - f1_top_queries_demo.csv")
    print("   - f1_rising_queries_demo.csv")
    
    print("\n💡 To get REAL data:")
    print("   1. Wait 1-2 hours")
    print("   2. Use VPN to change IP")
    print("   3. Run: python Google_minimal.py")

if __name__ == "__main__":
    main()
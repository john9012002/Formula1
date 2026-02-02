"""
Analyze F1 search trends with Google Trends
FIXED VERSION - With Rate Limiting Protection & Retry Logic
"""

from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError, ResponseError
import pandas as pd
import matplotlib.pyplot as plt
import time
import random

def setup_pytrends():
    """Setup PyTrends with backoff"""
    # Add random delay to avoid rate limiting
    time.sleep(random.uniform(1, 3))
    pytrends = TrendReq(
        hl='en-US', 
        tz=360,
        timeout=(10, 25),  # Connection and read timeout
        retries=2,
        backoff_factor=0.5
    )
    return pytrends

def safe_request(func, max_retries=3, base_delay=5):
    """Wrapper for safe API requests with exponential backoff"""
    
    for attempt in range(max_retries):
        try:
            # Random delay before request
            time.sleep(random.uniform(2, 5))
            return func()
            
        except TooManyRequestsError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 5)
                print(f"⚠️ Rate limited. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            else:
                print("❌ Max retries reached. Google Trends is blocking requests.")
                print("💡 Solutions:")
                print("   1. Wait 1-2 hours before trying again")
                print("   2. Use VPN to change IP")
                print("   3. Reduce number of queries")
                return None
                
        except ResponseError as e:
            print(f"❌ Response error: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                return None
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    return None

def analyze_f1_interest(pytrends, keywords=['Formula 1', 'F1', 'Drive to Survive']):
    """Analyze F1 search interest"""
    
    print("📈 Analyzing F1 Search Interest...")
    
    def _get_interest():
        # Build payload
        pytrends.build_payload(
            keywords,
            cat=0,
            timeframe='today 5-y',  # Last 5 years
            geo='',  # Worldwide
            gprop=''
        )
        
        # Get interest over time
        return pytrends.interest_over_time()
    
    # Use safe request wrapper
    interest_over_time = safe_request(_get_interest)
    
    if interest_over_time is not None and not interest_over_time.empty:
        # Remove isPartial column
        if 'isPartial' in interest_over_time.columns:
            interest_over_time = interest_over_time.drop('isPartial', axis=1)
        
        print("\n📊 F1 Search Interest (0-100 scale):")
        print(interest_over_time.tail(10))
        
        # Plot
        plt.figure(figsize=(14, 6))
        for keyword in keywords:
            if keyword in interest_over_time.columns:
                plt.plot(interest_over_time.index, interest_over_time[keyword], 
                        label=keyword, linewidth=2, marker='o', markersize=3, alpha=0.7)
        
        plt.xlabel('Date', fontsize=11, fontweight='bold')
        plt.ylabel('Search Interest', fontsize=11, fontweight='bold')
        plt.title('F1 Search Interest Over Time (Google Trends)', 
                 fontsize=13, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('f1_google_trends.png', dpi=300)
        print("\n✅ Saved: f1_google_trends.png")
        
        return interest_over_time
    
    print("❌ Could not get interest over time data")
    return None

def analyze_regional_interest(pytrends, keyword='Formula 1'):
    """Analyze interest by region with rate limiting protection"""
    
    print(f"\n🌍 Analyzing Regional Interest for '{keyword}'...")
    
    def _get_regional():
        pytrends.build_payload([keyword], timeframe='today 12-m')
        
        # Interest by region
        return pytrends.interest_by_region(
            resolution='COUNTRY',
            inc_low_vol=False,
            inc_geo_code=False
        )
    
    # Use safe request wrapper
    regional_interest = safe_request(_get_regional)
    
    if regional_interest is not None and not regional_interest.empty:
        # Top 20 countries
        top_countries = regional_interest.nlargest(20, keyword)
        
        print(f"\n🏆 Top 20 Countries - '{keyword}' Search Interest:")
        for idx, (country, row) in enumerate(top_countries.iterrows(), 1):
            print(f"   {idx:2d}. {country:20s}: {row[keyword]}")
        
        # Save to CSV
        top_countries.to_csv('f1_regional_interest.csv')
        print("\n✅ Saved: f1_regional_interest.csv")
        
        return regional_interest
    
    print("❌ Could not get regional interest data")
    return None

def analyze_related_queries(pytrends, keyword='Formula 1'):
    """Get related queries with rate limiting protection"""
    
    print(f"\n🔍 Analyzing Related Queries for '{keyword}'...")
    
    def _get_related():
        pytrends.build_payload([keyword], timeframe='today 12-m')
        return pytrends.related_queries()
    
    # Use safe request wrapper
    related_queries = safe_request(_get_related)
    
    if related_queries is not None and keyword in related_queries:
        top_queries = related_queries[keyword]['top']
        rising_queries = related_queries[keyword]['rising']
        
        print(f"\n📊 Top Related Queries for '{keyword}':")
        if top_queries is not None and not top_queries.empty:
            print(top_queries.head(10))
            top_queries.to_csv('f1_top_queries.csv', index=False)
        else:
            print("   No top queries available")
        
        print(f"\n📈 Rising Queries:")
        if rising_queries is not None and not rising_queries.empty:
            print(rising_queries.head(10))
            rising_queries.to_csv('f1_rising_queries.csv', index=False)
        else:
            print("   No rising queries available")
        
        return related_queries
    
    print("❌ Could not get related queries data")
    return None

def main():
    """Main function with proper error handling"""
    
    print("="*60)
    print("🏎️  F1 GOOGLE TRENDS ANALYSIS")
    print("="*60)
    
    # Setup
    print("\n⚙️  Setting up PyTrends...")
    pytrends = setup_pytrends()
    print("✅ Ready!\n")
    
    # Analysis 1: Interest over time
    print("-"*60)
    interest_df = analyze_f1_interest(pytrends, ['Formula 1', 'F1', 'Drive to Survive'])
    
    # Wait between requests
    if interest_df is not None:
        print("\n⏳ Waiting before next request...")
        time.sleep(random.uniform(10, 15))
    
    # Analysis 2: Regional interest (OPTIONAL - comment out if rate limited)
    print("-"*60)
    
    choice = input("\nContinue with regional analysis? (y/n - may trigger rate limit): ").lower()
    
    if choice == 'y':
        regional_df = analyze_regional_interest(pytrends, 'Formula 1')
        
        if regional_df is not None:
            print("\n⏳ Waiting before next request...")
            time.sleep(random.uniform(10, 15))
    else:
        regional_df = None
        print("⏭️  Skipped regional analysis")
    
    # Analysis 3: Related queries (OPTIONAL)
    print("-"*60)
    
    choice = input("\nContinue with related queries? (y/n - may trigger rate limit): ").lower()
    
    if choice == 'y':
        related = analyze_related_queries(pytrends, 'Formula 1')
    else:
        related = None
        print("⏭️  Skipped related queries")
    
    # Summary
    print("\n" + "="*60)
    print("📊 ANALYSIS COMPLETE")
    print("="*60)
    
    files_created = []
    if interest_df is not None:
        files_created.append("f1_google_trends.png")
    if regional_df is not None:
        files_created.append("f1_regional_interest.csv")
    if related is not None:
        files_created.extend(["f1_top_queries.csv", "f1_rising_queries.csv"])
    
    if files_created:
        print("\n✅ Files created:")
        for file in files_created:
            print(f"   - {file}")
    
    print("\n💡 Tips to avoid rate limiting:")
    print("   1. Wait 10-15 seconds between queries")
    print("   2. Don't run script multiple times quickly")
    print("   3. Use VPN if blocked")
    print("   4. Limit to 1-2 queries per session")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
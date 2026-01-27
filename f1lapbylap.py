"""
F1 Lap-by-Lap Data Crawler
Crawl dữ liệu chi tiết từng vòng đua với tất cả thông số kỹ thuật
"""

import fastf1
import pandas as pd
import numpy as np
import warnings
import os
from datetime import timedelta

warnings.filterwarnings('ignore')

# ==================== SETUP ====================

CACHE_FOLDER = 'f1_cache'
OUTPUT_FOLDER = 'f1_lapbylap_data'

def setup():
    """Setup folders and cache"""
    for folder in [CACHE_FOLDER, OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    fastf1.Cache.enable_cache(CACHE_FOLDER)
    print("✅ Setup hoàn tất!\n")

# ==================== HELPER FUNCTIONS ====================

def timedelta_to_seconds(td):
    """Convert timedelta to seconds"""
    if pd.isna(td):
        return np.nan
    try:
        if isinstance(td, timedelta):
            return td.total_seconds()
        return float(td)
    except:
        return np.nan

def check_pit_stop(current_lap, next_lap):
    """Check if there was a pit stop"""
    if pd.isna(current_lap) or pd.isna(next_lap):
        return False
    try:
        # Pit stop nếu có PitInTime hoặc PitOutTime
        return pd.notna(current_lap) and (
            pd.notna(current_lap.get('PitInTime')) or 
            pd.notna(current_lap.get('PitOutTime'))
        )
    except:
        return False

# ==================== MAIN CRAWLER ====================

def crawl_lapbylap_data(year, round_number, session_type='R'):
    """
    Crawl dữ liệu chi tiết lap-by-lap
    
    Returns DataFrame với các cột:
    - Driver info
    - Lap number
    - Lap time & sector times
    - Speed metrics
    - Pit stop info
    - Tyre info
    - Position tracking
    - DRS usage
    """
    
    print(f"🔍 Crawling Lap-by-Lap Data")
    print(f"   Year: {year}, Round: {round_number}, Session: {session_type}")
    print("-" * 60)
    
    try:
        # Load session
        session = fastf1.get_session(year, round_number, session_type)
        session.load()
        
        event_name = session.event['EventName']
        print(f"📍 Event: {event_name}")
        print(f"📅 Date: {session.event['EventDate']}")
        
        # Get all laps
        laps = session.laps
        
        if laps is None or len(laps) == 0:
            print("❌ Không có dữ liệu laps")
            return None
        
        print(f"📊 Tổng số laps: {len(laps)}")
        
        # Prepare detailed lap data
        lap_details = []
        
        drivers = laps['Driver'].unique()
        print(f"👥 Số tay đua: {len(drivers)}\n")
        
        for driver in drivers:
            driver_laps = laps.pick_driver(driver)
            
            if len(driver_laps) == 0:
                continue
            
            print(f"  🏎️  {driver}... ", end="")
            
            for idx, lap in driver_laps.iterlaps():
                try:
                    # Basic lap info
                    lap_data = {
                        # Event & Driver Info
                        'Year': year,
                        'Round': round_number,
                        'EventName': event_name,
                        'Driver': driver,
                        'DriverNumber': lap['DriverNumber'],
                        'Team': lap['Team'],
                        
                        # Lap Number
                        'LapNumber': lap['LapNumber'],
                        'LapNumberStr': f"Lap {lap['LapNumber']}",
                        
                        # Lap Time & Duration
                        'LapTime': timedelta_to_seconds(lap['LapTime']),
                        'LapTimeStr': str(lap['LapTime']) if pd.notna(lap['LapTime']) else None,
                        
                        # Sector Times
                        'Sector1Time': timedelta_to_seconds(lap['Sector1Time']),
                        'Sector2Time': timedelta_to_seconds(lap['Sector2Time']),
                        'Sector3Time': timedelta_to_seconds(lap['Sector3Time']),
                        
                        'Sector1TimeStr': str(lap['Sector1Time']) if pd.notna(lap['Sector1Time']) else None,
                        'Sector2TimeStr': str(lap['Sector2Time']) if pd.notna(lap['Sector2Time']) else None,
                        'Sector3TimeStr': str(lap['Sector3Time']) if pd.notna(lap['Sector3Time']) else None,
                        
                        # Speed Metrics (km/h)
                        'SpeedI1': lap.get('SpeedI1', np.nan),  # Speed at intermediate 1
                        'SpeedI2': lap.get('SpeedI2', np.nan),  # Speed at intermediate 2
                        'SpeedFL': lap.get('SpeedFL', np.nan),  # Speed at finish line
                        'SpeedST': lap.get('SpeedST', np.nan),  # Speed at speed trap
                        
                        # Position
                        'Position': lap['Position'],
                        'PositionStr': f"P{lap['Position']}" if pd.notna(lap['Position']) else None,
                        
                        # Pit Stop Info
                        'PitInTime': lap.get('PitInTime'),
                        'PitOutTime': lap.get('PitOutTime'),
                        'IsPitLap': pd.notna(lap.get('PitInTime')) or pd.notna(lap.get('PitOutTime')),
                        
                        # Tyre Info
                        'Compound': lap['Compound'],
                        'TyreLife': lap['TyreLife'],
                        'Stint': lap['Stint'],
                        'IsNewTyre': lap['TyreLife'] == 1 if pd.notna(lap['TyreLife']) else False,
                        
                        # Track Status
                        'TrackStatus': lap.get('TrackStatus', np.nan),
                        'IsPersonalBest': lap['IsPersonalBest'] if 'IsPersonalBest' in lap else False,
                        
                        # Flags & Conditions
                        'IsAccurate': lap.get('IsAccurate', True),
                        'Deleted': lap.get('Deleted', False),
                        
                        # Time stamps
                        'Time': lap['Time'],
                        'LapStartTime': lap.get('LapStartTime'),
                    }
                    
                    # Calculate pit stop duration if available
                    if pd.notna(lap.get('PitInTime')) and pd.notna(lap.get('PitOutTime')):
                        pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
                        lap_data['PitDuration'] = pit_duration
                    else:
                        lap_data['PitDuration'] = np.nan
                    
                    lap_details.append(lap_data)
                    
                except Exception as e:
                    # Skip problematic laps
                    continue
            
            print(f"✅ ({len(driver_laps)} laps)")
        
        # Create DataFrame
        df = pd.DataFrame(lap_details)
        
        # Add position change calculation
        if len(df) > 0:
            df = df.sort_values(['Driver', 'LapNumber'])
            df['PositionChange'] = df.groupby('Driver')['Position'].diff() * -1  # -1 because lower is better
            df['CumulativePositionChange'] = df.groupby('Driver')['PositionChange'].cumsum()
        
        print(f"\n✅ Crawl hoàn tất: {len(df)} lap records")
        return df
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==================== TELEMETRY DATA CRAWLER ====================

def crawl_telemetry_per_lap(year, round_number, driver, session_type='R', max_laps=5):
    """
    Crawl dữ liệu telemetry chi tiết cho từng vòng đua
    Bao gồm: Speed, Throttle, Brake, DRS, Gear, RPM
    """
    
    print(f"\n📡 Crawling Telemetry for {driver}")
    print("-" * 60)
    
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load()
        
        driver_laps = session.laps.pick_driver(driver)
        
        if len(driver_laps) == 0:
            print(f"❌ Không có laps cho {driver}")
            return None
        
        # Lấy một số laps để phân tích (không lấy hết vì dữ liệu rất lớn)
        laps_to_analyze = driver_laps.head(max_laps)
        
        all_telemetry = []
        
        for idx, lap in laps_to_analyze.iterlaps():
            try:
                print(f"  Lap {lap['LapNumber']}... ", end="")
                
                # Get telemetry for this lap
                telemetry = lap.get_telemetry()
                
                if telemetry is not None and len(telemetry) > 0:
                    # Add lap info
                    telemetry['Driver'] = driver
                    telemetry['LapNumber'] = lap['LapNumber']
                    telemetry['Year'] = year
                    telemetry['Round'] = round_number
                    
                    # Calculate DRS activated percentage
                    drs_activated = telemetry['DRS'].sum() if 'DRS' in telemetry.columns else 0
                    drs_percentage = (drs_activated / len(telemetry) * 100) if len(telemetry) > 0 else 0
                    
                    telemetry['DRSActivatedPct'] = drs_percentage
                    
                    all_telemetry.append(telemetry)
                    print(f"✅ ({len(telemetry)} points)")
                else:
                    print("⚠️ No data")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)[:30]}")
                continue
        
        if all_telemetry:
            df = pd.concat(all_telemetry, ignore_index=True)
            print(f"\n✅ Telemetry: {len(df)} data points")
            return df
        else:
            print("\n❌ Không có telemetry data")
            return None
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None

# ==================== ENHANCED LAP ANALYSIS ====================

def analyze_lap_details(df):
    """Phân tích chi tiết dữ liệu lap-by-lap"""
    
    print("\n" + "="*60)
    print("📊 PHÂN TÍCH CHI TIẾT LAP-BY-LAP")
    print("="*60)
    
    if df is None or len(df) == 0:
        print("❌ Không có dữ liệu")
        return
    
    # 1. Tổng quan
    print(f"\n📈 TỔNG QUAN:")
    print(f"   - Tổng số laps: {len(df)}")
    print(f"   - Số tay đua: {df['Driver'].nunique()}")
    print(f"   - Số pit stops: {df['IsPitLap'].sum()}")
    print(f"   - Laps không hợp lệ: {df['Deleted'].sum()}")
    
    # 2. Sector times analysis
    print(f"\n⏱️  SECTOR TIMES (trung bình):")
    sector_avg = df[['Sector1Time', 'Sector2Time', 'Sector3Time']].mean()
    print(f"   - Sector 1: {sector_avg['Sector1Time']:.3f}s")
    print(f"   - Sector 2: {sector_avg['Sector2Time']:.3f}s")
    print(f"   - Sector 3: {sector_avg['Sector3Time']:.3f}s")
    
    # 3. Speed analysis
    print(f"\n🚀 TỐC ĐỘ TRUNG BÌNH:")
    speed_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
    for col in speed_cols:
        if col in df.columns:
            speed_avg = df[col].mean()
            speed_max = df[col].max()
            if not pd.isna(speed_avg):
                print(f"   - {col}: {speed_avg:.1f} km/h (max: {speed_max:.1f})")
    
    # 4. Tyre compound usage
    print(f"\n🏎️  SỬ DỤNG LỐP:")
    tyre_usage = df['Compound'].value_counts()
    for compound, count in tyre_usage.items():
        pct = count / len(df) * 100
        print(f"   - {compound}: {count} laps ({pct:.1f}%)")
    
    # 5. Position changes
    print(f"\n📊 THAY ĐỔI VỊ TRÍ:")
    position_changes = df.groupby('Driver')['CumulativePositionChange'].last().sort_values(ascending=False)
    print(f"   Top 5 tay đua vượt nhiều nhất:")
    for i, (driver, change) in enumerate(position_changes.head(5).items(), 1):
        print(f"   {i}. {driver}: {change:+.0f} vị trí")
    
    # 6. Pit stop analysis
    pit_laps = df[df['IsPitLap']]
    if len(pit_laps) > 0:
        print(f"\n🔧 PIT STOPS:")
        pit_duration_avg = pit_laps['PitDuration'].mean()
        if not pd.isna(pit_duration_avg):
            print(f"   - Thời gian pit trung bình: {pit_duration_avg:.2f}s")
        
        # Pit stop by driver
        pit_by_driver = pit_laps.groupby('Driver').size().sort_values(ascending=False)
        print(f"   - Số lần pit theo tay đua:")
        for driver, count in pit_by_driver.items():
            print(f"     • {driver}: {count} lần")

# ==================== EXPORT FUNCTIONS ====================

def export_to_csv(df, filename):
    """Export DataFrame to CSV"""
    try:
        filepath = f"{OUTPUT_FOLDER}/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu: {e}")
        return False

# ==================== MAIN EXECUTION ====================

def main():
    """Main function"""
    setup()
    
    # ==================== CẤU HÌNH ====================
    YEAR = 2024
    ROUND = 1  # Bahrain GP
    SESSION = 'R'  # Race
    
    print("⚙️  CẤU HÌNH:")
    print(f"   - Năm: {YEAR}")
    print(f"   - Round: {ROUND}")
    print(f"   - Session: {SESSION}")
    print(f"   - Output: {OUTPUT_FOLDER}/")
    print("="*60)
    
    # ==================== CRAWL LAP-BY-LAP DATA ====================
    print("\n" + "="*60)
    print("📥 PHASE 1: CRAWLING LAP-BY-LAP DATA")
    print("="*60)
    
    lap_data = crawl_lapbylap_data(YEAR, ROUND, SESSION)
    
    if lap_data is not None:
        # Save full dataset
        export_to_csv(lap_data, f'lapbylap_detailed_{YEAR}_R{ROUND}.csv')
        
        # Analyze
        analyze_lap_details(lap_data)
        
        # Save summary by driver
        driver_summary = lap_data.groupby('Driver').agg({
            'LapNumber': 'count',
            'LapTime': 'mean',
            'Position': 'last',
            'CumulativePositionChange': 'last',
            'IsPitLap': 'sum',
            'Compound': lambda x: x.mode()[0] if len(x.mode()) > 0 else None
        }).round(3)
        driver_summary.columns = ['TotalLaps', 'AvgLapTime', 'FinalPosition', 
                                  'PositionChange', 'PitStops', 'MainCompound']
        export_to_csv(driver_summary, f'driver_summary_{YEAR}_R{ROUND}.csv')
    
    # ==================== CRAWL TELEMETRY (OPTIONAL) ====================
    print("\n" + "="*60)
    print("📥 PHASE 2: CRAWLING TELEMETRY DATA (Optional)")
    print("="*60)
    print("⚠️  Telemetry data rất lớn, chỉ lấy mẫu 5 laps của 1 tay đua")
    
    # Example: Get telemetry for top driver
    if lap_data is not None:
        top_driver = lap_data.groupby('Driver')['Position'].last().idxmin()
        print(f"📍 Chọn tay đua: {top_driver}")
        
        telemetry_data = crawl_telemetry_per_lap(YEAR, ROUND, top_driver, SESSION, max_laps=5)
        
        if telemetry_data is not None:
            export_to_csv(telemetry_data, f'telemetry_sample_{top_driver}_{YEAR}_R{ROUND}.csv')
    
    # ==================== DONE ====================
    print("\n" + "="*60)
    print("✅ HOÀN TẤT!")
    print(f"📁 Tất cả files đã lưu tại: {OUTPUT_FOLDER}/")
    print("="*60)
    
    # Print file summary
    print("\n📋 CÁC FILE ĐÃ TẠO:")
    if os.path.exists(OUTPUT_FOLDER):
        for file in os.listdir(OUTPUT_FOLDER):
            filepath = os.path.join(OUTPUT_FOLDER, file)
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"   - {file} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
"""
F1 Advanced Data Crawler
Crawl các dữ liệu bổ sung: Telemetry, Weather, Pit Stops, Track Info
"""

import fastf1
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== SETUP ====================

CACHE_FOLDER = 'f1_cache'
OUTPUT_FOLDER = 'f1_advanced_data'

def setup():
    """Setup folders"""
    for folder in [CACHE_FOLDER, OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    fastf1.Cache.enable_cache(CACHE_FOLDER)
    print("✅ Setup hoàn tất!\n")

# ==================== 1. TELEMETRY DATA ====================

def crawl_telemetry_data(year, round_number, session_type='R', drivers=None):
    """
    Crawl dữ liệu telemetry chi tiết
    
    Returns: DataFrame với các cột:
    - Time, Distance, Speed, nGear, Throttle, Brake, DRS, RPM
    """
    print(f"📡 Crawling Telemetry - {year} Round {round_number} {session_type}...")
    
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load()
        
        if drivers is None:
            drivers = session.drivers[:5]  # Top 5 tay đua
        
        all_telemetry = []
        
        for driver in drivers:
            try:
                # Lấy vòng nhanh nhất
                driver_laps = session.laps.pick_driver(driver)
                fastest_lap = driver_laps.pick_fastest()
                
                if fastest_lap is not None:
                    telemetry = fastest_lap.get_telemetry()
                    telemetry['Driver'] = driver
                    telemetry['Year'] = year
                    telemetry['Round'] = round_number
                    telemetry['LapNumber'] = fastest_lap['LapNumber']
                    
                    all_telemetry.append(telemetry[['Driver', 'Year', 'Round', 'LapNumber',
                                                    'Time', 'Distance', 'Speed', 'nGear', 
                                                    'Throttle', 'Brake', 'DRS', 'RPM']])
                    print(f"  ✅ {driver}")
            except Exception as e:
                print(f"  ⚠️ {driver}: {str(e)[:50]}")
        
        if all_telemetry:
            result = pd.concat(all_telemetry, ignore_index=True)
            print(f"  📊 Đã lấy {len(result)} data points")
            return result
        else:
            print(f"  ❌ Không có dữ liệu")
            return None
            
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return None

# ==================== 2. WEATHER DATA ====================

def crawl_weather_data(year, round_number, session_type='R'):
    """
    Crawl dữ liệu thời tiết
    
    Returns: DataFrame với các cột:
    - Time, AirTemp, TrackTemp, Humidity, Pressure, WindSpeed, Rainfall
    """
    print(f"🌤️ Crawling Weather - {year} Round {round_number} {session_type}...")
    
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load()
        
        weather = session.weather_data
        
        if weather is not None and len(weather) > 0:
            weather['Year'] = year
            weather['Round'] = round_number
            weather['SessionType'] = session_type
            
            # Chọn các cột quan trọng
            cols = ['Year', 'Round', 'SessionType', 'Time', 'AirTemp', 'TrackTemp', 
                   'Humidity', 'Pressure', 'WindSpeed', 'Rainfall']
            available_cols = [col for col in cols if col in weather.columns]
            
            result = weather[available_cols]
            print(f"  ✅ Đã lấy {len(result)} weather records")
            return result
        else:
            print(f"  ⚠️ Không có dữ liệu thời tiết")
            return None
            
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return None

# ==================== 3. PIT STOP DATA ====================

def crawl_pitstop_data(year, round_number):
    """
    Crawl dữ liệu pit stop
    
    Returns: DataFrame với thông tin pit stop của từng tay đua
    """
    print(f"🔧 Crawling Pit Stops - {year} Round {round_number}...")
    
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load()
        
        all_pitstops = []
        
        for driver in session.drivers:
            try:
                driver_laps = session.laps.pick_driver(driver)
                
                # Tìm các vòng có pit
                pit_laps = driver_laps[driver_laps['PitInTime'].notna()]
                
                if len(pit_laps) > 0:
                    for idx, lap in pit_laps.iterrows():
                        # Tính thời gian pit
                        if pd.notna(lap['PitInTime']) and pd.notna(lap['PitOutTime']):
                            pit_duration = (lap['PitOutTime'] - lap['PitInTime']).total_seconds()
                        else:
                            pit_duration = None
                        
                        all_pitstops.append({
                            'Year': year,
                            'Round': round_number,
                            'Driver': driver,
                            'LapNumber': lap['LapNumber'],
                            'PitInTime': lap['PitInTime'],
                            'PitOutTime': lap['PitOutTime'],
                            'PitDuration': pit_duration,
                            'TyreCompound': lap['Compound']
                        })
            except Exception as e:
                continue
        
        if all_pitstops:
            result = pd.DataFrame(all_pitstops)
            print(f"  ✅ Đã lấy {len(result)} pit stops")
            return result
        else:
            print(f"  ⚠️ Không có dữ liệu pit stop")
            return None
            
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return None

# ==================== 4. RACE CONTROL MESSAGES ====================

def crawl_race_control_messages(year, round_number):
    """
    Crawl race control messages (penalties, safety car, etc.)
    """
    print(f"📢 Crawling Race Control - {year} Round {round_number}...")
    
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load()
        
        messages = session.race_control_messages
        
        if messages is not None and len(messages) > 0:
            messages['Year'] = year
            messages['Round'] = round_number
            
            print(f"  ✅ Đã lấy {len(messages)} messages")
            return messages
        else:
            print(f"  ⚠️ Không có messages")
            return None
            
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return None

# ==================== 5. DRIVER STANDINGS ====================

def crawl_driver_standings_ergast(year):
    """
    Crawl driver standings từ Ergast API
    """
    print(f"🏆 Crawling Driver Standings - {year}...")
    
    try:
        import requests
        
        url = f'http://ergast.com/api/f1/{year}/driverStandings.json'
        response = requests.get(url)
        data = response.json()
        
        standings_list = data['MRData']['StandingsTable']['StandingsLists']
        
        all_standings = []
        
        for standing in standings_list:
            round_num = standing['round']
            
            for driver_standing in standing['DriverStandings']:
                all_standings.append({
                    'Year': year,
                    'Round': int(round_num),
                    'Position': int(driver_standing['position']),
                    'Points': float(driver_standing['points']),
                    'Wins': int(driver_standing['wins']),
                    'DriverId': driver_standing['Driver']['driverId'],
                    'DriverName': f"{driver_standing['Driver']['givenName']} {driver_standing['Driver']['familyName']}",
                    'Constructor': driver_standing['Constructors'][0]['name']
                })
        
        result = pd.DataFrame(all_standings)
        print(f"  ✅ Đã lấy {len(result)} records")
        return result
        
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return None

# ==================== 6. CONSTRUCTOR STANDINGS ====================

def crawl_constructor_standings_ergast(year):
    """
    Crawl constructor standings từ Ergast API
    """
    print(f"🏗️ Crawling Constructor Standings - {year}...")
    
    try:
        import requests
        
        url = f'http://ergast.com/api/f1/{year}/constructorStandings.json'
        response = requests.get(url)
        data = response.json()
        
        standings_list = data['MRData']['StandingsTable']['StandingsLists']
        
        all_standings = []
        
        for standing in standings_list:
            round_num = standing['round']
            
            for constructor_standing in standing['ConstructorStandings']:
                all_standings.append({
                    'Year': year,
                    'Round': int(round_num),
                    'Position': int(constructor_standing['position']),
                    'Points': float(constructor_standing['points']),
                    'Wins': int(constructor_standing['wins']),
                    'ConstructorId': constructor_standing['Constructor']['constructorId'],
                    'ConstructorName': constructor_standing['Constructor']['name']
                })
        
        result = pd.DataFrame(all_standings)
        print(f"  ✅ Đã lấy {len(result)} records")
        return result
        
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return None

# ==================== MAIN CRAWL FUNCTION ====================

def crawl_all_advanced_data(year, max_rounds=2):
    """
    Crawl tất cả dữ liệu nâng cao cho một mùa giải
    """
    print(f"{'='*60}")
    print(f"🏁 CRAWL DỮ LIỆU NÂNG CAO - MÙA GIẢI {year}")
    print(f"{'='*60}\n")
    
    # Get schedule
    schedule = fastf1.get_event_schedule(year)
    rounds = schedule['RoundNumber'].values[:max_rounds]
    
    # Storage
    all_telemetry = []
    all_weather = []
    all_pitstops = []
    all_messages = []
    
    # Crawl từng round
    for round_num in rounds:
        event = schedule[schedule['RoundNumber'] == round_num].iloc[0]
        print(f"\n📍 Round {round_num}: {event['EventName']}")
        print("-" * 60)
        
        # 1. Telemetry
        telemetry = crawl_telemetry_data(year, round_num, 'R')
        if telemetry is not None:
            all_telemetry.append(telemetry)
        
        # 2. Weather
        weather = crawl_weather_data(year, round_num, 'R')
        if weather is not None:
            all_weather.append(weather)
        
        # 3. Pit stops
        pitstops = crawl_pitstop_data(year, round_num)
        if pitstops is not None:
            all_pitstops.append(pitstops)
        
        # 4. Race control messages
        messages = crawl_race_control_messages(year, round_num)
        if messages is not None:
            all_messages.append(messages)
    
    # 5. Standings (toàn mùa)
    print(f"\n{'='*60}")
    driver_standings = crawl_driver_standings_ergast(year)
    constructor_standings = crawl_constructor_standings_ergast(year)
    
    # Save data
    print(f"\n{'='*60}")
    print("💾 Đang lưu dữ liệu...")
    print(f"{'='*60}\n")
    
    if all_telemetry:
        df = pd.concat(all_telemetry, ignore_index=True)
        filename = f'{OUTPUT_FOLDER}/telemetry_{year}.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Telemetry: {filename} ({len(df)} records)")
    
    if all_weather:
        df = pd.concat(all_weather, ignore_index=True)
        filename = f'{OUTPUT_FOLDER}/weather_{year}.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Weather: {filename} ({len(df)} records)")
    
    if all_pitstops:
        df = pd.concat(all_pitstops, ignore_index=True)
        filename = f'{OUTPUT_FOLDER}/pitstops_{year}.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Pit Stops: {filename} ({len(df)} records)")
    
    if all_messages:
        df = pd.concat(all_messages, ignore_index=True)
        filename = f'{OUTPUT_FOLDER}/race_control_{year}.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Race Control: {filename} ({len(df)} records)")
    
    if driver_standings is not None:
        filename = f'{OUTPUT_FOLDER}/driver_standings_{year}.csv'
        driver_standings.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Driver Standings: {filename} ({len(driver_standings)} records)")
    
    if constructor_standings is not None:
        filename = f'{OUTPUT_FOLDER}/constructor_standings_{year}.csv'
        constructor_standings.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ Constructor Standings: {filename} ({len(constructor_standings)} records)")
    
    print(f"\n🎉 Hoàn tất! Dữ liệu đã lưu tại: {OUTPUT_FOLDER}/")

# ==================== MAIN ====================

if __name__ == "__main__":
    setup()
    
    # Cấu hình
    YEAR = 2024
    MAX_ROUNDS = 2  # Thay đổi để crawl nhiều hơn
    
    print(f"⚙️ Cấu hình:")
    print(f"   - Năm: {YEAR}")
    print(f"   - Số round: {MAX_ROUNDS}")
    print(f"   - Output: {OUTPUT_FOLDER}/\n")
    
    # Crawl
    crawl_all_advanced_data(YEAR, MAX_ROUNDS)
    
    print(f"\n{'='*60}")
    print("✅ HOÀN TẤT TẤT CẢ!")
    print(f"{'='*60}")
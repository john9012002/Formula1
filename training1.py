"""
F1 Data Crawler - Standalone Script
Crawl dữ liệu F1 từ các mùa giải 2023, 2024, 2025
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
from tqdm import tqdm  # Progress bar (optional)

# Cấu hình
warnings.filterwarnings('ignore')

# ==================== CẤU HÌNH CƠ BẢN ====================
SEASONS = [2023, 2024, 2025]  # Các mùa giải cần crawl
MAX_ROUNDS = 2  # Số chặng tối đa mỗi mùa (None = tất cả)
SESSION_TYPES = ['FP1', 'FP2', 'FP3', 'Q', 'S', 'R']  # Các phiên đua
OUTPUT_FOLDER = 'f1_data_output'  # Folder lưu kết quả
CACHE_FOLDER = 'f1_cache'  # Folder cache

# Tên đầy đủ của các phiên
SESSION_NAMES = {
    'FP1': 'Practice 1',
    'FP2': 'Practice 2', 
    'FP3': 'Practice 3',
    'Q': 'Qualifying',
    'S': 'Sprint',
    'R': 'Race'
}

# ==================== SETUP ====================

def setup_environment():
    """Tạo các folder cần thiết và enable cache"""
    # Tạo folder cache
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)
        print(f"📁 Đã tạo folder cache: {CACHE_FOLDER}")
    
    # Enable cache
    fastf1.Cache.enable_cache(CACHE_FOLDER)
    print(f"✅ Cache đã được kích hoạt")
    
    # Tạo folder output
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 Đã tạo folder output: {OUTPUT_FOLDER}")
    
    print(f"FastF1 version: {fastf1.__version__}\n")

# ==================== FUNCTIONS ====================

def get_season_schedule(year):
    """Lấy lịch thi đấu của một mùa giải"""
    try:
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        print(f"❌ Lỗi khi lấy lịch {year}: {e}")
        return None

def get_session_results(year, round_number, session_name):
    """Lấy kết quả của một phiên đua cụ thể"""
    try:
        session = fastf1.get_session(year, round_number, session_name)
        session.load()
        
        results = session.results
        
        # Thêm thông tin bổ sung
        results['Year'] = year
        results['Round'] = round_number
        results['SessionType'] = session_name
        results['EventName'] = session.event['EventName']
        results['Country'] = session.event['Country']
        
        return results
    except Exception as e:
        # print(f"      ⚠️ Không thể load {session_name}: {str(e)[:50]}...")
        return None

def get_lap_times(year, round_number, session_name):
    """Lấy thời gian vòng đua chi tiết"""
    try:
        session = fastf1.get_session(year, round_number, session_name)
        session.load()
        
        laps = session.laps
        
        # Chọn các cột quan trọng
        lap_data = laps[['Driver', 'DriverNumber', 'LapTime', 'LapNumber', 
                         'Stint', 'Compound', 'TyreLife', 'Team']].copy()
        
        lap_data['Year'] = year
        lap_data['Round'] = round_number
        lap_data['SessionType'] = session_name
        
        return lap_data
    except Exception as e:
        return None

def crawl_all_data():
    """Crawl toàn bộ dữ liệu"""
    all_results = []
    all_laps = []
    all_schedules = {}
    
    print(f"{'='*60}")
    print(f"🏁 BẮT ĐẦU CRAWL DỮ LIỆU F1")
    print(f"{'='*60}\n")
    
    # Lấy lịch thi đấu
    for year in SEASONS:
        print(f"📅 Đang lấy lịch mùa giải {year}...", end=" ")
        schedule = get_season_schedule(year)
        if schedule is not None:
            all_schedules[year] = schedule
            print(f"✅ ({len(schedule)} chặng)")
        else:
            print("❌")
    
    print()
    
    # Crawl dữ liệu từng mùa giải
    for year in SEASONS:
        if year not in all_schedules:
            continue
        
        print(f"{'='*60}")
        print(f"🏁 MÙA GIẢI {year}")
        print(f"{'='*60}")
        
        schedule = all_schedules[year]
        
        # Lấy số chặng đua
        if MAX_ROUNDS is None:
            rounds = schedule['RoundNumber'].values
        else:
            rounds = schedule['RoundNumber'].values[:MAX_ROUNDS]
        
        total_rounds = len(rounds)
        
        for idx, round_num in enumerate(rounds, 1):
            event_info = schedule[schedule['RoundNumber'] == round_num].iloc[0]
            print(f"\n📍 [{idx}/{total_rounds}] {event_info['EventName']} ({event_info['Country']})")
            
            for session_type in SESSION_TYPES:
                session_name = SESSION_NAMES[session_type]
                print(f"    🔄 {session_name:15s}", end=" ")
                
                # Lấy kết quả
                results = get_session_results(year, round_num, session_type)
                if results is not None:
                    all_results.append(results)
                    print("✅")
                else:
                    print("⏭️")
                
                # Lấy lap times (chỉ cho Race và Qualifying)
                if session_type in ['R', 'Q'] and results is not None:
                    laps = get_lap_times(year, round_num, session_type)
                    if laps is not None:
                        all_laps.append(laps)
    
    print(f"\n{'='*60}")
    print(f"✅ HOÀN TẤT CRAWL DỮ LIỆU!")
    print(f"{'='*60}")
    print(f"📊 Tổng số phiên đua: {len(all_results)}")
    print(f"⏱️ Tổng số lap times: {len(all_laps)}")
    
    return all_results, all_laps, all_schedules

def save_data(all_results, all_laps, all_schedules):
    """Lưu dữ liệu ra file CSV"""
    print(f"\n💾 Đang lưu dữ liệu...")
    
    # Lưu kết quả phiên đua
    if all_results:
        df_results = pd.concat(all_results, ignore_index=True)
        
        # Chọn các cột quan trọng
        important_cols = ['Year', 'Round', 'EventName', 'Country', 'SessionType', 
                         'DriverNumber', 'BroadcastName', 'Abbreviation', 'TeamName', 
                         'Position', 'GridPosition', 'Status', 'Points']
        
        available_cols = [col for col in important_cols if col in df_results.columns]
        df_results_clean = df_results[available_cols].copy()
        
        # Lưu file tổng hợp
        results_file = f'{OUTPUT_FOLDER}/f1_session_results_2023_2025.csv'
        df_results_clean.to_csv(results_file, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu: {results_file}")
        
        # Lưu theo từng năm
        for year in df_results['Year'].unique():
            year_data = df_results_clean[df_results_clean['Year'] == year]
            year_file = f'{OUTPUT_FOLDER}/f1_results_{int(year)}.csv'
            year_data.to_csv(year_file, index=False, encoding='utf-8-sig')
            print(f"✅ Đã lưu: {year_file}")
    
    # Lưu lap times
    if all_laps:
        df_laps = pd.concat(all_laps, ignore_index=True)
        laps_file = f'{OUTPUT_FOLDER}/f1_lap_times_2023_2025.csv'
        df_laps.to_csv(laps_file, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu: {laps_file}")
    
    # Lưu lịch thi đấu
    for year, schedule in all_schedules.items():
        schedule_file = f'{OUTPUT_FOLDER}/f1_schedule_{year}.csv'
        schedule.to_csv(schedule_file, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu: {schedule_file}")
    
    print(f"\n🎉 Tất cả dữ liệu đã được lưu vào: {OUTPUT_FOLDER}/")
    
    return df_results_clean if all_results else None

def show_statistics(df_results):
    """Hiển thị thống kê cơ bản"""
    if df_results is None or len(df_results) == 0:
        return
    
    print(f"\n{'='*60}")
    print(f"📈 THỐNG KÊ TỔNG QUAN")
    print(f"{'='*60}\n")
    
    # Thống kê theo năm
    print("1. Số phiên đua theo năm:")
    print(df_results.groupby('Year')['SessionType'].count())
    
    # Thống kê theo loại phiên
    print("\n2. Số lượng theo loại phiên đua:")
    print(df_results['SessionType'].value_counts())
    
    # Top tay đua
    if 'Points' in df_results.columns:
        race_results = df_results[df_results['SessionType'] == 'R'].copy()
        if len(race_results) > 0:
            top_drivers = race_results.groupby('BroadcastName')['Points'].sum().sort_values(ascending=False).head(10)
            print("\n3. Top 10 tay đua có nhiều điểm nhất:")
            for idx, (driver, points) in enumerate(top_drivers.items(), 1):
                print(f"   {idx:2d}. {driver:20s} - {points:.0f} điểm")
    
    # Thống kê theo đội
    if 'TeamName' in df_results.columns:
        print("\n4. Số lần xuất hiện của các đội:")
        team_counts = df_results['TeamName'].value_counts().head(5)
        for team, count in team_counts.items():
            print(f"   - {team:30s}: {count} lần")

# ==================== MAIN ====================

def main():
    """Hàm main"""
    print("🏎️  F1 DATA CRAWLER")
    print("="*60)
    print(f"Mùa giải: {SEASONS}")
    print(f"Số chặng tối đa/mùa: {MAX_ROUNDS if MAX_ROUNDS else 'TẤT CẢ'}")
    print(f"Các phiên: {', '.join(SESSION_TYPES)}")
    print("="*60 + "\n")
    
    # Setup môi trường
    setup_environment()
    
    # Crawl dữ liệu
    all_results, all_laps, all_schedules = crawl_all_data()
    
    # Lưu dữ liệu
    df_results = save_data(all_results, all_laps, all_schedules)
    
    # Hiển thị thống kê
    show_statistics(df_results)
    
    print(f"\n{'='*60}")
    print("🏁 HOÀN TẤT!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
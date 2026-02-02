"""
Analyze F1 YouTube content performance
"""

from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime

def setup_youtube_api():
    """Setup YouTube Data API v3"""
    
    API_KEY = "AIzaSyC-T_KnYCClxcXKIu4HGUBvptKHUeBfEHg"
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    return youtube

def get_channel_statistics(youtube, channel_id="UCd8iY-kEHtaB8qt8MH--zGw"):
    """
    Get F1 Official Channel statistics
    Channel ID: UCd8iY-kEHtaB8qt8MH--zGw
    """
    
    request = youtube.channels().list(
        part='statistics,snippet',
        id=channel_id
    )
    
    response = request.execute()
    
    if response['items']:
        channel = response['items'][0]
        stats = channel['statistics']
        
        data = {
            'Channel': channel['snippet']['title'],
            'Subscribers': int(stats['subscriberCount']),
            'Total_Views': int(stats['viewCount']),
            'Total_Videos': int(stats['videoCount'])
        }
        
        print(f"📺 {data['Channel']}")
        print(f"   Subscribers: {data['Subscribers']:,}")
        print(f"   Total views: {data['Total_Views']:,}")
        print(f"   Videos: {data['Total_Videos']:,}")
        
        return data

def get_recent_videos(youtube, channel_id, max_results=50):
    """Get recent videos from F1 channel"""
    
    # Search for videos
    request = youtube.search().list(
        part='snippet',
        channelId=channel_id,
        maxResults=max_results,
        order='date',
        type='video'
    )
    
    response = request.execute()
    
    video_ids = [item['id']['videoId'] for item in response['items']]
    
    # Get video statistics
    stats_request = youtube.videos().list(
        part='statistics,snippet,contentDetails',
        id=','.join(video_ids)
    )
    
    stats_response = stats_request.execute()
    
    videos_data = []
    
    for video in stats_response['items']:
        videos_data.append({
            'video_id': video['id'],
            'title': video['snippet']['title'],
            'published_at': video['snippet']['publishedAt'],
            'views': int(video['statistics'].get('viewCount', 0)),
            'likes': int(video['statistics'].get('likeCount', 0)),
            'comments': int(video['statistics'].get('commentCount', 0)),
            'duration': video['contentDetails']['duration']
        })
    
    return pd.DataFrame(videos_data)

def analyze_video_performance(df):
    """Analyze video performance"""
    
    # Calculate engagement rate
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'] * 100
    
    # Average metrics
    print(f"\n📊 Video Performance Metrics:")
    print(f"   Avg views: {df['views'].mean():,.0f}")
    print(f"   Avg likes: {df['likes'].mean():,.0f}")
    print(f"   Avg comments: {df['comments'].mean():,.0f}")
    print(f"   Avg engagement rate: {df['engagement_rate'].mean():.2f}%")
    
    # Top performers
    print(f"\n🔥 Top 5 Videos by Views:")
    top_videos = df.nlargest(5, 'views')
    for idx, video in top_videos.iterrows():
        print(f"   {video['views']:,} views - {video['title'][:60]}...")
    
    return df

# Example usage
youtube = setup_youtube_api()
channel_stats = get_channel_statistics(youtube)
videos_df = get_recent_videos(youtube, "UCd8iY-kEHtaB8qt8MH--zGw", max_results=50)
videos_df = analyze_video_performance(videos_df)
videos_df.to_csv('f1_youtube_analysis.csv', index=False)
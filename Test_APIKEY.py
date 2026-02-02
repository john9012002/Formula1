"""
F1 YouTube Channel Finder - FIXED VERSION
Tự động tìm F1 Official Channel
"""

from googleapiclient.discovery import build
import json

def find_f1_channel(api_key):
    """Tự động tìm F1 Official Channel"""
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    print("🔍 Searching for Formula 1 channel...\n")
    
    try:
        search_request = youtube.search().list(
            part='snippet',
            q='Formula 1 official',
            type='channel',
            maxResults=10
        )
        
        search_response = search_request.execute()
        
        if search_response.get('items'):
            print("📺 Found channels:\n")
            
            f1_official_id = None
            
            for idx, item in enumerate(search_response['items'], 1):
                channel_id = item['snippet']['channelId']
                channel_title = item['snippet']['title']
                description = item['snippet']['description'][:80]
                
                print(f"{idx}. {channel_title}")
                print(f"   ID: {channel_id}")
                print(f"   Description: {description}...")
                
                # Get detailed info
                try:
                    channel_request = youtube.channels().list(
                        part='statistics',
                        id=channel_id
                    )
                    channel_response = channel_request.execute()
                    
                    if channel_response.get('items'):
                        stats = channel_response['items'][0]['statistics']
                        
                        # Format numbers properly
                        subs = stats.get('subscriberCount', '0')
                        videos = stats.get('videoCount', '0')
                        views = stats.get('viewCount', '0')
                        
                        try:
                            subs_int = int(subs)
                            videos_int = int(videos)
                            views_int = int(views)
                            
                            print(f"   📊 Subscribers: {subs_int:,}")
                            print(f"   📹 Videos: {videos_int:,}")
                            print(f"   👁️  Total Views: {views_int:,}")
                            
                            # Check if this is the official F1 channel
                            if subs_int > 5_000_000 and 'FORMULA 1' in channel_title.upper():
                                print(f"   ⭐ THIS IS THE OFFICIAL F1 CHANNEL!")
                                f1_official_id = channel_id
                            
                        except ValueError:
                            print(f"   📊 Subscribers: {subs}")
                            print(f"   📹 Videos: {videos}")
                    
                except Exception as e:
                    print(f"   ⚠️ Could not get stats: {e}")
                
                print()
            
            return f1_official_id
        
        else:
            print("❌ No channels found")
            return None
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return None

def get_channel_details(api_key, channel_id):
    """Get detailed info về channel"""
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        request = youtube.channels().list(
            part='snippet,statistics,contentDetails',
            id=channel_id
        )
        
        response = request.execute()
        
        if response.get('items'):
            channel = response['items'][0]
            
            print("\n" + "="*60)
            print("📺 CHANNEL DETAILS")
            print("="*60)
            
            # Snippet
            snippet = channel['snippet']
            print(f"\n📌 Basic Info:")
            print(f"   Channel ID: {channel['id']}")
            print(f"   Title: {snippet['title']}")
            print(f"   Description: {snippet['description'][:200]}...")
            print(f"   Published: {snippet['publishedAt']}")
            print(f"   Country: {snippet.get('country', 'N/A')}")
            
            # Statistics
            stats = channel['statistics']
            print(f"\n📊 Statistics:")
            
            try:
                subs = int(stats.get('subscriberCount', 0))
                videos = int(stats.get('videoCount', 0))
                views = int(stats.get('viewCount', 0))
                
                print(f"   Subscribers: {subs:,}")
                print(f"   Videos: {videos:,}")
                print(f"   Total Views: {views:,}")
                print(f"   Avg Views per Video: {views//videos:,}")
                
            except (ValueError, ZeroDivisionError):
                print(f"   Subscribers: {stats.get('subscriberCount', 'Hidden')}")
                print(f"   Videos: {stats.get('videoCount', 'N/A')}")
                print(f"   Total Views: {stats.get('viewCount', 'N/A')}")
            
            # Content Details
            content = channel.get('contentDetails', {})
            if 'relatedPlaylists' in content:
                uploads_playlist = content['relatedPlaylists'].get('uploads')
                print(f"\n📁 Content:")
                print(f"   Uploads Playlist ID: {uploads_playlist}")
            
            print("\n" + "="*60)
            
            return channel
        
        else:
            print("❌ Channel not found")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def get_recent_videos(api_key, channel_id, max_results=10):
    """Lấy videos gần đây từ channel"""
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        # Get uploads playlist
        channel_request = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        )
        
        channel_response = channel_request.execute()
        
        if not channel_response.get('items'):
            print("❌ Channel not found")
            return None
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Get videos from uploads playlist
        playlist_request = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_playlist_id,
            maxResults=max_results
        )
        
        playlist_response = playlist_request.execute()
        
        if playlist_response.get('items'):
            print(f"\n📹 RECENT VIDEOS (Top {max_results}):")
            print("="*60)
            
            video_ids = []
            
            for idx, item in enumerate(playlist_response['items'], 1):
                video_id = item['snippet']['resourceId']['videoId']
                title = item['snippet']['title']
                published = item['snippet']['publishedAt'][:10]
                
                video_ids.append(video_id)
                
                print(f"\n{idx}. {title}")
                print(f"   Video ID: {video_id}")
                print(f"   Published: {published}")
            
            # Get video statistics
            print(f"\n📊 Getting video statistics...")
            
            videos_request = youtube.videos().list(
                part='statistics',
                id=','.join(video_ids)
            )
            
            videos_response = videos_request.execute()
            
            if videos_response.get('items'):
                print()
                for idx, video in enumerate(videos_response['items'], 1):
                    stats = video['statistics']
                    
                    try:
                        views = int(stats.get('viewCount', 0))
                        likes = int(stats.get('likeCount', 0))
                        comments = int(stats.get('commentCount', 0))
                        
                        print(f"{idx}. Views: {views:,} | Likes: {likes:,} | Comments: {comments:,}")
                        
                    except ValueError:
                        print(f"{idx}. Views: {stats.get('viewCount', 'N/A')}")
            
            print("\n" + "="*60)
            
            return playlist_response['items']
        
        else:
            print("❌ No videos found")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    
    print("="*60)
    print("🏎️  F1 YOUTUBE CHANNEL ANALYZER")
    print("="*60)
    
    # Get API key
    api_key = input("\nEnter your YouTube API Key: ").strip()
    
    if not api_key:
        print("❌ API key required!")
        return
    
    # Find F1 channel
    channel_id = find_f1_channel(api_key)
    
    if channel_id:
        print(f"\n✅ F1 Official Channel Found!")
        print(f"Channel ID: {channel_id}")
        
        # Get detailed info
        channel = get_channel_details(api_key, channel_id)
        
        # Get recent videos
        if channel:
            print("\n" + "="*60)
            choice = input("\nGet recent videos? (y/n): ").lower().strip()
            
            if choice == 'y':
                videos = get_recent_videos(api_key, channel_id, max_results=10)
                
                if videos:
                    # Save results
                    save = input("\nSave results to file? (y/n): ").lower().strip()
                    
                    if save == 'y':
                        import json
                        
                        data = {
                            'channel_id': channel_id,
                            'channel_info': channel,
                            'recent_videos': videos
                        }
                        
                        with open('f1_youtube_data.json', 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        
                        print("✅ Saved to: f1_youtube_data.json")
        
        # Show test URLs
        print("\n" + "="*60)
        print("🔗 TEST URLs:")
        print("="*60)
        print(f"\nChannel details:")
        print(f"https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&id={channel_id}&key={api_key}")
        
        print(f"\nRecent uploads:")
        print(f"https://www.googleapis.com/youtube/v3/search?part=snippet&channelId={channel_id}&order=date&maxResults=10&key={api_key}")
        
    else:
        print("\n⚠️ Could not automatically find F1 channel")
        print("\nManual options:")
        print("1. Use Channel ID from search results above")
        print("2. Try: UCB_qr75-ydFVKSF9Dmo6izg (from search results)")
        
        manual_id = input("\nEnter Channel ID manually (or press Enter to skip): ").strip()
        
        if manual_id:
            get_channel_details(api_key, manual_id)
            get_recent_videos(api_key, manual_id, max_results=5)

if __name__ == "__main__":
    main()
import requests
import os
import re
import time
from urllib.parse import urljoin, urlparse
import random

def safe_request(url, headers=None, timeout=30, max_retries=5, min_wait=60, max_wait=300):
    """Make a GET request, handling 429 errors by sleeping and retrying."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait = random.uniform(min_wait, max_wait) * (attempt + 1)
                print(f"Got 429 Too Many Requests for {url}, sleeping for {int(wait)} seconds...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            print(f"Request error for {url}: {e}")
            time.sleep(random.uniform(min_wait, max_wait))
    print(f"Giving up on {url} after {max_retries} retries.")
    return None

def download_pgn_direct(game_id):
    """Try to download PGN directly using the game ID"""
    base_url = "https://www.chessgames.com"
    pgn_url = f"{base_url}/perl/downloadGamePGN?gid={game_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    response = safe_request(pgn_url, headers=headers)
    if response is None:
        print(f"    Direct download failed for game {game_id}: too many retries or error.")
        return None
    content = response.text
    if content.strip() and len(content) > 50:
        return content
    else:
        return None

def download_pgn_from_game_page(game_id):
    """Fetch the game page, parse for the PGN download link, and download the PGN."""
    base_url = "https://www.chessgames.com"
    game_url = f"{base_url}/perl/chessgame?gid={game_id}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    resp = safe_request(game_url, headers=headers)
    if resp is None:
        print(f"    Fallback: could not fetch game page for {game_id}")
        return None
    html = resp.text
    # Find the PGN download link
    match = re.search(r'<a href="([^"]*downloadGamePGN[^"]*\.pgn[^"]*)"[^>]*>download<', html, re.IGNORECASE)
    if not match:
        print(f"    No PGN download link found in game page for {game_id}")
        return None
    pgn_link = match.group(1)
    if not pgn_link.startswith("http"):
        pgn_link = urljoin(base_url, pgn_link)
    pgn_resp = safe_request(pgn_link, headers=headers)
    if pgn_resp is None:
        print(f"    Fallback PGN download failed for game {game_id}: too many retries or error.")
        return None
    content = pgn_resp.text
    if content.strip() and len(content) > 50:
        print(f"    Fallback PGN download succeeded for game {game_id}")
        return content
    else:
        print(f"    Fallback PGN download returned empty or invalid content for game {game_id}")
        return None

def extract_game_ids_from_tournament(tournament_url):
    """Extract all game IDs from a tournament page"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    response = safe_request(tournament_url, headers=headers)
    if response is None:
        print(f"Error extracting games from tournament {tournament_url}: too many retries or error.")
        return []
    # Find all game links - try multiple patterns
    game_patterns = [
        r'href="/perl/chessgame\?gid=(\d+)"',
        r'href="chessgame\?gid=(\d+)"',
        r'chessgame\?gid=(\d+)',
        r'gid=(\d+)'
    ]
    game_ids = []
    for pattern in game_patterns:
        matches = re.findall(pattern, response.text)
        game_ids.extend(matches)
    return list(set(game_ids))  # Remove duplicates

def extract_tournament_links(page_num):
    """Extract tournament links from a tournament list page"""
    base_url = "https://www.chessgames.com"
    url = f"{base_url}/perl/tournaments?page={page_num}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    response = safe_request(url, headers=headers)
    if response is None:
        print(f"Error extracting tournaments from page {page_num}: too many retries or error.")
        return []
    # Save the page for debugging if no tournaments found
    if page_num == 14:  # Debug page 14 specifically
        with open(f"debug_page_{page_num}.html", "w", encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved debug page {page_num} for inspection")
    # Try multiple patterns for tournament links
    tournament_patterns = [
        r'href="(/perl/tournaments\?tid=\d+)"',
        r'href="(/perl/chess.pl\?tid=\d+)"',
        r'tournaments\?tid=(\d+)',
        r'chess.pl\?tid=(\d+)'
    ]
    tournament_urls = []
    for pattern in tournament_patterns:
        matches = re.findall(pattern, response.text)
        for match in matches:
            if 'tid=' in match:
                # It's a tournament link
                tournament_urls.append(urljoin(base_url, match))
            elif 'gid=' in match:
                # It's a game link, construct tournament URL
                # We'll need to get the tournament ID from the game page
                game_url = urljoin(base_url, f"/perl/chessgame?gid={match}")
                tournament_urls.append(game_url)
    # Remove duplicates and return
    unique_urls = list(set(tournament_urls))
    # If still no tournaments found, try a different approach
    if not unique_urls:
        print(f"No tournaments found with regex patterns on page {page_num}")
        print("Trying to find any links that might be tournaments...")
        # Look for any links that contain 'tournament' or 'tid'
        all_links = re.findall(r'href="([^"]*)"', response.text)
        tournament_links = [link for link in all_links if 'tournament' in link.lower() or 'tid=' in link]
        print(f"Found potential tournament links: {tournament_links[:5]}")  # Show first 5
        for link in tournament_links:
            if link.startswith('/'):
                unique_urls.append(urljoin(base_url, link))
            elif link.startswith('http'):
                unique_urls.append(link)
    return unique_urls

def main():
    os.makedirs("data/tournaments", exist_ok=True)
    # Track progress
    downloaded_count = 0
    failed_count = 0
    # Loop through tournament list pages
    for page_num in range(14, 215):
        print(f"\nProcessing page {page_num}...")
    try:
            tournament_urls = extract_tournament_links(page_num)
            print(f"Found {len(tournament_urls)} tournaments on page {page_num}")
            if not tournament_urls:
                print(f"Skipping page {page_num} - no tournaments found")
            for tournament_url in tournament_urls:
                print(f"Processing tournament: {tournament_url}")
                try:
                    game_ids = extract_game_ids_from_tournament(tournament_url)
                    print(f"  Found {len(game_ids)} games in tournament")
                    for game_id in game_ids:
                        # Check if already downloaded
                        pgn_file = f"data/tournaments/pgn_game_{game_id}.pgn"
                        if os.path.exists(pgn_file):
                            print(f"    Game {game_id} already exists, skipping")
                        print(f"    Downloading game {game_id}...")
                        # Try to download PGN directly
                        pgn_content = download_pgn_direct(game_id)
                        # Fallback: try to get PGN from game page if direct download fails
                        if not pgn_content:
                            pgn_content = download_pgn_from_game_page(game_id)
                        if pgn_content:
                            with open(pgn_file, "w", encoding='utf-8') as f:
                                f.write(pgn_content)
                            print(f"    Successfully downloaded game {game_id}")
                            downloaded_count += 1
                        else:
                            print(f"    Failed to download game {game_id}")
                            failed_count += 1
                        # Rate limiting
                        time.sleep(random.uniform(5, 10))
                    # Rate limiting between tournaments
                    time.sleep(random.uniform(10, 20))
                except Exception as e:
                    print(f"  Error processing tournament {tournament_url}: {e}")
                    failed_count += 1
                    continue
            # Rate limiting between pages
            time.sleep(random.uniform(20, 40))
    except Exception as e:
            print(f"Error processing page {page_num}: {e}")
    print(f"\nScraping completed!")
    print(f"Successfully downloaded: {downloaded_count} games")
    print(f"Failed downloads: {failed_count} games")

if __name__ == "__main__":
    main() 
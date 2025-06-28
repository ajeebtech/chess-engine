import chess.pgn
import chess.engine
import os
import json
from tqdm import tqdm
import argparse
import time

STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon"  # Update if needed
PGN_PATH = "data/9mil/DATABASE4U.pgn"
MOVESET_PATH = "moveset.json"
FEN_BESTMOVE_PATH = "fen_bestmove.jsonl"
CHECKPOINT_PATH = "checkpoint.txt"

# Use Stockfish at maximum depth
MAX_DEPTH = 12  # For faster testing
SAVE_EVERY = 1000  # Save progress every N games
PROGRESS_EVERY = 100  # Show progress every N games

def save_progress(move_set, pairs, moveset_path, fen_bestmove_path, checkpoint_path, games_processed):
    print(f"\n[Checkpoint] Saving progress at game {games_processed}...")
    with open(moveset_path, "w") as f:
        json.dump(sorted(list(move_set)), f)
    with open(fen_bestmove_path, "a") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    with open(checkpoint_path, "w") as f:
        f.write(str(games_processed))
    print(f"[Checkpoint] Progress saved. Moveset size: {len(move_set)}")

def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return int(f.read().strip())
    return 0

def main(pgn_path, moveset_path, fen_bestmove_path, max_games=None, checkpoint_path=CHECKPOINT_PATH):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    move_set = set()
    pairs = []
    last_checkpoint = load_checkpoint(checkpoint_path)
    print(f"Resuming from game {last_checkpoint} (if 0, starting fresh)...")
    
    # If resuming, load existing moveset
    if os.path.exists(moveset_path):
        with open(moveset_path, "r") as f:
            move_set = set(json.load(f))
        print(f"Loaded existing moveset with {len(move_set)} moves")
    
    # Only count games if we're starting fresh or if max_games is specified
    if last_checkpoint == 0 or max_games:
        print(f"Counting games in {pgn_path}...")
        with open(pgn_path, encoding='utf-8', errors='ignore') as pgn:
            num_games = 0
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                num_games += 1
                if max_games and num_games >= max_games:
                    break
                if num_games % 10000 == 0:
                    print(f"Counted {num_games} games...")
        print(f"Found {num_games} games.")
    else:
        # Estimate total games for progress bar (rough estimate)
        num_games = 9000000  # Approximate for 9mil file
        print(f"Using estimated total of {num_games} games for progress bar")
    
    start_time = time.time()
    with open(pgn_path, encoding='utf-8', errors='ignore') as pgn:
        pbar = tqdm(total=num_games, desc="Games", initial=last_checkpoint)
        games_processed = 0
        positions_analyzed = 0
        
        # Skip already processed games
        while games_processed < last_checkpoint:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games_processed += 1
        
        print(f"Starting analysis from game {games_processed + 1}...")
        
        # Main processing loop
        while True:
            if max_games and games_processed >= max_games:
                break
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            black_pos_count = 0
            for move in game.mainline_moves():
                if board.turn == chess.BLACK:
                    fen = board.fen()
                    black_pos_count += 1
                    try:
                        result = engine.analyse(board, chess.engine.Limit(depth=MAX_DEPTH))
                        best_move = result["pv"][0].uci()
                        move_set.add(best_move)
                        pairs.append({"fen": fen, "best_move": best_move})
                        positions_analyzed += 1
                    except Exception as e:
                        print(f"  Error analysing position: {e}")
                board.push(move)
            
            games_processed += 1
            pbar.update(1)
            
            # Show progress more frequently
            if games_processed % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                rate = games_processed / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'Moves': len(move_set),
                    'Positions': positions_analyzed,
                    'Rate': f"{rate:.1f} games/s"
                })
            
            # Periodic checkpoint
            if games_processed % SAVE_EVERY == 0:
                save_progress(move_set, pairs, moveset_path, fen_bestmove_path, checkpoint_path, games_processed)
                pairs = []  # Clear pairs after saving
        
        pbar.close()
    
    # Final save
    save_progress(move_set, pairs, moveset_path, fen_bestmove_path, checkpoint_path, games_processed)
    engine.quit()
    
    total_time = time.time() - start_time
    print(f"\nDone! Processed {games_processed} games in {total_time/3600:.1f} hours")
    print(f"Final moveset size: {len(move_set)} unique moves")
    print(f"Total positions analyzed: {positions_analyzed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn_path", type=str, default=PGN_PATH, help="Path to PGN file")
    parser.add_argument("--moveset_path", type=str, default=MOVESET_PATH, help="Path to save moveset.json")
    parser.add_argument("--fen_bestmove_path", type=str, default=FEN_BESTMOVE_PATH, help="Path to save fen_bestmove.jsonl")
    parser.add_argument("--max_games", type=int, default=None, help="Maximum number of games to process")
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH, help="Checkpoint file path")
    args = parser.parse_args()
    main(args.pgn_path, args.moveset_path, args.fen_bestmove_path, args.max_games, args.checkpoint_path) 
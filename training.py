import chess.pgn
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re
import os
import json
import time

STOCKFISH_PATH = "/Users/jatin/Documents/python/supervisedchess/stockfish/stockfish-macos-m1-apple-silicon"  # update this

# Set up engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Load moveset from file
def load_moveset(moveset_path="moveset.json"):
    """Load the moveset from the JSON file and create mappings."""
    with open(moveset_path, 'r') as f:
        moveset = json.load(f)
    
    move_to_idx = {move: idx for idx, move in enumerate(moveset)}
    idx_to_move = {idx: move for idx, move in enumerate(moveset)}
    
    print(f"Loaded {len(moveset)} unique moves from moveset")
    return moveset, move_to_idx, idx_to_move

# Load the moveset
MOVE_LIST, MOVE_TO_IDX, IDX_TO_MOVE = load_moveset()

# FEN to tensor (8x8x12 one-hot)
def board_to_tensor(board):
    piece_map = board.piece_map()
    tensor = torch.zeros(12, 8, 8)
    piece_to_index = {
        "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
        "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
    }
    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece_to_index[piece.symbol()]
        tensor[idx][row][col] = 1
    return tensor

# Simple CNN
class ChessNet(nn.Module):
    def __init__(self, num_moves):
        super(ChessNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 8 * 8, num_moves)

    def forward(self, x):
        return self.fc(self.conv(x))

class FenBestMoveDataset(Dataset):
    def __init__(self, jsonl_path, move_to_idx, max_positions=None):
        self.samples = []
        self.move_to_idx = move_to_idx
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_positions and i >= max_positions:
                    break
                data = json.loads(line)
                fen = data['fen']
                best_move = data['best_move']
                if best_move in move_to_idx:
                    self.samples.append((fen, best_move))
        print(f"Loaded {len(self.samples)} positions from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, best_move = self.samples[idx]
        board = chess.Board(fen)
        x = board_to_tensor(board)
        y = self.move_to_idx[best_move]
        return x, y

if __name__ == "__main__":
    # Replace dataset creation with FenBestMoveDataset
    DATA_PATH = "fen_bestmovev0.jsonl"
    dataset = FenBestMoveDataset(DATA_PATH, MOVE_TO_IDX, max_positions=None)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model, optimizer, and loss remain the same
    model = ChessNet(num_moves=len(MOVE_LIST))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Model initialized with {len(MOVE_LIST)} output classes")

    # Training
    print("Training...")
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        sample_pbar = tqdm(total=len(dataset), desc=f"Epoch {epoch+1} (positions)")
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            sample_pbar.update(batch_x.size(0))
            sample_pbar.set_postfix(loss=loss.item())
            time.sleep(0.1)
        sample_pbar.close()
        print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}")
        torch.save(model.state_dict(), f"chess_model_epoch{epoch+1}.pth")
        print(f"Model saved to chess_model_epoch{epoch+1}.pth")

    # Save the model after training
    torch.save(model.state_dict(), "chess_model.pth")
    print("Model saved to chess_model.pth")

    engine.quit()

    # Example inference usage after training
    from inference import ChessMovePredictor

    predictor = ChessMovePredictor("chess_model.pth", "moveset.json")
    fen = input("Enter FEN for inference: ")
    board = chess.Board(fen)
    move = predictor.predict_best_move(board)
    print("Predicted move:", move)
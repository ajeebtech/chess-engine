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

class ChessDataset(Dataset):
    def __init__(self, data_dir, move_to_idx, max_games=None):
        self.data_dir = data_dir
        self.move_to_idx = move_to_idx
        self.max_games = max_games or float('inf')
        self.positions = []
        self._index_games()

    def _index_games(self):
        import os
        count = 0
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".pgn"):
                    file_path = os.path.join(root, file)
                    with open(file_path) as pgn:
                        while count < self.max_games:
                            game = chess.pgn.read_game(pgn)
                            if game is None:
                                break
                            board = game.board()
                            for move in game.mainline_moves():
                                if board.turn == chess.BLACK:
                                    self.positions.append(board.fen())
                                board.push(move)
                            count += 1
                            if count >= self.max_games:
                                break

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        board = chess.Board(self.positions[idx])
        x = board_to_tensor(board)

        # Get label from Stockfish
        result = engine.analyse(board, chess.engine.Limit(depth=12))
        best_move = result["pv"][0].uci()
        
        # Check if move is in our moveset, if not skip this position
        if best_move in self.move_to_idx:
            y = self.move_to_idx[best_move]
            return x, y
        else:
            # Return a dummy position if move not in moveset
            # You might want to handle this differently
            return x, 0

# Create dataset and model with the correct number of moves
dataset = ChessDataset(
    "/Users/jatin/Documents/python/supervisedchess/data",
    MOVE_TO_IDX,
    max_games=10  # You can increase this
)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create model with the exact number of moves from moveset
model = ChessNet(num_moves=len(MOVE_LIST))
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

print(f"Model initialized with {len(MOVE_LIST)} output classes")

# Training
print("Training...")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}")

engine.quit()
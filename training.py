import chess.pgn
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

STOCKFISH_PATH = "/Users/jatin/Documents/python/supervisedchess/stockfish/stockfish-macos-m1-apple-silicon"  # update this

# Set up engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Fixed list of possible moves
MOVE_LIST = []
MOVE_TO_IDX = {}
IDX_TO_MOVE = {}

def update_move_list(move_uci):
    if move_uci not in MOVE_TO_IDX:
        idx = len(MOVE_LIST)
        MOVE_LIST.append(move_uci)
        MOVE_TO_IDX[move_uci] = idx
        IDX_TO_MOVE[idx] = move_uci

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
    def __init__(self, pgn_path, move_to_idx, max_games=None):
        self.pgn_path = pgn_path
        self.move_to_idx = move_to_idx
        self.max_games = max_games or float('inf')
        self.positions = []
        self._index_games()

    def _index_games(self):
        with open(self.pgn_path) as pgn:
            count = 0
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

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        board = chess.Board(self.positions[idx])
        x = board_to_tensor(board)

        # Get label from Stockfish
        result = engine.analyse(board, chess.engine.Limit(depth=12))
        best_move = result["pv"][0].uci()
        update_move_list(best_move)

        y = MOVE_TO_IDX[best_move]
        return x, y

dataset = ChessDataset(
    "/Users/jatin/Documents/python/supervisedchess/data/tournaments/pgn_game_1001470&comp=1.pgn",
    MOVE_TO_IDX,
    max_games=10  # You can increase this
)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ChessNet(num_moves=5000)  # We'll resize later
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# Resize model to actual number of moves after scanning data
model = ChessNet(num_moves=len(MOVE_LIST))

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
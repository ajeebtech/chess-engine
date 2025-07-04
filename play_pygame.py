import pygame
import sys
import os
import chess
import chess.engine
from inference import ChessMovePredictor

# --- CONFIG ---
BOARD_SIZE = 8
SQUARE_SIZE = 80
WINDOW_SIZE = BOARD_SIZE * SQUARE_SIZE
PIECES_DIR = os.path.join(os.path.dirname(__file__), 'pieces')
FPS = 60

# --- INIT ---
pygame.init()
window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('Chess: Stockfish (White) vs Model (Black)')
clock = pygame.time.Clock()

# --- LOAD PIECE IMAGES ---
PIECE_SYMBOL_TO_FILENAME = {
    'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
    'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk'
}
PIECE_IMAGES = {}
for symbol, fname in PIECE_SYMBOL_TO_FILENAME.items():
    path = os.path.join(PIECES_DIR, f'{fname}.png')
    if os.path.exists(path):
        try:
            img = pygame.image.load(path)
            img = pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))
            PIECE_IMAGES[symbol] = img
            print(f"Loaded image for {symbol} from {path}")
        except Exception as e:
            print(f"Failed to load image for {symbol} from {path}: {e}")
    else:
        print(f"Missing image file for {symbol}: {path}")

# --- DRAWING ---
def draw_board(screen, board, selected_square=None, drag_piece=None, drag_pos=None):
    colors = [pygame.Color(240, 217, 181), pygame.Color(181, 136, 99)]
    for rank in range(BOARD_SIZE):
        for file in range(BOARD_SIZE):
            square = chess.square(file, 7 - rank)
            color = colors[(rank + file) % 2]
            rect = pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)
            piece = board.piece_at(square)
            if piece:
                if drag_piece and drag_piece['from_sq'] == square:
                    continue  # Don't draw the piece being dragged
                img = PIECE_IMAGES.get(piece.symbol())
                if img:
                    screen.blit(img, rect)
            if selected_square == square:
                pygame.draw.rect(screen, (0, 255, 0), rect, 4)
    # Draw dragged piece on top
    if drag_piece and drag_pos:
        img = PIECE_IMAGES.get(drag_piece['piece'].symbol())
        if img:
            rect = img.get_rect(center=drag_pos)
            screen.blit(img, rect)

# --- MAIN GAME LOOP ---
def main():
    board = chess.Board()
    STOCKFISH_PATH = "/Users/jatin/Documents/python/supervisedchess/stockfish/stockfish-macos-m1-apple-silicon"
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    from inference import ChessMovePredictor
    predictor = ChessMovePredictor("chess_model.pth", "moveset.json")
    running = True
    stockfish_color = chess.WHITE
    model_color = chess.BLACK
    selected_square = None
    drag_piece = None
    drag_pos = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
        # Stockfish (White) move
        if board.turn == stockfish_color and not board.is_game_over():
            pygame.event.pump()
            pygame.display.set_caption('Stockfish (White) thinking...')
            result = engine.play(board, chess.engine.Limit(depth=12))
            stockfish_move = result.move
            if stockfish_move is None:
                print("Stockfish cannot find a legal move. Game over.")
                running = False
                continue
            board.push(stockfish_move)
            pygame.display.set_caption('Stockfish (White) vs Model (Black)')
        # Model (Black) move
        elif board.turn == model_color and not board.is_game_over():
            pygame.event.pump()
            pygame.display.set_caption('Model (Black) thinking...')
            model_move_uci = predictor.predict_best_move(board)
            if model_move_uci is None:
                print("Model cannot find a legal move. Game over.")
                running = False
                continue
            model_move = chess.Move.from_uci(model_move_uci)
            board.push(model_move)
            pygame.display.set_caption('Stockfish (White) vs Model (Black)')
        # Draw
        window.fill((0, 0, 0))
        draw_board(window, board, selected_square, drag_piece, drag_pos)
        pygame.display.flip()
        clock.tick(FPS)
        # End game check
        if board.is_game_over():
            print("Game over! Result:", board.result())
            print("Reason:", board.outcome().termination.name)
            running = False
    engine.quit()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 
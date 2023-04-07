import numpy as np
from creversi import dtypeBitboard
import torch

def get_board(board,mode):
    """現在の盤面を取得する関数

    Boardオブジェクトを渡すと、引数modeで指定された型で盤面を取得する。
    
    Parameters
    ----------
    board: creversi.creversi.Board
        creversiのBoardオブジェクト。
    mode: string
        "array"または"bitboard"または"one-channel"。

    Returns
    -------
    b: numpy.array
        mode="array"とすると、dtype=float32のNumPy配列(2チャンネル8x8)を返す。
        mode="bitboard"とすると、dtype=uint8のNumPy配列(1x16)を返す。
        mode="one-channel"とすると、dtype=float32のNumPy配列(1チャンネル8x8)を返す。1が黒、-1が白を表す。

    """
    if mode == "bitboard":
        b = np.empty(1, dtypeBitboard)
        board.to_bitboard(b)
    elif mode == "array":        
        b = np.empty((2,8,8),dtype=np.float32)
        board.piece_planes(b)
    elif mode == "one-channel":
        b = np.empty((2,8,8),dtype=np.float32)
        board.piece_planes(b)
        b = b[0] - b[1]
    return b

def get_q(board, model):
    """Value-Networkの出力を返す関数

    Boardオブジェクトとモデルを渡すと、現在の盤面に対するValue-Networkの出力を返す。

    Parameters
    ----------
    board: creversi.creversi.Board
        creversiのBoardオブジェクト。
    model: DuelingNet
        Value-Networkの学習済みモデル。

    Returns
    -------
    q: numpy.array
        modelに現在の盤面を入力した時の出力。
    """
    model.eval()
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    s = torch.from_numpy(get_board(board,"array")).to(torch.float32).to(device).view(1,2,8,8)
    q = model(s)[0].detach().to("cpu").numpy()
    return q

def softmax(x):
    """softmax関数"""
    u = np.sum(np.exp(x))
    return np.exp(x) / u

def get_prob(board, model):
    """Policy-Networkの出力を返す関数

    Boardオブジェクトとモデルを渡すと、現在の盤面に対するPolicy-Networkの出力(確率値)を返す。

    Parameters
    ----------
    board: creversi.creversi.Board
        creversiのBoardオブジェクト。
    model: PolicyNet
        Policy-Networkの学習済みモデル。

    Returns
    -------
    p: numpy.array
        modelに現在の盤面を入力した時の出力(確率値)。
    """
    model.eval()
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    s = torch.from_numpy(get_board(board,"array")).to(torch.float32).to(device).view(1,2,8,8)
    p = model(s)[0].detach().to("cpu").numpy()
    return softmax(p)


def show_board(board, mode):
    """盤面をテキストとして表示する関数

    Parameters
    ----------
    board: creversi.creversi.Board
        creversiのBoardオブジェクト。
    mode: string
        iOS版向け表示、Android版向け表示を指定。
    """
    b = get_board(board,"one-channel")
    if mode == "iOS":
        board_str = "　a  b  c   d  e   f  g  h\n"
    else:
        board_str = "   a b c d  e f g h\n"
    for i,row in enumerate(b):
        board_str += str(i+1) + " "
        for j,v in enumerate(row):
            if v == 0:
                board_str += "-- "
            if v == (1 if board.turn else -1):
                board_str += "● "
            if v == (-1 if board.turn else 1):
                board_str += "○ "
            if j == 7 and i < 7:
                board_str += "\n"
    return board_str

def get_v(board, model):
    """Value-Network(状態価値)の出力を返す関数

    Boardオブジェクトとモデルを渡すと、先手の勝率を返す。

    Parameters
    ----------
    board: creversi.creversi.Board
        creversiのBoardオブジェクト。
    model: ValueNet
        Value-Networkの学習済みモデル。

    Returns
    -------
    v: float32
        modelに現在の盤面を入力した時の出力。
    """
    model.eval()
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    s = torch.from_numpy(get_board(board,"array")).to(device).view(1,2,8,8)
    v = model(s).detach().to("cpu")
    sigmoid = torch.nn.Sigmoid()
    return sigmoid(v).item()
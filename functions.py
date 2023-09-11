import torch
import numpy as np
from copy import copy
    
def board_to_array(board):
    """
    boardオブジェクトからndarrayに変換する関数。
    第1チャンネルは黒石の位置、第2チャンネルに白石の位置、第3チャンネルに空白の位置、
    第4チャンネルに合法手の位置、第5チャンネルに返せる石の個数、第6チャンネルに隅=1、
    第7チャンネルに1埋め、第8チャンネルに0埋め。
    """
    b = np.zeros((8,8,8), dtype=np.float32)
    board.piece_planes(b)
    if not board.turn:
        b = b[[1,0,2,3,4,5,6,7],:,:]
    b[2] = np.where(b[0]+b[1]==1, 0, 1)
    legal_moves = list(board.legal_moves)
    if legal_moves != [64]:
        n_returns = []
        for move in legal_moves:
            board_ = copy(board)
            n_before = board_.opponent_piece_num()
            board_.move(move)
            n_after = board_.piece_num()
            n_returns.append(n_before-n_after)
        tmp = np.zeros(64)
        tmp[legal_moves] = n_returns
        tmp = tmp.reshape(8,8)
        b[3] = np.where(tmp > 0,1,0)
        b[4] = tmp
    b[5] = np.array([1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.]).reshape(8,8)
    b[6] = 1
    return b

def board_to_array2(board):
    """
    boardオブジェクトからndarrayに変換する関数(ValueNetwork用)。
    第1チャネルは黒石の位置、第2チャネルに白石の位置、第3チャネルに空白の位置、
    第4チャネルに合法手の位置、第5チャネルに返せる石の個数、第6チャネルに隅=1、
    第7チャネルに1埋め、第8チャネルに0埋め、第9チャネルに手番情報(黒番=0埋め、白番=1埋め)
    """
    b = np.zeros((9,8,8), dtype=np.float32)
    board.piece_planes(b)
    if not board.turn:
        b = b[[1,0,2,3,4,5,6,7,8],:,:]
        b[8] = 1
    b[2] = np.where(b[0]+b[1]==1, 0, 1)
    legal_moves = list(board.legal_moves)
    if legal_moves != [64]:
        n_returns = []
        for move in legal_moves:
            board_ = copy(board)
            n_before = board_.opponent_piece_num()
            board_.move(move)
            n_after = board_.piece_num()
            n_returns.append(n_before-n_after)
        tmp = np.zeros(64)
        tmp[legal_moves] = n_returns
        tmp = tmp.reshape(8,8)
        b[3] = np.where(tmp > 0,1,0)
        b[4] = tmp
    b[5] = np.array([1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1.]).reshape(8,8)
    b[6] = 1
    return b

def board_to_array_aug(board, return_torch=False):
    boards = []
    board_array = board_to_array(board)
    boards.append(board_array)
    boards.append(np.flip(board_array,axis=2).copy())
    for k in range(1,4):
        board_array_rot = np.rot90(board_array, k=k, axes=(1,2)).copy()
        boards.append(board_array_rot)
        boards.append(np.flip(board_array_rot, axis=2).copy())
    if return_torch:
        return torch.from_numpy(np.array(boards))
    else:
        return np.array(boards)

def board_to_array_aug2(board, return_torch=False):
    boards = []
    board_array = board_to_array2(board)
    boards.append(board_array)
    boards.append(np.flip(board_array,axis=2).copy())
    for k in range(1,4):
        board_array_rot = np.rot90(board_array, k=k, axes=(1,2)).copy()
        boards.append(board_array_rot)
        boards.append(np.flip(board_array_rot, axis=2).copy())
    if return_torch:
        return torch.from_numpy(np.array(boards))
    else:
        return np.array(boards)
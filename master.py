from creversi import *
import re
import torch
import numpy as np
from copy import copy
import pickle
import io
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
    
def board_to_array(board, return_torch=False):
    """
    (PolicyNetwork用)
    boardオブジェクトからndarrayに変換する関数。
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
    if return_torch:
        return torch.from_numpy(b)
    else:
        return b

def board_to_array2(board, return_torch=False):
    """
    (ValueNetwork用)
    boardオブジェクトからndarrayに変換する関数。
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
    if return_torch:
        return torch.from_numpy(b)
    else:
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

class Master:
    def __init__(self, mode):
        assert mode in ['strong', 'enjoy']
        self.mode = mode
        self.board = Board()
        self.moves = ''
        self.tree = None

    def input_move(self, input_str):
        """
        Parameters
        ----------
        input_str : str
            入力文字列

        Returns
        -------
        move_str : str
            ムーブ文字列

        Process
        -------
        1. 全角英数字を半角化する
        2. pass系の文字列なら'pass'を返す
        3. a1~h8の形式ならその文字列を返す
        4. それ以外ならNoneを返す (--> Error)
        """
        zenkaku = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
        hankaku = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        table = str.maketrans(zenkaku, hankaku)
        input_str = input_str.translate(table).lower()

        if input_str in ['pass', 'パス', 'ぱす', 'pas']:
            return 'pass'
        
        if re.match(r'^[a-h][1-8]$', input_str):
            return input_str

        return None

    def move(self, move_str):
        """
        Parameters
        ----------
        move_str : str
            ムーブ文字列

        Returns
        -------
        None

        Process
        -------
        1. 盤面をmove_strに従って更新する
        2. move_strを履歴(moves)に追加する
        """
        self.board.move_from_str(move_str)
        self.moves += move_str
    
    def get_move(self, model_p=None, model_v=None):
        """
        Parameters
        ----------
        model_p : PolicyNetwork
            方策関数
        model_v : ValueNetwork
            価値関数

        Returns
        -------
        move_str : str
            ムーブ文字列

        Process
        -------
        1. modeが'strong'なら、方策関数を用いて最善手を探索する
        2. modeが'enjoy'なら、価値関数/MiniMaxを用いて最善手を探索する
        """
        if self.mode == 'strong':
            model_p.eval()
            legal_moves = list(self.board.legal_moves)
            if 64 in legal_moves:
                return 'pass'
            
            with torch.no_grad():
                output = model_p(board_to_array(self.board,True).unsqueeze(0)).numpy()
            output = output[0][legal_moves]
            return move_to_str(legal_moves[np.argmax(output)])
        
        elif self.mode == 'enjoy':
            """
            1. 合計石個数が53個以下なら、深さ3の木を作成
            2. 合計石個数が54個以上なら、深さmaxの木を作成
            """
            if self.board.piece_sum() <= 53:
                model_v.eval()
                self.tree = Node(self.board)
                create_tree(self.tree, 3)
                self.tree.backward(model_v)
                _, move = minimax(self.tree, self.board.turn, 3)
                return move_to_str(move)
            else:
                self.tree = Node(self.board)
                create_tree(self.tree, 100)
                self.tree.backward()
                _, move = minimax(self.tree, self.board.turn, 100)
                return move_to_str(move)

    def evaluate_board(self, model_v):
        """
        Parameters
        ----------
        model_v : ValueNetwork
            価値関数

        Returns
        -------
        z : float
            盤面の評価値

        Process
        -------
        1. 現在の木にもとづいて盤面の評価値を計算する
        """
        if self.tree is not None:
            zmax = self.tree.get_max()
            zmin = self.tree.get_min()
            return zmax, zmin
        else:
            return 0, 0
        
    def save(self, path):
        """Masterオブジェクトの保存"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, path):
        """Masterオブジェクトの読み込み"""
        with open(path, 'rb') as f:
            master = pickle.load(f)
        self.mode = master.mode
        self.board = master.board
        self.moves = master.moves
        self.tree = master.tree

    def save_moves(self, path):
        """棋譜の保存"""
        # 記録する文字列を作成
        ## 対戦が途中の場合、棋譜の末尾に##をつける
        if not self.board.is_game_over():
            moves = self.moves + '##'
        else:
            moves = self.moves + '\n'

        # 書き込み
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(moves)
        else:
            ## 末尾に##がある場合、その行を削除してから追加
            with open(path, 'r') as f:
                lines = f.readlines()
            if lines[-1].strip()[-2:] == '##':
                lines = lines[:-1]
            lines.append(moves)
            with open(path, 'w') as f:
                f.writelines(lines)
    
    def save_image(self, path):
        """盤面の画像を保存"""
        renderPM.drawToFile(svg2rlg(io.StringIO(self.board.to_svg())), path, fmt="JPEG")

        

class Node:
    def __init__(self, board):
        self.data = [None, None, None, None]
        self.board = board
        self.children = []

    def add(self, child):
        self.children.append(child)

    def backward(self, model_v=None):
        if not self.children:
            self.data = self.evaluate(model_v)
        else:
            for child in self.children:
                child.backward(model_v)
            absmin,absmax,min_,max_ = np.inf,-np.inf,np.inf,-np.inf
            for child in self.children:
                absmin = min(absmin, child.get_absmin())
                absmax = max(absmax, child.get_absmax())
                min_ = min(min_, child.get_min())
                max_ = max(max_, child.get_max())
            self.data = [absmin, absmax, min_, max_]

    def evaluate(self, model_v=None):
        if model_v is None:
            assert self.board.is_game_over(), self.board.to_line()
            z = self.board.diff_num() if self.board.turn else -self.board.diff_num()
            return (abs(z), abs(z), z, z)
        else:
            with torch.no_grad():
                z = model_v(board_to_array_aug2(self.board,True)).mean().item()*64
            return (abs(z), abs(z), z, z)
    
    def get_absmin(self):
        return self.data[0]

    def get_absmax(self):
        return self.data[1]

    def get_min(self):
        return self.data[2]

    def get_max(self):
        return self.data[3]

def apply_move(board, move):
    """仮想盤面を生成"""
    board_ = copy(board)
    board_.move(move)
    return board_

def create_tree(node, depth):
    """根=node,深さ=depthの木を作成"""
    if depth == 0:
        return
    for move in list(node.board.legal_moves):
        new_board = apply_move(node.board, move)
        child = Node(new_board)
        node.add(child)
        if not new_board.is_game_over():
            create_tree(child, depth - 1)

def minimax(node, turn, depth):
    '''ミニマックス法で引き分け最善手を探索する関数'''
    if depth == 0 or node.board.is_game_over():
        z = node.get_absmin()
        return z, list(node.board.legal_moves)[0]
    
    if node.board.turn == turn:
        zbest = np.inf
        mbest = None
        for i,child in enumerate(node.children):
            z,_ = minimax_draw(child, turn, depth-1)
            if zbest > z:
                zbest = z
                mbest = list(node.board.legal_moves)[i]
    else:
        zbest = -np.inf
        mbest = None
        for i,child in enumerate(node.children):
            z,_ = minimax_draw(child, turn, depth-1)
            if zbest < z:
                zbest = z
                mbest = list(node.board.legal_moves)[i]
    assert (abs(zbest)<100) and (mbest is not None), f'zbest={zbest}, mbest={mbest}'
    return zbest, mbest
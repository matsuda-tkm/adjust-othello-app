import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from creversi import *
from network import ValueNetwork, PolicyNetwork
from master import board_to_array, board_to_array_aug2, Master
import torch
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.header('忖度オセロAI')

model_v = ValueNetwork().to(device)
model_p = PolicyNetwork().to(device)
model_v.load_state_dict(torch.load('../trained_models/value-network-v2.pth', map_location=device))
model_p.load_state_dict(torch.load('../trained_models/policy-network-v3.pth', map_location=device))
model_v.eval()
model_p.eval()

def move_AI(board):
    global model_p
    with torch.no_grad():
        output = model_p(board_to_array(board,True).unsqueeze(0).to(device)).numpy()
        move = legal_moves[np.argmax(output[0][legal_moves]).item()]
    return move

def move_AI2(master):
    move_str = master.get_move(model_p, model_v)
    return move_from_str(move_str)

def eval(board):
    with torch.no_grad():
        output = model_v(board_to_array_aug2(board,True).to(device)).numpy() *64
        output = output.mean()
    return output

def eval2(master):
    return master.evaluate_board(model_v)


def move(x, y):
    move = int(8*x+y)
    if move in list(board.legal_moves):
        board.move(move)
        master.move(move_to_str(move))
        st.session_state.board = board
        st.session_state.master = master
        st.session_state.move = move
        st.session_state.eval.append(eval(board))
        M,m = eval2(master)
        st.session_state.eval_max.append(M)
        st.session_state.eval_min.append(m)

        legal_moves = list(board.legal_moves)
        if legal_moves != [64]:
            move = move_AI2(master)
            board.move(move)
            master.move(move_to_str(move))
            st.session_state.board = board
            st.session_state.master = master
            st.session_state.move = move
            st.session_state.eval.append(eval(board))
            M,m = eval2(master)
            st.session_state.eval_max.append(M)
            st.session_state.eval_min.append(m)
        else:
            board.move(64)
            master.move('pass')
            st.session_state.board = board
            st.session_state.master = master
            st.session_state.move = 64
            st.session_state.eval.append(eval(board))
            M,m = eval2(master)
            st.session_state.eval_max.append(M)
            st.session_state.eval_min.append(m)

def move_pass():
    board.move(64)
    st.session_state.board = board
    st.session_state.move = 64

    move = move_AI(board)
    board.move(move)
    st.session_state.board = board
    st.session_state.move = move
    st.session_state.eval.append(eval(board))

st.markdown("""
    <style>
        div.row-widget {
            height: 18px;
        }
        div.row-widget > button[data-testid="baseButton-secondary"] {
            border-color: #dddddd;
            border-style: dashed;
        }
    </style>
    """, unsafe_allow_html=True)

# INITIALIZE
if 'board' not in st.session_state:
    board = Board()
    master = Master(mode='adjust')
    st.session_state.board = board
    st.session_state.master = master
    st.session_state.eval = []
    st.session_state.eval_max = []
    st.session_state.eval_min = []
else:
    board = st.session_state.board
    master = st.session_state.master

# SET BOARD
planes = np.empty((2, 8, 8), dtype=np.float32)
board.piece_planes(planes)
planes = planes[0]-planes[1] if board.turn else planes[1]-planes[0]

# SCORE
black = board.piece_num() if board.turn else board.opponent_piece_num()
white = board.opponent_piece_num() if board.turn else board.piece_num()
col1,col2,col3 = st.columns(3)
if board.is_game_over():
    col1.metric(label='勝者', value='先手の勝ち' if black>white else ('後手の勝ち' if white>black else '引き分け'), delta=abs(black-white))
else:
    col1.metric(label='手番', value='●先手番' if board.turn else '○後手番')
col2.metric(label='●先手', value=black)
col3.metric(label='○後手', value=white)


# BOARD
cols = st.columns(25)
for j,col in enumerate(cols):
    if j == 8: break
    for i in range(8):
        color = ':black_circle:' if planes[i][j]>0 else (':white_circle:' if planes[i][j]<0 else '')
        col.button(color, key=i*8+j, use_container_width=True, on_click=move, args=(i,j))

# PASS
st.write('')
st.button('パスする', disabled=False if (list(board.legal_moves) == [64]) and (not board.is_game_over()) else True, type='primary', on_click=move_pass)
st.write('')

# PLOT
width = 60 if len(st.session_state.eval)<60 else len(st.session_state.eval)
plt.style.use('ggplot')
fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(1,1,1)
ax.plot(range(width), st.session_state.eval+[np.nan]*(width-len(st.session_state.eval)), color='red', marker='o')
ax.fill_between(
    range(width), 
    st.session_state.eval_max+[np.nan]*(width-len(st.session_state.eval_max)), 
    st.session_state.eval_min+[np.nan]*(width-len(st.session_state.eval_min)),
    color='red', alpha=0.1)
ax.set_xlim(-1, width+1)
ax.set_ylim(-65,65)
ax.hlines(0, -1, width+1, color='k', zorder=0)

st.pyplot(fig)



# !pip install creversi streamlit streamlit-image-coordinates cairosvg
# !npm install localtunnel
# !streamlit run app.py &>/content/logs.txt & npx localtunnel --port 8501 & curl ipv4.icanhazip.com
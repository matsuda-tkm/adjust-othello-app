# flask
from flask import Flask, request, abort, render_template
app = Flask(__name__)
# line-bot
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
# my module
from network import DuelingNet, ValueNet
from functions import *
from creversi import Board, move_to_str, move_from_str
import creversi
# translation
from googletrans import Translator
# basic module
import datetime, json, pickle, os, random
# svg --> jpg
import io
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
# calc module
import torch
import numpy as np


model_dir = "./mysite/trained_models/"
json_dir = "./mysite/"
log_message_dir = "./mysite/log/message/"
log_mode_dir = "./mysite/log/mode/"
log_board_dir = "./mysite/log/board/"
log_move_dir = "./mysite/log/move/"
img_dir = "https://matsudatkm.pythonanywhere.com/static/"
img_save_dir = "./mysite/static/"


# moveをまとめる
for i,path in enumerate(os.listdir("./mysite/log/move")):
    if path != "MOVE_ALL.txt":
        with open(f"{log_move_dir}{path}","r") as f:
            text = f.read()
        with open(f"{log_move_dir}MOVE_ALL.txt", "w" if i==0 else "a") as f:
            f.write(text)

## save models(only first time)
# value_net_state_dict = torch.load(f"{model_dir}value_net2.pth")
# result_net_state_dict = torch.load(f"{model_dir}result_net.pth")
# value_net = DuelingNet(100)
# result_net = ValueNet(100)
# value_net.load_state_dict(value_net_state_dict)
# result_net.load_state_dict(result_net_state_dict)
# torch.save(value_net,f"{model_dir}v2.pth")
# torch.save(result_net,f"{model_dir}r.pth")

# load models
value_net = torch.load(f"{model_dir}v2.pth")
result_net = torch.load(f"{model_dir}r.pth")


# read ACCESS_TOKEN and SECRET from json file
with open(f"{json_dir}token.json", "r") as f:
    tokens = json.load(f)
ACCESS_TOKEN = tokens["ACCESS_TOKEN"]
SECRET = tokens["SECRET"]

# authentation
line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(SECRET)

# home page
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

# gallery page
@app.route("/board-gallery", methods=["GET", "POST"])
def board_gallery():
    image_dir = './mysite/static/'
    image_list = []
    userId_list = []
    AI_list = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_list.append(os.path.join('https://matsudatkm.pythonanywhere.com/static/', filename))
            userId_list.append(log_board_dir + filename[:-4].replace("image_","BOARD_") + ".pkl")
    for user in userId_list:
        with open(user, "rb") as f:
            _,_,ai_type = pickle.load(f)
            AI_list.append(ai_type)
    return render_template('board_gallery.html', image_list=image_list, ai_type_list=AI_list)

# endpoint from linebot
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

# 1手先読み
def read_forward(board, model):
    bitboard = get_board(board,"bitboard")
    legal_moves = list(board.legal_moves)
    Qs_forward = []
    for move in legal_moves:
        board_ = Board()
        board_.set_bitboard(bitboard, True)  # AI先手ならTrueに設定
        board_.move(move)
        legal_moves_next = list(board_.legal_moves)
        q = get_q(board_, model)
        Qs_forward.append(q[legal_moves_next].mean())
    Qs_forward = np.array(Qs_forward)
    return legal_moves[Qs_forward.argmin()]

# get move
def get_move(board, q_opponent, threshold=0.2, ai_type="adjust"):
    global value_net
    assert board.turn, "board.turn is False!!"
    legal_moves = list(board.legal_moves)
    q = get_q(board, value_net)
    q_mean = get_v(board, result_net)
    if ai_type=="adjust":
        if q_mean >= 0.5+threshold:
            return legal_moves[q[legal_moves].argmin()]  # AI優勢なら弱い手を打つ
        elif 0.5-threshold < q_mean < 0.5+threshold:
            return legal_moves[q[legal_moves].argmax()]  # 閾値内なら強めの手を打つ
        else:
            return read_forward(board, value_net)  # 閾値を超えたら本気の手を打つ
    elif ai_type=="strong":
        if random.random() < 0.5:
            return legal_moves[q[legal_moves].argmax()]
        return read_forward(board, value_net)

##########
## MAIN ##
##########
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    ############################
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d-%H:%M:%S.%f")
    message = event.message.text
    userId = event.source.user_id
    send_list = []
    # [SAVE] massage
    ############################
    try:
        with open(f"{log_message_dir}MESSAGE_{userId}.txt","r") as f:
            log_massage = f.read()
            log_massage += "\n"
    except FileNotFoundError:
        log_massage = ""
    log_massage += f"'{now}','{message}'"
    with open(f"{log_message_dir}MESSAGE_{userId}.txt","w") as f:
        f.write(log_massage)
    # [RECEIVE] start a game
    #############################
    if message in ["【手加減AI】とオセロを始める", "【強いAI】とオセロを始める"]:
        ai_type = "adjust" if message=="【手加減AI】とオセロを始める" else "strong"
        board = Board()
        #[SAVE] board
        #############################
        with open(f"{log_board_dir}BOARD_{userId}.pkl", "wb") as f:
            pickle.dump([board.to_line(),board.turn,ai_type], f)
        # AI's turn
        move = get_move(board, 0, ai_type=ai_type)
        board.move(move)
        #[SAVE] board
        #############################
        with open(f"{log_board_dir}BOARD_{userId}.pkl", "wb") as f:
            pickle.dump([board.to_line(),board.turn,ai_type], f)
        # [SAVE] move
        #############################
        with open(f"{log_move_dir}MOVE_{userId}.txt", "a") as f:
            f.write("\n" + move_to_str(move))
        # show board
        #############################
        renderPM.drawToFile(svg2rlg(io.StringIO(board.to_svg())), f"{img_save_dir}image_{userId}.jpg", fmt="JPEG")
        send_list.append(TextSendMessage(text="あなたは白○で、私は黒●です。"))
        send_list.append(ImageSendMessage(f"{img_dir}image_{userId}.jpg", f"{img_dir}image_{userId}.jpg"))
        send_list.append(TextSendMessage(text=f"私は【{move_to_str(move)}】に置きました。\n次はあなたの番(白○)です。"))
        line_bot_api.reply_message(event.reply_token, send_list)
        return 0  # [SEND]

    # [RECEIVE] move
    #############################
    elif message in [chr(i) + str(j) for i in range(ord("a"), ord("h")+1) for j in range(1, 9)] + ["pass","パス","ぱす","pas","PASS","Pass"]:
        if message in ["パス","ぱす","pas","PASS","Pass"]:
            message = "pass"
        # [READ] board
        #############################
        try:
            with open(f"{log_board_dir}BOARD_{userId}.pkl", "rb") as f:
                line,turn,ai_type = pickle.load(f)
                board = creversi.Board(line,turn)
        except:
            send_list.append(TextSendMessage(text="対戦がまだ始まっていません！"))
            line_bot_api.reply_message(event.reply_token, send_list)
            return 0  # [SEND]
        # [IF-yes] is gameover?
        if board.is_game_over():
            send_list.append(TextSendMessage(text="対戦は既に終了しています！"))
            line_bot_api.reply_message(event.reply_token, send_list)
            return 0  # [SEND]
        # [IF-no] is game over?
        elif move_from_str(message) not in list(board.legal_moves):
            # [IF-no] is puttable?
            send_list.append(TextSendMessage(text="その場所には置けません！"))
            line_bot_api.reply_message(event.reply_token, send_list)
            return 0  # [SEND]
        # [IF-yes] is puttable?
        else:
            # User's turn
            q = get_q(board, value_net)
            legal_moves = list(board.legal_moves)
            if message=="pass":
                board.move_pass()
            else:
                board.move_from_str(message)
            # [SAVE] move
            #############################
            with open(f"{log_move_dir}MOVE_{userId}.txt", "a") as f:
                f.write("," + message)
            # [IF-yes] is gameover?
            if board.is_game_over():
                # [SAVE] board
                #############################
                with open(f"{log_board_dir}BOARD_{userId}.pkl", "wb") as f:
                    pickle.dump([board.to_line(),board.turn,ai_type], f)
                # show board & results
                #############################
                ai = board.piece_num()
                you = board.opponent_piece_num()
                winner = "あなた" if ai < you else ("私" if ai > you else "引き分け")
                renderPM.drawToFile(svg2rlg(io.StringIO(board.to_svg())), f"{img_save_dir}image_{userId}.jpg", fmt="JPEG")
                send_list.append(ImageSendMessage(f"{img_dir}image_{userId}.jpg", f"{img_dir}image_{userId}.jpg"))
                if winner == "引き分け":
                    send_list.append(TextSendMessage(text=f"AI(●):{ai}\nあなた(○):{you}\nで{winner}！"))
                else:
                    send_list.append(TextSendMessage(text=f"AI(●):{ai}\nあなた(○):{you}\nで{winner}の勝ち({you-ai})！"))
                line_bot_api.reply_message(event.reply_token, send_list)
                return 0  # [SEND]
            # [IF-no] is gameover?
            else:
                # AI's turn
                move = get_move(board, q[move_from_str(message)], ai_type=ai_type)
                board.move(move)
                # [SAVE] move
                #############################
                with open(f"{log_move_dir}MOVE_{userId}.txt", "a") as f:
                    f.write("," + move_to_str(move))
                # [SAVE] board
                #############################
                with open(f"{log_board_dir}BOARD_{userId}.pkl", "wb") as f:
                    pickle.dump([board.to_line(),board.turn,ai_type], f)
                # [IF-yes] is gameover?
                if board.is_game_over():
                    # show board & results
                    #############################
                    you = board.piece_num()
                    ai = board.opponent_piece_num()
                    winner = "あなた" if ai < you else ("私" if ai > you else "引き分け")
                    renderPM.drawToFile(svg2rlg(io.StringIO(board.to_svg())), f"{img_save_dir}image_{userId}.jpg", fmt="JPEG")
                    send_list.append(ImageSendMessage(f"{img_dir}image_{userId}.jpg", f"{img_dir}image_{userId}.jpg"))
                    if winner == "引き分け":
                        send_list.append(TextSendMessage(text=f"AI(●):{ai}\nあなた(○):{you}\nで{winner}！"))
                    else:
                        send_list.append(TextSendMessage(text=f"AI(●):{ai}\nあなた(○):{you}\nで{winner}の勝ち({you-ai})！"))
                    line_bot_api.reply_message(event.reply_token, send_list)
                    return 0  # [SEND]
                # [IF-no] is gameover?
                else:
                    # show board
                    #############################
                    ai = int(get_v(board,result_net)*100)
                    you = 100 - ai
                    # eval_text = f"評価値：{q[move_from_str(message)]*100:.1f}"
                    # eval_text += f"\nＭＡＸ：{q[legal_moves].max()*100:.1f}"
                    # eval_text += f"\nＭＩＮ：{q[legal_moves].min()*100:.1f}"
                    # if q[move_from_str(message)] == q[legal_moves].min():
                    #     eval_text = "う～ん…。\n" + eval_text
                    # elif q[move_from_str(message)] == q[legal_moves].max():
                    #     eval_text = "ベストな手！\n" + eval_text
                    # elif q[move_from_str(message)] >= np.median(q[legal_moves]):
                    #     eval_text = "その調子！\n" + eval_text
                    # else:
                    #     eval_text = "もっとがんばれ！\n" + eval_text
                    # send_list.append(TextSendMessage(text=eval_text))
                    send_list.append(TextSendMessage(text=f"AI {ai}%       あなた {you}%\n{'|'*round(ai/2)} {'|'*round(you/2)}"))
                    renderPM.drawToFile(svg2rlg(io.StringIO(board.to_svg())), f"{img_save_dir}image_{userId}.jpg", fmt="JPEG")
                    send_list.append(ImageSendMessage(f"{img_dir}image_{userId}.jpg", f"{img_dir}image_{userId}.jpg"))
                    send_list.append(TextSendMessage(text=f"私は{move_to_str(move)}に置きました。\n次はあなたの番(白○)です。\n(置ける場所がないときは「パス」と言ってください)"))
                    line_bot_api.reply_message(event.reply_token, send_list)
                    return 0  # [SEND]

    # [RECEIVE] invalid input : translation
    #############################
    else:
        thank_you = ["((っ´；ω；)っｱﾘｶﾞﾄｳ…｡+ﾟ",
                     "(❁´ω`❁)ｱﾘｶﾞﾄｳｺﾞｻﾞｲﾏｽ",
                     "(*´ｰ`*人)ｱﾘｶﾞﾀﾔｰ",
                     " 'ω')ｱｻﾞｽ",
                     "感謝(*´ω｀人)感謝",
                     "(⁎˃ᴗ˂⁎) thank you♡",
                     "ᵗʱᵃᵑᵏᵧₒᵤ⸜(*ˊᵕˋ*)⸝ﾜｰｲ",
                     "(*ﾉ>ᴗ<)ﾃﾍｯ"
                    ]
        birthday_words = ["誕生","おめでとう","HPB","birthday","Birthday","ハッピーバースデー","ハッピーバースデイ","ハピバ"]
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        if ((now.month, now.day)) == (4,3) and any(word in message for word in birthday_words):
            send_list.append(TextSendMessage(text=random.choice(thank_you)))
            line_bot_api.reply_message(event.reply_token, send_list)
            return 0 # [SEND]

        if "大機" in message:
            message = message.replace("大機","だいき")
        try:
            translator = Translator()
            if message[:3] == "to:":
                translated_text = translator.translate(message[5:], dest=message[3:5]).text
            elif message[:7] == "detect:":
                detected = translator.detect(message[7:]).lang
                with open(f"{json_dir}languages.json", "r") as f:
                    lang_dict = json.load(f)
                translated_text = f"言語：{lang_dict[detected]}\n言語コード：{detected}"
            else:
                translated_text = translator.translate(message, dest="en").text
            send_list.append(TextSendMessage(text=translated_text))
        except Exception:
            send_list.append(TextSendMessage(text=f"すみません、翻訳できません。"))
        line_bot_api.reply_message(event.reply_token, send_list)
        return 0  # [SEND]

if __name__ == "__main__":
    app.run()
from flask import Flask, request, abort, render_template
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
from googletrans import Translator
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import os
import pickle
import json
import io
import datetime
import random
import torch
import numpy as np
from network import PolicyNetwork, ValueNetwork
from functions import *
from creversi import *

app = Flask(__name__)

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
# value_net_state_dict = torch.load(f"{model_dir}value-network-v2.pth")
# policy_net_state_dict = torch.load(f"{model_dir}policy-network-v3.pth")
# value_net = ValueNetwork()
# policy_net = PolicyNetwork()
# value_net.load_state_dict(value_net_state_dict)
# policy_net.load_state_dict(policy_net_state_dict)
# torch.save(value_net,f"{model_dir}LOCAL-value-network-v2.pth")
# torch.save(policy_net,f"{model_dir}LOCAL-policy-network-v3.pth")

# load models
value_net = torch.load(f"{model_dir}LOCAL-value-network-v2.pth")
policy_net = torch.load(f"{model_dir}LOCAL-policy-network-v3.pth")


# read ACCESS_TOKEN, SECRET, CALLBACK_URL from json file
with open(f"{json_dir}token.json", "r") as f:
    tokens = json.load(f)
ACCESS_TOKEN = tokens["ACCESS_TOKEN"]
SECRET = tokens["SECRET"]
CALLBACK_URL = tokens["CALLBACK_URL"]

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
    image_list = []
    userId_list = []
    AI_list = []
    for filename in os.listdir(img_save_dir):
        if filename.endswith('.jpg'):
            image_list.append(os.path.join('https://matsudatkm.pythonanywhere.com/static/', filename))
            userId_list.append(log_board_dir + filename[:-4].replace("image_","BOARD_") + ".pkl")
    for user in userId_list:
        with open(user, "rb") as f:
            _,_,ai_type = pickle.load(f)
            AI_list.append(ai_type)
    return render_template('board_gallery.html', image_list=image_list, ai_type_list=AI_list)

# endpoint from linebot
@app.route("/"+CALLBACK_URL, methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

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
        move = get_move(board, policy_net)
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
                move = get_move(board, policy_net)
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
                    ai = int(32+get_v(board,value_net)*32)
                    you = 64 - ai
                    send_list.append(TextSendMessage(text=f"【予想最終得点】\nAI {ai}点       あなた {you}点\n{'|'*round(ai/1.5)} {'|'*round(you/1.5)}"))
                    renderPM.drawToFile(svg2rlg(io.StringIO(board.to_svg())), f"{img_save_dir}image_{userId}.jpg", fmt="JPEG")
                    send_list.append(ImageSendMessage(f"{img_dir}image_{userId}.jpg", f"{img_dir}image_{userId}.jpg"))
                    send_list.append(TextSendMessage(text=f"私は{move_to_str(move)}に置きました。\n次はあなたの番(白○)です。\n(置ける場所がないときは「パス」と言ってください)"))
                    line_bot_api.reply_message(event.reply_token, send_list)
                    return 0  # [SEND]

    # [RECEIVE] invalid input : translation
    #############################
    else:
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
from flask import Flask, request, abort, render_template
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
from googletrans import Translator
import os
import pickle
import json
import torch
from network import PolicyNetwork, ValueNetwork
from master import *
from fileio import *
from creversi import *

app = Flask(__name__)

class CONFIG:
    json = './mysite/'
    model = './mysite/trained_models/'
    message = './mysite/log/message/'
    move = './mysite/log/move/'
    master = './mysite/db/'
    image = './mysite/static/'
    static = 'https://matsudatkm.pythonanywhere.com/static/'

# moveをまとめる
for i,filename in enumerate(os.listdir(CONFIG.move)):
    if filename != 'MOVE_ALL.txt':
        with open(f'{CONFIG.move}{filename}','r') as f:
            text = f.read()
        with open(f'{CONFIG.move}MOVE_ALL.txt', 'w' if i==0 else 'a') as f:
            f.write(text)

## save models(only first time)
# value_net_state_dict = torch.load(f'{CONFIG.model}value-network-v4.pth')
# policy_net_state_dict = torch.load(f'{CONFIG.model}policy-network-v3.pth')
# model_v = ValueNetwork()
# model_p = PolicyNetwork()
# model_v.load_state_dict(value_net_state_dict)
# model_p.load_state_dict(policy_net_state_dict)
# torch.save(model_v, f'{CONFIG.model}LOCAL-value-network-v4.pth')
# torch.save(model_p, f'{CONFIG.model}LOCAL-policy-network-v3.pth')

# load models
model_v = torch.load(f'{CONFIG.model}LOCAL-value-network-v4.pth')
model_p = torch.load(f'{CONFIG.model}LOCAL-policy-network-v3.pth')


# read ACCESS_TOKEN, SECRET, CALLBACK_URL from json file
with open(f'{CONFIG.json}token.json', 'r') as f:
    tokens = json.load(f)
ACCESS_TOKEN = tokens['ACCESS_TOKEN']
SECRET = tokens['SECRET']
CALLBACK_URL = tokens['CALLBACK_URL']

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
    mode_list = []
    for filename in os.listdir(CONFIG.image):
        if filename.endswith('.jpg'):
            image_list.append(CONFIG.static + filename)
            userId_list.append(CONFIG.master + filename[:-4].replace("image_","MASTER_") + ".pkl")
    for user in userId_list:
        with open(user, "rb") as f:
            master_ = pickle.load(f)
        mode_list.append(master_.mode)
    return render_template('board_gallery.html', image_list=image_list, mode_list=mode_list)

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

# MAIN
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = event.message.text
    userId = event.source.user_id
    send_list = []

    save_message(message, f'{CONFIG.message}MESSAGE_{userId}.txt')

    # [CASE-1] select mode
    if message in ["【手加減AI】とオセロを始める", "【強いAI】とオセロを始める"]:
        mode = "adjust" if message=="【手加減AI】とオセロを始める" else "strong"
        master = Master(mode)
        
        master.save_image(f'{CONFIG.image}image_{userId}.jpg')
        master.save(f'{CONFIG.master}MASTER_{userId}.pkl')

        send_list.append(TextSendMessage(text='あなたは黒●で、私は白○です。\n置きたい場所を「a1」～「h8」から選んでください。'))
        send_list.append(ImageSendMessage(f'{CONFIG.static}image_{userId}.jpg', f'{CONFIG.static}image_{userId}.jpg'))
        line_bot_api.reply_message(event.reply_token, send_list)
        
        return 0

    # [CASE-2] input move
    #############################
    else:
        master = Master('strong')
        try:
            master.load(f'{CONFIG.master}MASTER_{userId}.pkl')
            move = master.input_move(message)
        except:
            send_list.append(TextSendMessage(text='エラー：対戦を始めてください。'))
            line_bot_api.reply_message(event.reply_token, send_list)
            return 0

        if move is not None:
            
            if master.is_legal(move):
                master.move(move)
            else:
                send_list.append(TextSendMessage(text='そこには置けません！'))
                line_bot_api.reply_message(event.reply_token, send_list)
                return 0
            
            if master.board.is_game_over():
                master.save_image(f'{CONFIG.image}image_{userId}.jpg')
                master.save_moves(f'{CONFIG.move}MOVE_{userId}.txt')
                send_list.append(TextSendMessage(text=f'あなたは「{move}」に置きました。ゲーム終了です。'))
                send_list.append(ImageSendMessage(f'{CONFIG.static}image_{userId}.jpg', f'{CONFIG.static}image_{userId}.jpg'))
                black, white = master.count_result()
                master.save(f'{CONFIG.master}MASTER_{userId}.pkl')
                result = "あなたの勝ち" if black > white else ("AIの勝ち" if black < white else "引き分け")
                send_list.append(TextSendMessage(text=f'あなた(黒)：{black}個\nAI(白)：{white}個\nで{result}です！'))
                line_bot_api.reply_message(event.reply_token, send_list)
                return 0
            else:
                ai_move = master.get_move(model_p, model_v)
                assert master.is_legal(ai_move)
                master.move(ai_move)
                master.save_image(f'{CONFIG.image}image_{userId}.jpg')
                master.save_moves(f'{CONFIG.move}MOVE_{userId}.txt')
                if master.board.is_game_over():
                    send_list.append(TextSendMessage(text=f'あなたは「{move}」に置きました。ゲーム終了です。'))
                    send_list.append(ImageSendMessage(f'{CONFIG.static}image_{userId}.jpg', f'{CONFIG.static}image_{userId}.jpg'))
                    black, white = master.count_result()
                    master.save(f'{CONFIG.master}MASTER_{userId}.pkl')
                    result = "あなたの勝ち" if black > white else ("AIの勝ち" if black < white else "引き分け")
                    send_list.append(TextSendMessage(text=f'あなた(黒)：{black}個\nAI(白)：{white}個\nで{result}です！'))
                    line_bot_api.reply_message(event.reply_token, send_list)
                    return 0
                else:
                    send_list.append(TextSendMessage(text=f'あなたは「{move}」に置きました。私は「{ai_move}」に置きました。'))
                    send_list.append(ImageSendMessage(f'{CONFIG.static}image_{userId}.jpg', f'{CONFIG.static}image_{userId}.jpg'))
                    zmax, zmin = master.evaluate_board(model_v)
                    master.save(f'{CONFIG.master}MASTER_{userId}.pkl')
                    send_list.append(TextMessage(text=f'【予想最終得点差】\n(正の値→黒が優勢)\nMAX：{zmax:.0f}\nMIN：{zmin:.0f}'))
                    line_bot_api.reply_message(event.reply_token, send_list)
                    return 0

        # [CASE-3] other
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
                    with open(f"{CONFIG.json}languages.json", "r") as f:
                        lang_dict = json.load(f)
                    translated_text = f"言語：{lang_dict[detected]}\n言語コード：{detected}"
                else:
                    translated_text = translator.translate(message, dest="en").text
                send_list.append(TextSendMessage(text=translated_text))
            except Exception:
                send_list.append(TextSendMessage(text="すみません、翻訳できません。"))
            line_bot_api.reply_message(event.reply_token, send_list)
            return 0

if __name__ == "__main__":
    app.run()
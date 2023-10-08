import os
import datetime
import pickle

def save_message(message, path):
    """メッセージログ"""
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d-%H:%M:%S.%f")
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(now + ',' + message + '\n')
    else:
        with open(path, 'a') as f:
            f.write(now + ',' + message + '\n')




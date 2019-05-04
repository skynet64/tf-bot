import telebot
import sqlite3
import time
import bot_model
import numpy as np

from telebot import apihelper

db_name = 'bot.db'      # bot database
token = r'817106542:AAF7xon-Dwhcsm5SMZZ2NkbTyG70Qn2EOrI'    # bot token
default_profile = 0     # default configuratiin profile
time_limit = 1          # time in seconds after which chat hidden states will be stored on disk
echo_message = True     # echo received message in console

apihelper.proxy = {'https': 'socks5://78.47.60.141:1080'}
bot = telebot.TeleBot(token)


def load_configuration(profile=default_profile):
    """Initializes bot s2s network from selected profile.
    Returns instance of class Bot()."""

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    cur.execute('''
        SELECT vocab, embed, units, inp_size, out_size, tokenizer, checkpoint
        FROM profile
        WHERE id = {}
        LIMIT 1;'''.format(profile))
    profile = cur.fetchone()
    if profile:
        bot_nn = bot_model.Bot(profile[0], profile[1], profile[2], 1, profile[3], profile[4])
        bot_nn.load_model(profile[5], profile[6])
        cur.close()
        conn.close()
        return bot_nn
    else:
        print('Selected invalid profile: ' + str(profile))
        exit(-1)


def get_hidden(id, cur):
    """Returns hidden state either from database or initializes it with default value."""

    cur.execute('''
        SELECT file
        FROM chats
        WHERE id = {}
        LIMIT 1;'''.format(id))
    file = cur.fetchone()
    if file:
        file = file[0]
        try:
            h = np.load(file)
        except Exception:
            return bot_nn.encoder.initialize_hidden_state()
        else:
            return bot_model.tf.convert_to_tensor(h)
    else:
        return bot_nn.encoder.initialize_hidden_state()


def collect_old_hidden(conn, cur):
    """Removes all outdated hidden states from hidden() dictionary and saves them on disk."""

    deleted = []
    for key, value in hidden.items():
        if time.time() - value[1] > time_limit:
            cur.execute('''
                SELECT file
                FROM chats
                WHERE id = {}
                LIMIT 1;'''.format(key))
            file_name = cur.fetchone()
            if not file_name:
                file_name = './data/' + str(key) + '.npy'
                cur.execute('''INSERT INTO chats(id, file) VALUES
                               ({}, "{}");'''.format(key, file_name))
                conn.commit()
            else:
                file_name = file_name[0]
            arr = value[0].numpy()
            np.save(file_name, arr)
            deleted.append(key)
            print('Chat {} state collected'.format(key))
    for key in deleted:
        del hidden[key]


bot_nn = load_configuration()
last_cleaned = time.time()
hidden = {}


@bot.message_handler(commands=['start'])
def start(message):
    """Welcome message."""

    bot.send_message(message.chat.id, 'Hi!')


@bot.message_handler(commands=['collect'])
def start(message):
    """Manual cleanup."""

    old_len = len(hidden)
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    collect_old_hidden(conn, cur)
    cur.close()
    conn.close()
    bot.send_message(message.chat.id, 'Done! Collected: ' + str(old_len - len(hidden)))


@bot.message_handler(content_types=['text'])
def answer(message):
    """Replies to a given message."""

    global last_cleaned, hidden, bot_nn
    if echo_message:
        print(message.chat.id, message.text)

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    if not (message.chat.id in hidden.keys()):
        hidden.update({message.chat.id: [get_hidden(message.chat.id, cur), time.time()]})

    try:
        output, hidden[message.chat.id][0] = bot_nn.evaluate(message.text, hidden[message.chat.id][0])
        hidden[message.chat.id][1] = time.time()
    except Exception:       # if message after preparation became empty
        pass
    else:
        bot.send_message(message.chat.id, output)

    if time.time() - last_cleaned > time_limit:         # perform cleanup
        collect_old_hidden(conn, cur)
        last_cleaned = time.time()
    cur.close()
    conn.close()


bot.polling(none_stop=True, interval=1)

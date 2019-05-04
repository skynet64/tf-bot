# Telegram chat bot on tensorflow

Created by Dmitry Ryabkov Group Б05-812

### File description

`reddit_download.py` - downloads reddit archived comments

`reddit_unpack.py` - upacks archived comments

`create_db_sqlite.py` - creates database based on comments

`prep_data.py` - prepares data from database to be used in training

`bot_model.py` - model definition and training loop

`bot.py` - telegram connectivity part

`bot.db` - example of bot database file

`config.zip` - contains checkpoint file for model

### Requirements

All these files were runned on python 3.6.8.

Installing all requirement packages:

`pip install tensorflow==2.0.0-alpha0 pyTelegramBotAPI pysocks`

`pysocks` only needed if you are going to use socks5 proxy server

### Launching

Steps:

1. Insert correct values at the beginning of file `bot.py`

2. Extract `config.zip` and create directory `data` near the scripts

3. Run `python3 bot.py`

Currect checkpoint is bad and will result in model repeating word 'the' (and token (__go__)).

### Database structure

#### Table `profile`

Structure:

`  id  |  vocab  |  embed  | units |  inp_size  |  out_size  | tokenizer | checkpoint  `

1. `id` - profile id

2. `vocab` - vocabulary size

3. `embed` - embedded layer size

4. `units` - encoder output dimension

5. `inp_size` - max input sequence (tokens)

6. `put_size` - max output sequence (tokens)

7. `tokenizer` - directory where `tokenizer.pickle` is stored

8. `checkpoint` - path to checkpoint

If you change parameters `2-4` you will have to re-train network.

#### Table `chats`

`  id  |  file  `

1. `id` - chat id

2. `file` - file where hidden state is saved

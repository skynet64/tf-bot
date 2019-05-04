from bot_model import Bot
import sqlite3


def get_chain(id, cur):
    """Returns all comments from chain(list of strings)."""

    cur.execute('''
        SELECT next_id, data
        FROM comments
        WHERE id = "{}";'''.format(id))
    id, data = cur.fetchone()
    output = [Bot.prep_sentence(data)]
    while id:
        cur.execute('''
            SELECT next_id, data
            FROM comments
            WHERE id = "{}";'''.format(id))
        id, data = cur.fetchone()
        output.append(Bot.prep_sentence(data))
    return output


def prepare_dataset():
    """Extracts comments from database and returns tuple of 2 lists of strings(input / output)."""

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    cur.execute('''
        SELECT id
        FROM chains;''')
    chains = cur.fetchall()
    inputs = []
    outputs = []
    rows = 0
    skip = False
    for chain in chains:
        rows += 1
        chain_data = get_chain(chain[0], cur)
        depth = len(chain_data) - len(chain_data) % 2       # only need even part of chain
        for i in range(depth):
            if i % 2 == 1:
                if not (chain_data[i] is None):             # if string after preparation became empty
                    inputs.append(chain_data[i])
                    skip = False
                else:
                    skip = True
            else:
                if not (chain_data[i] is None) and not skip:
                    outputs.append(chain_data[i])
                elif not skip:
                    inputs.pop()
        if rows % 1000 == 0:
            print('Processed chains: ' + str(rows))
    return inputs, outputs


if __name__ == '__main__':
    db_name = 'reddit.db'
    inp_seq_file = 'inp.txt'
    out_seq_file = 'out.txt'

    inputs, outputs = prepare_dataset()
    with open(inp_seq_file, 'w', encoding='utf-8') as file:
        for inp in inputs:
            file.write(inp + '\n')

    with open(out_seq_file, 'w', encoding='utf-8') as file:
        for out in outputs:
            file.write(out + '\n')

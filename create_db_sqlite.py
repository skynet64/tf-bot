import os
import sqlite3
import json
from functools import wraps


# Database configuration
db_name = 'reddit.db'

# Filter parameters for data
start_year = 2005
end_year = 2011
score_threshold = 10
max_tokens = 100
max_comment_length = 1000

# Logging parameters
prep_log_frequency = 10000
chain_log_frequency = 500

buffer_limit = 1000


def fix_params(*fargs, **fkwargs):
    """Decorator that fixes parameters for give function."""

    def params(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.update(fkwargs)
            return func(*fargs, *args, **kwargs)
        return wrapper
    return params


def file_names(begin, end):
    """Yields names in format YYYY-MM."""

    for year in range(begin, end):
        for month in range(1, 13):
            file_name = str(year) + '-'
            if month < 10:
                file_name += '0'
            file_name += str(month)
            yield file_name


def create_table_comments(cur):
    """Creates table with comments."""

    cur.execute('''
        CREATE TABLE IF NOT EXISTS "comments"(
        id TEXT PRIMARY KEY,
        next_id TEXT UNIQUE,
        subreddit TEXT,
        score INT,
        data TEXT);''')
    cur.execute('''
        CREATE UNIQUE INDEX t_comments_id_idx ON comments (id);''')
    cur.execute('''
        CREATE UNIQUE INDEX t_comments_next_id_idx ON comments (next_id);''')


def create_table_alt_comments(cur):
    # Needed because sqlite doesn't support ALTER TABLE DROP COLUMN.

    cur.execute('''
        CREATE TABLE IF NOT EXISTS "alt_comments"(
        id TEXT PRIMARY KEY,
        next_id TEXT UNIQUE,
        data TEXT);''')
    cur.execute('''
        CREATE UNIQUE INDEX comments_id_idx ON alt_comments (id);''')
    cur.execute('''
        CREATE UNIQUE INDEX comments_next_id_idx ON alt_comments (next_id);''')


def create_table_chains(cur):
    """Creates table for storing chain top comments."""

    cur.execute('''
        CREATE TABLE IF NOT EXISTS "chains"(
        id TEXT PRIMARY KEY,
        subreddit TEXT,
        depth INT);''')
    cur.execute('''
        CREATE UNIQUE INDEX t_chains_id_idx ON chains (id);''')


def create_table_alt_chains(cur):
    # Needed because sqlite doesn't support ALTER TABLE ADD FOREIGN KEY.

    cur.execute('''
        CREATE TABLE IF NOT EXISTS "alt_chains"(
        id TEXT PRIMARY KEY,
        subreddit TEXT,
        depth INT,
        FOREIGN KEY (id) REFERENCES comments(id));''')
    cur.execute('''
        CREATE UNIQUE INDEX chains_id_idx ON alt_chains (id);''')


def execute_query(conn, cur, buffer):
    """Executes query for given buffer."""

    cur.execute('BEGIN TRANSACTION')
    for request in buffer:
        try:
            cur.execute(request)
        except Exception as e:
            pass
    conn.commit()
    buffer.clear()


def append_query(conn, cur, buffer, limit, request):
    """Fills buffer with requests and calls execute_query()."""

    buffer.append(request)
    if len(buffer) > limit:
        execute_query(conn, cur, buffer)


def get_id(file_name, line):
    """Extracts comment_id from json."""
    # Reddit differently stored comment_id in json format in different years.

    if file_name < '2007-11':
        return 't1_' + line['id']
    elif file_name < '2017-11':
        return line['name']
    else:
        return line['link_id']


def get_child_of_parent(parent_id, cur):
    """Finds child score for given parent_id."""

    cur.execute('''
        SELECT score
        FROM comments
        WHERE next_id = "{}";'''.format(parent_id))
    return cur.fetchone()


def filter_data(data):
    """Filters data."""

    if len(data.split(' ')) > max_tokens or len(data) < 1:
        return False
    elif len(data) > max_comment_length:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return data.replace('\r', '').replace('\n\n', ' __line__ ').replace('\n', ' __line__ ').replace('"', '\'')


def prep_comments(conn):
    """Inserts comments from json files to database."""

    cur = conn.cursor()
    create_table_comments(cur)
    conn.commit()
    buffer = []
    query = fix_params(
        conn,
        cur,
        buffer,
        buffer_limit)(append_query)
    rows = 0
    for file_name in file_names(start_year, end_year):
        file_name += '.json'
        if os.path.isfile(file_name):
            print('Opening file: ' + file_name)
            with open(file_name, 'r') as file:
                for line in file:
                    line = json.loads(line)
                    rows += 1
                    score = line['score']
                    if score > score_threshold:
                        c_id = get_id(file_name, line)
                        parent_id = line['parent_id']
                        subreddit = line['subreddit']
                        data = filter_data(line['body'])
                        if parent_id[:3] == 't3_':              # grant unique to top comments (t3 - thread_id,
                            parent_id = 't3_' + str(rows)       # t1 - comment_id)
                        if data:
                            old_child = get_child_of_parent(parent_id, cur)
                            if old_child:                       # update already existing comment in database
                                if old_child[0] < score:
                                    query('''
                                        UPDATE comments SET
                                        id = "{0}",
                                        next_id = "{1}",
                                        subreddit = "{2}",
                                        score = {3},
                                        data = "{4}"
                                        WHERE next_id = "{1}";'''.format(c_id, parent_id, subreddit, score, data))
                            else:
                                query('''
                                    INSERT INTO comments(id, next_id, subreddit, score, data) VALUES
                                    ("{}", "{}", "{}", {}, "{}");'''.format(c_id, parent_id, subreddit, score, data))
                    if rows % prep_log_frequency == 0:
                        print('Comments processed: ' + str(rows))
    execute_query(conn, cur, buffer)
    cur.close()


def get_top_comments(cur):
    """Returs all top(bottom) comments."""
    # Bottom because chains reversed

    cur.execute('''
        SELECT id, next_id
        FROM comments
        WHERE id NOT IN
        (SELECT next_id
        FROM comments);''')
    return cur.fetchall()


def get_next_comment(id, cur):
    """Returns next comment for given comment."""

    cur.execute('''
        SELECT id, next_id
        FROM comments
        WHERE id = "{}";'''.format(id[1]))
    return cur.fetchone()


def get_comment_subreddit(id, cur):
    """Returns subreddit for given comments."""

    cur.execute('''
        SELECT subreddit
        FROM comments
        WHERE id = "{}";'''.format(id))
    return cur.fetchone()


def build_chains(conn):
    """Reverses all chains and creates chain table."""

    cur = conn.cursor()
    create_table_chains(cur)
    conn.commit()
    buffer = []
    query = fix_params(
        conn,
        cur,
        buffer,
        buffer_limit)(append_query)
    top_comments = get_top_comments(cur)
    rows = 0
    for top_comment in top_comments:        # reverse chains like connected lists
        rows += 1
        depth = 1
        previous = 0
        current = top_comment
        next = get_next_comment(current, cur)
        query('''
            UPDATE comments SET
            next_id = NULL
            WHERE id = "{}";'''.format(current[0]))
        while next:
            depth += 1
            previous = current
            current = next
            next = get_next_comment(current, cur)
            query('''
                UPDATE comments SET
                next_id = "{}"
                WHERE id = "{}";'''.format(previous[0], current[0]))
        if previous:
            query('''
                UPDATE comments SET
                next_id = "{}"
                WHERE id = "{}";'''.format(previous[0], current[0]))
        query('''
            INSERT INTO chains(id, subreddit, depth) VALUES
            ("{}", "{}", {});'''.format(current[0], get_comment_subreddit(current[0], cur)[0], depth))
        if rows % chain_log_frequency == 0:
            print('Chains created ' + str(rows))
    execute_query(conn, cur, buffer)
    cur.close()


def cleanup(conn):
    """Removes chains with depth <= 1,
    drops useless columns from comments table."""

    cur = conn.cursor()
    create_table_alt_comments(cur)
    cur.executescript('''
        DELETE FROM comments
        WHERE id IN
        (SELECT id
        FROM chains
        WHERE depth <= 1);
        
        DELETE FROM chains
        WHERE depth <= 1;
        
        INSERT INTO alt_comments
        SELECT id, next_id, data FROM comments;
        
        DROP TABLE IF EXISTS comments;
        
        ALTER TABLE alt_comments RENAME TO comments;''')
    create_table_alt_chains(cur)
    cur.executescript('''
        INSERT INTO alt_chains
        SELECT id, subreddit, depth FROM chains;
        
        DROP TABLE IF EXISTS chains;
        
        ALTER TABLE alt_chains RENAME TO chains;
        
        VACUUM;''')
    conn.commit()
    cur.close()


if __name__ == '__main__':
    conn = sqlite3.connect(db_name)
    print('Preparing comments')
    prep_comments(conn)
    print('Building chains')
    build_chains(conn)
    print('Cleaning up data')
    cleanup(conn)

    cur = conn.cursor()
    cur.execute('''SELECT count(id) FROM chains;''')
    chains = cur.fetchone()[0]
    cur.execute('''SELECT count(id) FROM comments;''')
    comments = cur.fetchone()[0]
    print('Done! Chains: {}, Comments: {}'.format(chains, comments))

    cur.close()
    conn.close()

import sqlite3


if __name__ == "__main__":
    conn = sqlite3.connect("database.sqlite")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE `post_reports` (\
    id            INTEGER PRIMARY KEY AUTOINCREMENT,\
    post_id       INTEGER NOT NULL,\
    reporter_id       INTEGER NOT NULL);")
    conn.commit()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("Existing tables in database:")
    for table in tables:
        print(table[0])

    conn.close()
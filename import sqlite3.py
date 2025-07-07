import sqlite3


def is_valid_bcrypt(hash_string):
    # Valid bcrypt hashes usually start with $2b$, $2a$, or $2y$, and are 60 chars long
    return (
        isinstance(hash_string, str)
        and len(hash_string) == 60
        and hash_string.startswith(("$2a$", "$2b$", "$2y$"))
    )


db_path = "cvmanagizer.db"  # Update if your DB file is elsewhere

with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, password FROM users")
    all_users = cursor.fetchall()
    invalid_user_ids = [uid for uid, email, pw in all_users if not is_valid_bcrypt(pw)]
    if invalid_user_ids:
        print(f"Deleting users with IDs: {invalid_user_ids}")
        cursor.executemany(
            "DELETE FROM users WHERE id = ?", [(uid,) for uid in invalid_user_ids]
        )
        conn.commit()
    else:
        print("No invalid user hashes found.")

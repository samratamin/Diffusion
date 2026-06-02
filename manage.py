#!/usr/bin/env python3
"""
Admin management CLI for NMR Diffusion Analysis.

Usage:
    python manage.py set-password    # Set or change the admin panel password
    python manage.py status          # Check whether a password has been configured
"""
import sqlite3
import hashlib
import secrets
import getpass
import sys
import os

DB_FILE = os.path.join(os.path.dirname(__file__), 'nmr_diffusion.db')


def _get_setting(key):
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute("SELECT value FROM admin_settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def _set_setting(key, value):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("CREATE TABLE IF NOT EXISTS admin_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT OR REPLACE INTO admin_settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(32)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return h, salt


def cmd_set_password():
    """Interactively set or change the admin panel password."""
    if not os.path.exists(DB_FILE):
        print(f"Error: database not found at {DB_FILE}")
        print("Start the app at least once to initialise the database, then re-run this.")
        sys.exit(1)

    print("Set Admin Password")
    print("=" * 40)
    while True:
        password = getpass.getpass("New password: ")
        if len(password) < 8:
            print("Password must be at least 8 characters. Try again.\n")
            continue
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match. Try again.\n")
            continue
        break

    hashed, salt = _hash_password(password)
    _set_setting('password_hash', hashed)
    _set_setting('password_salt', salt)
    print("\nAdmin password updated successfully.")
    print("You can now log in at /admin/login")


def cmd_status():
    """Print whether an admin password is currently configured."""
    if not os.path.exists(DB_FILE):
        print("Database not found — app has not been started yet.")
        sys.exit(1)
    has_password = bool(_get_setting('password_hash'))
    print(f"Admin password configured: {'yes' if has_password else 'NO — run: python manage.py set-password'}")


COMMANDS = {
    'set-password': cmd_set_password,
    'status':       cmd_status,
}

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'set-password'
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available commands: {', '.join(COMMANDS)}")
        sys.exit(1)
    COMMANDS[cmd]()

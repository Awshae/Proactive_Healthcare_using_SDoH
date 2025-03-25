from cryptography.fernet import Fernet
import hashlib

def generate_key():
    return Fernet.generate_key()

def encrypt_data(data, key):
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data, key):
    f = Fernet(key)
    return f.decrypt(encrypted_data.encode()).decode()

def hash_data(data):
    return hashlib.sha256(data.encode()).hexdigest()

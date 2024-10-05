import random
import re
import json
import datetime
import math
import string
import requests
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import sqlite3
import os
from textblob import TextBlob  # Untuk analisis sentimen dan NLP
from sklearn.feature_extraction.text import TfidfVectorizer  # Untuk analisis teks
from sklearn.metrics.pairwise import cosine_similarity  # Untuk mencari kesamaan teks

class AdvancedAISystem:
    def __init__(self):
        self.name = "Advanced AI System"
        self.version = "2.0"
        self.initialize_system()
        
    def initialize_system(self):
        """Inisialisasi semua komponen sistem"""
        # Inisialisasi database
        self.setup_database()
        
        # Inisialisasi model NLP
        self.vectorizer = TfidfVectorizer()
        self.response_vectors = None
        self.load_knowledge_base()
        
        # Inisialisasi memori konteks
        self.conversation_history = []
        self.max_history = 10
        
        # Inisialisasi pembelajaran
        self.learning_data = defaultdict(list)
        
    def setup_database(self):
        """Setup database SQLite untuk menyimpan data"""
        self.conn = sqlite3.connect('ai_system.db')
        self.cursor = self.conn.cursor()
        
        # Buat tabel-tabel yang diperlukan
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             user_input TEXT,
             ai_response TEXT,
             timestamp DATETIME,
             sentiment REAL)
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_responses
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             pattern TEXT,
             response TEXT,
             usage_count INTEGER)
        ''')
        
        self.conn.commit()

    def load_knowledge_base(self):
        """Muat basis pengetahuan dari file JSON"""
        try:
            with open('knowledge_base.json', 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
        except FileNotFoundError:
            # Buat basis pengetahuan default jika file tidak ada
            self.knowledge_base = {
                "general_responses": {
                    "greetings": [
                        "Halo! Saya AI Assistant yang canggih. Apa yang bisa saya bantu?",
                        "Selamat datang! Saya siap membantu dengan berbagai tugas.",
                        "Hi! Mari kita mulai percakapan yang produktif."
                    ],
                    "farewells": [
                        "Sampai jumpa! Terima kasih atas percakapannya.",
                        "Selamat tinggal! Senang bisa membantu Anda.",
                        "Bye! Semoga hari Anda menyenangkan!"
                    ]
                },
                "topics": {},
                "commands": {
                    "/help": "Menampilkan bantuan",
                    "/analyze": "Menganalisis teks",
                    "/learn": "Menambah pengetahuan baru",
                    "/stats": "Menampilkan statistik sistem",
                    "/clear": "Menghapus history percakapan"
                }
            }
            self.save_knowledge_base()

    def save_knowledge_base(self):
        """Simpan basis pengetahuan ke file"""
        with open('knowledge_base.json', 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=4, ensure_ascii=False)

    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analisis sentimen menggunakan TextBlob"""
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        
        if sentiment_score > 0:
            sentiment = "positif"
        elif sentiment_score < 0:
            sentiment = "negatif"
        else:
            sentiment = "netral"
            
        return sentiment_score, sentiment

    def analyze_text_advanced(self, text: str) -> Dict[str, Any]:
        """Analisis teks lengkap"""
        analysis = TextBlob(text)
        words = text.split()
        
        return {
            "word_count": len(words),
            "char_count": len(text),
            "sentence_count": len(analysis.sentences),
            "sentiment": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity,
            "language": analysis.detect_language(),
            "key_phrases": analysis.noun_phrases,
            "longest_word": max(words, key=len) if words else "",
            "unique_words": len(set(words))
        }

    def learn_from_interaction(self, user_input: str, response: str, success: bool):
        """Pembelajaran dari interaksi"""
        if success:
            # Simpan pola yang berhasil
            self.cursor.execute('''
                INSERT OR REPLACE INTO learned_responses (pattern, response, usage_count)
                VALUES (?, ?, COALESCE((SELECT usage_count + 1 FROM learned_responses 
                WHERE pattern = ?), 1))
            ''', (user_input, response, user_input))
            self.conn.commit()

    def generate_creative_response(self, topic: str) -> str:
        """Menghasilkan respons kreatif berdasarkan topik"""
        # Implementasi sederhana dari creative text generation
        templates = [
            f"Mari kita eksplorasi {topic} bersama-sama!",
            f"Ada banyak hal menarik tentang {topic} yang bisa kita bahas.",
            f"Saya punya beberapa ide menarik tentang {topic}."
        ]
        return random.choice(templates)

    def process_command(self, command: str) -> str:
        """Memproses perintah khusus dengan lebih banyak fitur"""
        parts = command.split()
        main_command = parts[0].lower()

        if main_command == "/help":
            return self.get_help_message()
            
        elif main_command == "/analyze":
            text = " ".join(parts[1:])
            analysis = self.analyze_text_advanced(text)
            return json.dumps(analysis, indent=2, ensure_ascii=False)
            
        elif main_command == "/learn":
            if len(parts) >= 3:
                pattern = parts[1]
                response = " ".join(parts[2:])
                self.add_to_knowledge_base(pattern, response)
                return f"Telah mempelajari respons baru untuk pola: {pattern}"
            
        elif main_command == "/stats":
            return self.get_system_stats()
            
        elif main_command == "/clear":
            self.conversation_history.clear()
            return "History percakapan telah dihapus."
            
        return "Perintah tidak dikenali. Gunakan /help untuk bantuan."

    def get_system_stats(self) -> str:
        """Mendapatkan statistik sistem"""
        self.cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM learned_responses")
        total_learned = self.cursor.fetchone()[0]
        
        stats = f"""Statistik Sistem:
- Total percakapan: {total_conversations}
- Pola yang dipelajari: {total_learned}
- Versi sistem: {self.version}
- Ukuran basis pengetahuan: {len(self.knowledge_base['topics'])} topik
- Memory terpakai: {self.get_memory_usage()} MB"""
        
        return stats

    def get_memory_usage(self) -> float:
        """Mendapatkan penggunaan memori dalam MB"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def add_to_knowledge_base(self, pattern: str, response: str):
        """Menambah pengetahuan baru ke basis pengetahuan"""
        if "topics" not in self.knowledge_base:
            self.knowledge_base["topics"] = {}
            
        if pattern not in self.knowledge_base["topics"]:
            self.knowledge_base["topics"][pattern] = []
            
        self.knowledge_base["topics"][pattern].append(response)
        self.save_knowledge_base()

    def get_weather(self, city: str) -> str:
        """Mendapatkan informasi cuaca (contoh integrasi API)"""
        # Gunakan API cuaca (perlu API key)
        API_KEY = "your_api_key"  # Ganti dengan API key Anda
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code == 200:
                temp = data['main']['temp']
                weather = data['weather'][0]['description']
                return f"Cuaca di {city}: {weather}, Suhu: {temp}°C"
            else:
                return "Maaf, tidak bisa mendapatkan informasi cuaca saat ini."
        except:
            return "Terjadi kesalahan saat mengakses layanan cuaca."

    def handle_mathematical_expression(self, expression: str) -> str:
        """Menangani ekspresi matematika"""
        try:
            # Bersihkan input
            clean_expr = expression.replace("hitung", "").replace("berapa", "").strip()
            # Evaluasi ekspresi
            result = eval(clean_expr)
            return f"Hasil perhitungan {clean_expr} = {result}"
        except:
            return "Maaf, saya tidak bisa melakukan perhitungan tersebut."

    def get_most_similar_response(self, user_input: str) -> str:
        """Mendapatkan respons yang paling mirip dari basis pengetahuan"""
        if not self.response_vectors:
            # Priproses semua respons yang ada
            responses = []
            for topic_responses in self.knowledge_base["topics"].values():
                responses.extend(topic_responses)
            
            if not responses:
                return None
                
            self.responses = responses
            self.response_vectors = self.vectorizer.fit_transform(responses)
            
        # Vectorize input user
        input_vector = self.vectorizer.transform([user_input])
        
        # Hitung kesamaan
        similarities = cosine_similarity(input_vector, self.response_vectors)
        most_similar_idx = similarities.argmax()
        
        if similarities[0][most_similar_idx] > 0.3:  # threshold kesamaan
            return self.responses[most_similar_idx]
        return None

    def get_response(self, user_input: str) -> str:
        """Fungsi utama untuk menghasilkan respons"""
        # Analisis input
        sentiment_score, sentiment = self.analyze_sentiment(user_input)
        
        # Simpan ke database
        self.cursor.execute('''
            INSERT INTO conversations (user_input, timestamp, sentiment)
            VALUES (?, datetime('now'), ?)
        ''', (user_input, sentiment_score))
        self.conn.commit()
        
        # Proses perintah khusus
        if user_input.startswith('/'):
            response = self.process_command(user_input)
            self.save_conversation(user_input, response, sentiment_score)
            return response

        # Cek untuk perhitungan matematika
        if any(keyword in user_input.lower() for keyword in ["hitung", "berapa", "+", "-", "*", "/"]):
            response = self.handle_mathematical_expression(user_input)
            self.save_conversation(user_input, response, sentiment_score)
            return response

        # Cek untuk pertanyaan cuaca
        if "cuaca" in user_input.lower():
            # Ekstrak nama kota (implementasi sederhana)
            words = user_input.split()
            try:
                city_index = words.index("cuaca") + 1
                city = words[city_index]
                response = self.get_weather(city)
                self.save_conversation(user_input, response, sentiment_score)
                return response
            except:
                pass

        # Cari respons yang mirip
        similar_response = self.get_most_similar_response(user_input)
        if similar_response:
            self.save_conversation(user_input, similar_response, sentiment_score)
            return similar_response

        # Generate respons kreatif jika tidak ada yang cocok
        response = self.generate_creative_response(user_input)
        self.save_conversation(user_input, response, sentiment_score)
        return response

    def save_conversation(self, user_input: str, response: str, sentiment: float):
        """Simpan percakapan ke database"""
        self.cursor.execute('''
            UPDATE conversations 
            SET ai_response = ?
            WHERE user_input = ? AND ai_response IS NULL
        ''', (response, user_input))
        self.conn.commit()
        
        # Tambahkan ke history
        self.conversation_history.append({
            "user": user_input,
            "ai": response,
            "sentiment": sentiment,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Batasi ukuran history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

def main():
    """Fungsi utama untuk menjalankan AI System"""
    ai = AdvancedAISystem()
    print(f"{ai.name} v{ai.version}")
    print("Ketik /help untuk melihat daftar perintah yang tersedia.")
    
    while True:
        try:
            user_input = input("\nAnda: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'selamat tinggal']:
                print(f"\n{ai.name}: Sampai jumpa! Semoga harimu menyenangkan!")
                break
                
            response = ai.get_response(user_input)
            print(f"\n{ai.name}:", response)
            
        except KeyboardInterrupt:
            print("\nProgram dihentikan oleh user.")
            break
        except Exception as e:
            print(f"\nTerjadi kesalahan: {str(e)}")
            continue

if __name__ == "__main__":
    main()

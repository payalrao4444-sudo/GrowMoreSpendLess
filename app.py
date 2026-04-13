"""
AI Smart Kitchen Gardening System - app.py v4.0
PostgreSQL + CNN + All Fixes
"""
import os, json, hashlib, secrets, random
from datetime import datetime, timedelta
from contextlib import contextmanager
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import requests
from PIL import Image, ImageStat

try:
    import psycopg2, psycopg2.extras
    USING_POSTGRES = True
except ImportError:
    import sqlite3
    USING_POSTGRES = False

try:
    import numpy as np
    import tensorflow as tf
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'kg_secret_2024_prod')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DATABASE_URL = os.environ.get('DATABASE_URL', '')
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
SQLITE_PATH = 'instance/garden.db'

# ---------------------------------------------------------------------------
# DATABASE ABSTRACTION
# ---------------------------------------------------------------------------
@contextmanager
def get_db():
    if USING_POSTGRES and DATABASE_URL:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    else:
        os.makedirs('instance', exist_ok=True)
        import sqlite3 as sq
        conn = sq.connect(SQLITE_PATH)
        conn.row_factory = sq.Row
    try:
        cur = conn.cursor()
        yield conn, cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def ph():
    return '%s' if (USING_POSTGRES and DATABASE_URL) else '?'

def db_exec(sql, params=(), one=False, many=False):
    if not (USING_POSTGRES and DATABASE_URL):
        sql = sql.replace('%s', '?')
    with get_db() as (conn, cur):
        cur.execute(sql, params)
        if one:
            row = cur.fetchone()
            return dict(row) if row else None
        if many:
            return [dict(r) for r in cur.fetchall()]
    return None

def safe_exec(sql, params=()):
    try: db_exec(sql, params)
    except Exception: pass

# ---------------------------------------------------------------------------
# DATABASE INIT
# ---------------------------------------------------------------------------
def init_db():
    is_pg = USING_POSTGRES and DATABASE_URL
    if is_pg:
        tables = [
            """CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY, username VARCHAR(80) UNIQUE NOT NULL,
                password VARCHAR(200) NOT NULL, location VARCHAR(120) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW())""",
            """CREATE TABLE IF NOT EXISTS crops (
                id SERIAL PRIMARY KEY, name VARCHAR(120) UNIQUE NOT NULL,
                category VARCHAR(80), soil_type VARCHAR(120), water_req VARCHAR(40),
                sunlight VARCHAR(80), season VARCHAR(40), temp_min INTEGER, temp_max INTEGER,
                container_size VARCHAR(40), days_to_harvest INTEGER,
                organic_fertilizer TEXT, companion_plants TEXT, avoid_plants TEXT,
                seeds_cost INTEGER, soil_cost INTEGER, fertilizer_cost INTEGER)""",
            """CREATE TABLE IF NOT EXISTS diseases (
                id SERIAL PRIMARY KEY, name VARCHAR(120) UNIQUE NOT NULL,
                affected_plants TEXT, symptoms TEXT, causes TEXT,
                organic_treatment TEXT, prevention TEXT, color_hint VARCHAR(5))""",
            """CREATE TABLE IF NOT EXISTS user_activity (
                id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id),
                activity_type VARCHAR(60), details JSONB, created_at TIMESTAMP DEFAULT NOW())""",
            """CREATE TABLE IF NOT EXISTS search_history (
                id SERIAL PRIMARY KEY, user_id INTEGER, crop_name VARCHAR(120) NOT NULL,
                results_found INTEGER DEFAULT 0, searched_at TIMESTAMP DEFAULT NOW())""",
            """CREATE TABLE IF NOT EXISTS user_inputs (
                id SERIAL PRIMARY KEY, user_id INTEGER, city VARCHAR(80),
                temperature NUMERIC(5,2), humidity NUMERIC(5,2), season VARCHAR(30),
                selected_crop VARCHAR(120), container_size VARCHAR(40),
                recorded_at TIMESTAMP DEFAULT NOW())""",
            """CREATE TABLE IF NOT EXISTS uploads (
                id SERIAL PRIMARY KEY, user_id INTEGER, upload_type VARCHAR(20),
                file_path VARCHAR(300), result_summary TEXT,
                uploaded_at TIMESTAMP DEFAULT NOW())""",
        ]
        with get_db() as (conn, cur):
            for t in tables: cur.execute(t)
    else:
        os.makedirs('instance', exist_ok=True)
        import sqlite3 as sq
        c = sq.connect(SQLITE_PATH)
        c.executescript('''
            CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL, password TEXT NOT NULL,
                location TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS crops (id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL, category TEXT, soil_type TEXT, water_req TEXT,
                sunlight TEXT, season TEXT, temp_min INTEGER, temp_max INTEGER,
                container_size TEXT, days_to_harvest INTEGER, organic_fertilizer TEXT,
                companion_plants TEXT, avoid_plants TEXT, seeds_cost INTEGER,
                soil_cost INTEGER, fertilizer_cost INTEGER);
            CREATE TABLE IF NOT EXISTS diseases (id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL, affected_plants TEXT, symptoms TEXT,
                causes TEXT, organic_treatment TEXT, prevention TEXT, color_hint TEXT);
            CREATE TABLE IF NOT EXISTS user_activity (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, activity_type TEXT, details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id));
            CREATE TABLE IF NOT EXISTS search_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, crop_name TEXT NOT NULL, results_found INTEGER DEFAULT 0,
                searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS user_inputs (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, city TEXT, temperature REAL, humidity REAL,
                season TEXT, selected_crop TEXT, container_size TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS uploads (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, upload_type TEXT, file_path TEXT, result_summary TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        ''')
        c.commit(); c.close()
    _seed(); print(f"DB ready ({'PostgreSQL' if is_pg else 'SQLite'})")

def _seed():
    CROPS = [
        ("Tomato","Fruit Vegetable","Rich Loamy","High","Full Sun","Summer",18,32,"Large",80,"Vermicompost,Cow Dung,Neem Oil","Basil,Carrot,Marigold","Fennel,Cabbage",40,120,80),
        ("Spinach","Leafy Vegetable","Rich Loamy","High","Partial Sun","Winter",7,20,"Medium",40,"Vermicompost,Compost","Strawberry,Garlic","Fennel",20,80,50),
        ("Carrot","Root Vegetable","Sandy Loam","Moderate","Full Sun","Winter",10,25,"Medium",75,"Compost,Wood Ash","Onion,Rosemary","Dill",25,90,60),
        ("Cucumber","Vine Vegetable","Rich Loamy","High","Full Sun","Summer",22,35,"Large",55,"Vermicompost,Kitchen Waste","Beans,Dill,Marigold","Potatoes",30,100,60),
        ("Eggplant","Fruit Vegetable","Rich Loamy","Moderate","Full Sun","Summer",22,32,"Large",75,"Cow Dung,Compost","Beans,Pepper","Fennel",35,100,70),
        ("Okra","Fruit Vegetable","Sandy Loam","Moderate","Full Sun","Summer",25,35,"Large",60,"Cow Dung,Compost","Basil,Pepper","None",30,90,60),
        ("Radish","Root Vegetable","Sandy Loam","Moderate","Full Sun","Winter",10,20,"Small",30,"Compost","Carrot,Lettuce","Hyssop",15,60,30),
        ("Lettuce","Leafy Vegetable","Rich Loamy","High","Partial Sun","Winter",10,22,"Small",45,"Compost,Kitchen Waste","Carrot,Radish","Celery",25,80,50),
        ("Kale","Leafy Vegetable","Rich Loamy","Moderate","Full Sun","Winter",7,20,"Large",60,"Vermicompost,Compost","Onion,Herbs","Strawberry",35,100,70),
        ("Broccoli","Brassica","Rich Loamy","High","Full Sun","Winter",10,20,"Large",80,"Vermicompost,Cow Dung","Onion,Marigold","Tomato",40,100,70),
        ("Cabbage","Brassica","Clay Loam","High","Full Sun","Winter",7,18,"Large",90,"Cow Dung,Compost","Dill,Mint","Strawberry",35,100,70),
        ("Peas","Legume","Sandy Loam","Moderate","Full Sun","Winter",10,20,"Medium",60,"Minimal - fixes nitrogen","Carrot,Turnip,Mint","Onion,Garlic",25,80,30),
        ("Bitter Gourd","Vine Vegetable","Sandy Loam","High","Full Sun","Summer",25,38,"Large",65,"Cow Dung,Compost","Beans","Potato",30,100,60),
        ("Ridge Gourd","Vine Vegetable","Rich Loamy","High","Full Sun","Summer",25,35,"Large",60,"Cow Dung,Compost","Beans","None",25,90,55),
        ("Bottle Gourd","Vine Vegetable","Sandy Loam","High","Full Sun","Summer",22,35,"Large",70,"Vermicompost","Corn,Beans","None",25,90,55),
        ("Coriander","Herb","Well-drained","Low","Partial Sun","Winter",15,25,"Small",25,"Compost","Dill,Anise","Fennel",10,50,25),
        ("Mint","Herb","Moist Loamy","High","Partial Sun","Spring",15,28,"Small",30,"Compost","Tomato,Peas","Parsley",15,60,30),
        ("Basil","Herb","Well-drained","Low","Full Sun","Summer",18,30,"Small",30,"Neem Oil,Compost","Tomato,Pepper","Sage",20,60,40),
        ("Fenugreek","Leafy Vegetable","Sandy Loam","Moderate","Full Sun","Winter",10,25,"Small",30,"Compost,Kitchen Waste","Corn","None",15,50,30),
        ("Green Onion","Bulb","Sandy Loam","Moderate","Full Sun","Winter",10,25,"Small",60,"Wood Ash,Compost","Tomato,Carrot","Beans,Peas",20,60,40),
        ("Garlic","Bulb","Sandy Loam","Low","Full Sun","Winter",10,20,"Medium",180,"Wood Ash,Compost","Tomato,Lettuce","Peas,Beans",20,80,50),
        ("Turmeric","Rhizome","Moist Loamy","High","Partial Sun","Summer",20,30,"Large",270,"Compost,Cow Dung","Ginger,Banana","None",50,100,70),
        ("Capsicum","Fruit Vegetable","Rich Loamy","Moderate","Full Sun","Summer",20,32,"Medium",75,"Compost,Vermicompost","Basil,Tomato","Fennel",35,90,65),
        ("Indian Spinach","Leafy Vegetable","Loamy","Moderate","Partial Sun","Summer",22,35,"Medium",40,"Kitchen Waste,Compost","Beans","None",20,70,40),
        ("Sweet Potato","Root Vegetable","Sandy Loam","Moderate","Full Sun","Summer",22,35,"Large",120,"Wood Ash,Compost","Beans,Marigold","Squash",40,100,60),
        ("French Beans","Legume","Sandy Loam","Moderate","Full Sun","Summer",18,30,"Medium",50,"Minimal","Carrot,Cucumber","Onion,Garlic",25,80,30),
        ("Cluster Beans","Legume","Sandy Loam","Low","Full Sun","Summer",25,38,"Medium",50,"Minimal","Maize","None",20,70,25),
        ("Pumpkin","Vine Vegetable","Rich Loamy","Moderate","Full Sun","Summer",20,35,"Large",90,"Cow Dung,Compost","Corn,Beans","Potato",30,100,60),
        ("Beetroot","Root Vegetable","Sandy Loam","Moderate","Full Sun","Winter",10,20,"Medium",65,"Compost,Wood Ash","Garlic,Lettuce","Runner Beans",25,80,50),
        ("Turnip","Root Vegetable","Sandy Loam","Moderate","Full Sun","Winter",7,18,"Medium",50,"Compost","Peas,Carrot","None",20,70,35),
        ("Spring Onion","Bulb","Sandy Loam","Moderate","Full Sun","Winter",10,22,"Small",50,"Compost,Wood Ash","Tomato,Carrot","Beans",15,55,35),
        ("Chilli","Fruit Vegetable","Sandy Loam","Moderate","Full Sun","Summer",20,35,"Medium",90,"Compost,Neem Oil","Basil,Carrot","Fennel",20,70,45),
        ("Amaranth","Leafy Vegetable","Loamy","Moderate","Full Sun","Summer",20,35,"Medium",45,"Compost,Vermicompost","Beans,Corn","None",30,80,50),
        ("Ivy Gourd","Vine Vegetable","Sandy Loam","Moderate","Full Sun","Summer",25,38,"Large",90,"Cow Dung,Compost","Beans","None",25,85,50),
        ("Dill","Herb","Well-drained","Low","Full Sun","Spring",15,25,"Small",45,"Compost","Cabbage,Onion","Fennel",15,50,30),
        ("Potato","Root Vegetable","Sandy Loam","Moderate","Partial Sun","Winter",10,25,"Large",90,"Compost,Cow Dung","Beans,Corn","Tomato",30,80,50),
        ("Onion","Bulb","Sandy Loam","Moderate","Full Sun","Winter",10,25,"Medium",100,"Compost,Wood Ash","Carrot,Tomato","Peas,Beans",20,60,40),
        ("Ginger","Rhizome","Moist Loamy","High","Partial Sun","Summer",20,30,"Medium",240,"Compost,Vermicompost","Turmeric,Chilli","None",40,90,60),
        ("Mustard Greens","Leafy Vegetable","Loamy","Moderate","Full Sun","Winter",10,25,"Medium",40,"Compost","Radish,Mint","None",15,60,30),
        ("Curry Leaves","Herb","Well-drained","Moderate","Full Sun","Summer",20,35,"Large",365,"Cow Dung,Compost","Mint,Basil","None",50,100,60),
        ("Drumstick","Tree Vegetable","Sandy Loam","Low","Full Sun","Summer",25,40,"Large",365,"Compost","Marigold,Basil","None",30,100,40),
        ("Pointed Gourd","Vine Vegetable","Sandy Loam","Moderate","Full Sun","Summer",25,38,"Large",120,"Compost,Cow Dung","Beans,Corn","Potato",40,90,60),
        ("Sponge Gourd","Vine Vegetable","Rich Loamy","High","Full Sun","Summer",22,35,"Large",70,"Compost,Neem Cake","Corn,Beans","None",25,90,50),
        ("Taro Root","Root Vegetable","Moist Loamy","High","Partial Sun","Summer",20,35,"Large",180,"Compost,Vermicompost","Curry Leaves,Ginger","None",35,80,50),
        ("Snake Gourd","Vine Vegetable","Sandy Loam","Moderate","Full Sun","Summer",25,35,"Large",80,"Compost,Cow Dung","Beans,Corn","None",25,85,55),
    ]
    is_pg = USING_POSTGRES and DATABASE_URL
    ins = ("INSERT INTO crops (name,category,soil_type,water_req,sunlight,season,temp_min,temp_max,"
           "container_size,days_to_harvest,organic_fertilizer,companion_plants,avoid_plants,"
           "seeds_cost,soil_cost,fertilizer_cost) VALUES "
           + ('(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
              if is_pg else '(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)')
           + (" ON CONFLICT (name) DO NOTHING" if is_pg else ""))
    if not is_pg:
        ins = "INSERT OR IGNORE INTO crops (name,category,soil_type,water_req,sunlight,season,temp_min,temp_max,container_size,days_to_harvest,organic_fertilizer,companion_plants,avoid_plants,seeds_cost,soil_cost,fertilizer_cost) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    with get_db() as (conn, cur):
        for row in CROPS:
            try: cur.execute(ins, row)
            except Exception: pass

    DISEASES = [
        ("Powdery Mildew","Cucumber,Peas,Squash,Beans","White/grey powdery coating on leaves; distorted shoots","High humidity, poor air circulation","1% Neem oil spray weekly; Baking soda 1tsp/L; Remove infected leaves","30cm spacing; avoid overhead watering","W"),
        ("Early Blight","Tomato,Potato,Pepper","Brown spots with concentric rings + yellow halo; lower leaves first","Alternaria fungus; warm humid weather","Neem oil 2ml/L; Bordeaux mixture; Remove infected leaves; Compost tea","Crop rotation; stake plants; water at base; mulch","R"),
        ("Root Rot","Most Plants,Tomato,Herbs","Wilting despite moist soil; yellowing; mushy dark roots","Pythium fungus; overwatering; compacted soil","Cinnamon powder on soil; Reduce watering; Repot with compost+perlite","Drainage holes; never sit in standing water","B"),
        ("Aphid Infestation","Tomato,Pepper,Spinach,Lettuce,Chilli","Tiny insects on shoot tips; honeydew; sooty mould; curled leaves","Warm weather; nitrogen-rich tissue; ants","Strong water jet; Neem oil 2ml/L; Garlic-chili spray; Soap solution","Encourage ladybugs; nasturtium trap crop","Y"),
        ("Mosaic Virus","Tomato,Cucumber,Beans","Yellow-green mosaic on leaves; distorted growth; misshapen fruits","Spread by aphids/thrips; contaminated tools","Remove and destroy infected plants; disinfect tools with neem oil","Virus-resistant seeds; sterilise tools; control aphids","Y"),
        ("Damping Off","All Seedlings","Stem collapses at soil; water-soaked base; seeds fail to emerge","Overwatering; poor ventilation; unsterilised mix","Cinnamon powder; Wood ash; Reduce watering; Chamomile tea drench","Sterilised seed mix; sow thinly; water from below","B"),
        ("Whitefly","Tomato,Eggplant,Okra,Chilli","White fly clouds when disturbed; yellowing; honeydew; sooty mould","Hot dry; dense planting; no natural predators","Yellow sticky traps; Neem oil weekly; Silver mulch; Garlic spray","Basil + marigold companions; inspect new plants","W"),
        ("Blossom End Rot","Tomato,Pepper,Eggplant","Dark leathery patch at fruit bottom; brown dry internal tissue","Calcium deficiency; irregular watering; high nitrogen","Crushed eggshells in soil; Consistent watering; Compost tea; Mulch","Consistent moisture; avoid excess nitrogen","R"),
        ("Downy Mildew","Spinach,Lettuce,Kale,Peas","Yellow patches on upper leaf; grey-purple fluffy growth underneath","Cool wet weather; humid nights; poor airflow","Neem oil; Copper soap spray; Remove infected leaves","Avoid wetting foliage; water mornings; good spacing","Y"),
        ("Leaf Spot","Amaranth,Beetroot,French Beans","Brown/black circular spots; yellow halo; leaf drop in severe cases","Fungal/bacterial; wet conditions; overhead watering","Neem oil; Remove infected leaves; Wood ash solution","Water at base; good airflow; crop rotation","R"),
    ]
    dins = ("INSERT INTO diseases (name,affected_plants,symptoms,causes,organic_treatment,prevention,color_hint) VALUES "
            + ('(%s,%s,%s,%s,%s,%s,%s)' if is_pg else '(?,?,?,?,?,?,?)')
            + (" ON CONFLICT (name) DO NOTHING" if is_pg else ""))
    if not is_pg:
        dins = "INSERT OR IGNORE INTO diseases (name,affected_plants,symptoms,causes,organic_treatment,prevention,color_hint) VALUES (?,?,?,?,?,?,?)"
    with get_db() as (conn, cur):
        for row in DISEASES:
            try: cur.execute(dins, row)
            except Exception: pass
    print("DB seeded")

# ---------------------------------------------------------------------------
# VEGETABLE DATASET
# ---------------------------------------------------------------------------
def load_veg_dataset():
    p = os.path.join(os.path.dirname(__file__), 'vegetables.json')
    fallback = {
        "tomato": {"name": "Tomato", "temp_min": 15, "temp_max": 35},
        "potato": {"name": "Potato", "temp_min": 10, "temp_max": 30},
        "carrot": {"name": "Carrot", "temp_min": 10, "temp_max": 25}
    }
    try:
        if not os.path.exists(p):
            return fallback
        with open(p, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        d = {k.lower(): v for k, v in raw.get('vegetables',{}).items()}
        print(f"Loaded {len(d)} vegetables")
        return d if d else fallback
    except Exception as e:
        print(f"vegetables.json error: {e}")
        return fallback

VEG = load_veg_dataset()

# ---------------------------------------------------------------------------
# CNN MODEL LAYER
# ---------------------------------------------------------------------------
_dm, _cm = None, None
DISEASE_LABELS  = ["Powdery Mildew","Early Blight","Root Rot","Aphid Infestation",
                   "Mosaic Virus","Damping Off","Whitefly","Blossom End Rot","Downy Mildew","Leaf Spot"]

def get_disease_labels():
    try:
        p = os.path.join(os.path.dirname(__file__), 'models', 'disease_labels.json')
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {str(i): lbl for i, lbl in enumerate(DISEASE_LABELS)}

CONTAINER_LABELS = ["large","medium","small"]

def get_disease_model():
    global _dm
    if _dm: return _dm
    if not CNN_AVAILABLE: return None
    p = os.path.join(os.path.dirname(__file__), 'models', 'disease_model.keras')
    if os.path.exists(p):
        try: _dm = tf.keras.models.load_model(p); return _dm
        except Exception as e: print(f"Disease model error: {e}")
    return None

def get_container_model():
    global _cm
    if _cm: return _cm
    if not CNN_AVAILABLE: return None
    p = os.path.join(os.path.dirname(__file__), 'models', 'container_model.keras')
    if os.path.exists(p):
        try: _cm = tf.keras.models.load_model(p); return _cm
        except Exception as e: print(f"Container model error: {e}")
    return None

def cnn_disease(pil_img):
    m = get_disease_model()
    labels = get_disease_labels()
    if not m: return None
    try:
        arr = np.array(pil_img.convert('RGB').resize((128,128)), dtype=np.float32)/255.0
        p = m.predict(np.expand_dims(arr,0), verbose=0)[0]
        
        idx = int(np.argmax(p))
        idx_str = str(idx) # Exact dictionary mapping without hashing conflicts
        conf = float(p[idx]) * 100
        
        print("\n--- Model Prediction Debug ---")
        print(f"Full Prediction Array: {p}")
        print(f"Max Index: {idx}, Confidence: {conf:.2f}%")
        
        if np.std(p) < 0.05 or np.max(p) < 0.3:
            print("⚠️ WARNING: Model not trained properly (predictions are almost identical)")

        print(f"Loaded Class Labels Keys: {list(labels.keys())}")
        
        if idx_str in labels:
            disease_name = labels[idx_str]
        else:
            try:
                disease_name = list(labels.values())[idx]
            except Exception:
                disease_name = "Unknown Disease"
                
        disease_name = disease_name.replace('_', ' ')
        
        
        print(f"Mapped Disease Name: {disease_name}")
        
        if conf < 30.0:
            plant_type = disease_name.split()[0] if ' ' in disease_name else "Unknown Plant"
            return f"{plant_type} General Issue", float(conf)
            
        return disease_name, float(conf)
    except Exception: return None

def cnn_container(pil_img):
    m = get_container_model()
    if not m: return None
    try:
        arr = np.array(pil_img.convert('RGB').resize((128,128)), dtype=np.float32)/255.0
        p = m.predict(np.expand_dims(arr,0), verbose=0)[0]
        idx = int(np.argmax(p))
        print("Prediction Array:", p)
        print("Predicted Class:", CONTAINER_LABELS[idx])
        conf = int(round(float(p[idx])*100))
        if conf < 40: return None
        return CONTAINER_LABELS[idx].capitalize(), conf
    except Exception: return None

def heuristic_disease(pil_img):
    img = pil_img.convert('RGB').resize((200,200))
    stat = ImageStat.Stat(img)
    mr,mg,mb = stat.mean; sr,sg,sb = stat.stddev
    brightness = (mr+mg+mb)/3; contrast = (sr+sg+sb)/3
    px = list(img.getdata()); tot = len(px)
    yf = sum(1 for r,g,b in px if r>160 and g>140 and b<80)/tot
    wf = sum(1 for r,g,b in px if r>200 and g>200 and b>200)/tot
    bf = sum(1 for r,g,b in px if r>120 and g<100 and b<80)/tot
    df = sum(1 for r,g,b in px if r<60 and g<60 and b<60)/tot
    v = {n:0 for n in DISEASE_LABELS}
    if mr>mg+15 and mr>mb+15: v["Early Blight"]+=3; v["Blossom End Rot"]+=2; v["Leaf Spot"]+=2
    if mb>mr+10: v["Root Rot"]+=3; v["Damping Off"]+=2
    if brightness>190: v["Powdery Mildew"]+=4; v["Whitefly"]+=2
    elif brightness<70: v["Root Rot"]+=2; v["Damping Off"]+=2
    if contrast>55: v["Early Blight"]+=2; v["Aphid Infestation"]+=1; v["Mosaic Virus"]+=1; v["Leaf Spot"]+=2
    elif contrast<20: v["Powdery Mildew"]+=3; v["Downy Mildew"]+=1
    if yf>0.15: v["Mosaic Virus"]+=3; v["Aphid Infestation"]+=2
    if wf>0.20: v["Powdery Mildew"]+=3; v["Whitefly"]+=2
    if bf>0.18: v["Blossom End Rot"]+=3; v["Early Blight"]+=2; v["Leaf Spot"]+=2
    if df>0.25: v["Root Rot"]+=2; v["Damping Off"]+=3
    winner = max(v, key=v.get); tv = sum(v.values()) or 1
    return winner, max(48, min(91, 45+int((v[winner]/tv)*55)))

def heuristic_container(pil_img):
    stat = ImageStat.Stat(pil_img.convert('RGB'))
    mr,mg,mb = stat.mean; w,h = pil_img.size
    vr,vg,vb = stat.var
    variance = (vr+vg+vb)/3
    greenness = mg/(mr+mb+1)
    area = w*h
    
    if greenness > 0.55 and variance < 3000: return "Mismatch", 90
    if greenness > 0.8: return "Mismatch", 85
    
    if area < 150000: return "small", 60
    elif area > 1000000 and greenness < 0.4: return "large", 65
    return "medium", 65

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def hash_pw(p): return hashlib.sha256(p.encode()).hexdigest()

def log_activity(uid, atype, details):
    if USING_POSTGRES and DATABASE_URL:
        safe_exec("INSERT INTO user_activity (user_id,activity_type,details) VALUES (%s,%s,%s)",
                  (uid,atype,json.dumps(details)))
    else:
        safe_exec("INSERT INTO user_activity (user_id,activity_type,details) VALUES (?,?,?)",
                  (uid,atype,json.dumps(details)))

def log_search(uid, name, n=0):
    P = ph()
    safe_exec(f"INSERT INTO search_history (user_id,crop_name,results_found) VALUES ({P},{P},{P})",(uid,name.strip(),n))

def log_upload(uid, utype, fpath, summary=''):
    P = ph()
    safe_exec(f"INSERT INTO uploads (user_id,upload_type,file_path,result_summary) VALUES ({P},{P},{P},{P})",(uid,utype,fpath,summary))

VALID_CITIES = {c.lower() for c in [
    "Delhi","New Delhi","Mumbai","Bangalore","Bengaluru","Chennai","Kolkata","Hyderabad","Pune",
    "Ahmedabad","Jaipur","Lucknow","Chandigarh","Bhopal","Indore","Nagpur","Patna","Agra","Varanasi",
    "Surat","Vadodara","Rajkot","Coimbatore","Madurai","Visakhapatnam","Vijayawada","Kochi","Mysuru",
    "Jodhpur","Udaipur","Ajmer","Kota","Shimla","Dehradun","Haridwar","Amritsar","Ludhiana","Guwahati",
    "Ranchi","Bhubaneswar","Raipur","Srinagar","Jammu","Noida","Gurgaon","Gurugram","Faridabad",
    "Meerut","Prayagraj","Kanpur","Nashik","Aurangabad","Pondicherry","London","Paris","New York",
    "Los Angeles","Chicago","Houston","Toronto","Vancouver","Sydney","Melbourne","Auckland","Tokyo",
    "Beijing","Shanghai","Hong Kong","Singapore","Dubai","Abu Dhabi","Berlin","Munich","Vienna",
    "Amsterdam","Madrid","Barcelona","Rome","Milan","Moscow","Warsaw","Prague","Stockholm","Lisbon",
    "Dublin","Karachi","Lahore","Islamabad","Dhaka","Colombo","Kathmandu","Bangkok","Jakarta",
    "Manila","Kuala Lumpur","Doha","Nairobi","Cairo","Lagos","Johannesburg","Buenos Aires","Sao Paulo","Mexico City",
]}

def validate_location(city):
    c = city.strip()
    if not c: return {"valid":False,"error":"Location cannot be empty."}
    if c.lower() in VALID_CITIES: return {"valid":True,"canonical":c.title()}
    KEY = os.environ.get('OPENWEATHER_API_KEY','demo')
    try:
        r = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={c}&appid={KEY}&units=metric",timeout=4)
        if r.status_code==200: return {"valid":True,"canonical":r.json().get("name",c)}
        return {"valid":False,"error":f"'{c}' not recognised. Check spelling."}
    except Exception:
        return {"valid":True,"canonical":c.title(),"warning":"Could not verify online."}

def get_weather(city):
    KEY = os.environ.get('OPENWEATHER_API_KEY','demo')
    try:
        r = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={KEY}&units=metric",timeout=5)
        if r.status_code==200:
            d = r.json()
            return {"temp":round(d["main"]["temp"]),"humidity":d["main"]["humidity"],
                    "condition":d["weather"][0]["main"],"description":d["weather"][0]["description"].title(),
                    "city":d["name"],"simulated":False}
        if r.status_code==404: return {"error":f"City '{city}' not found."}
    except Exception: pass
    cond = random.choice(["Sunny","Cloudy","Rainy","Clear","Partly Cloudy"])
    return {"temp":random.randint(18,35),"humidity":random.randint(45,85),
            "condition":cond,"description":cond,"city":city.title(),"simulated":True}

def infer_season(t):
    return "Winter" if t<=15 else "Spring" if t<=22 else "Summer"

def ai_recommend(temp, humidity, season=None):
    scored = []
    all_crops = db_exec("SELECT * FROM crops", many=True) or []
    for veg in all_crops:
        s = random.randint(0, 15) # Adds diversity
        tmin, tmax = veg.get('temp_min', 10), veg.get('temp_max', 35)
        if tmin <= temp <= tmax: s += 50 + int((1 - abs(temp - (tmin + tmax) / 2) / max(tmax - tmin, 1)) * 20)
        elif temp < tmin: s += max(0, 30 - (tmin - temp) * 3)
        else: s += max(0, 30 - (temp - tmax) * 3)
        if season and season.lower() in [x.lower() for x in (veg.get('season') or '').split(',')]: s += 25
        wr = (veg.get('water_req') or 'Moderate').lower()
        if (humidity > 70 and wr == 'high') or (humidity < 40 and wr == 'low') or (40 <= humidity <= 70 and wr == 'moderate'): s += 10
        if s > 20: scored.append((s, veg))
    scored.sort(key=lambda x: -x[0])
    
    # Shuffle top matches to prevent repeating exact same crops 
    top_candidates = [v for _, v in scored[:12]]
    random.shuffle(top_candidates)
    return top_candidates[:6]

# ---------------------------------------------------------------------------
# AUTH ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        u=request.form.get('username','').strip(); p=request.form.get('password','')
        P=ph()
        user=db_exec(f"SELECT * FROM users WHERE username={P} AND password={P}",(u,hash_pw(p)),one=True)
        if user:
            session.permanent=True
            session.update({'user_id':user['id'],'username':user['username'],'location':user['location']})
            return jsonify({"success":True,"redirect":url_for('dashboard'),
                            "username":user['username'],"location":user['location']})
        return jsonify({"success":False,"error":"Invalid username or password."})
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    d=request.get_json() or {}
    u=d.get('username','').strip(); p=d.get('password',''); loc=d.get('location','').strip()
    if not all([u,p,loc]): return jsonify({"success":False,"error":"All fields required."})
    lv=validate_location(loc)
    if not lv["valid"]: return jsonify({"success":False,"error":lv["error"]})
    P=ph()
    try:
        db_exec(f"INSERT INTO users (username,password,location) VALUES ({P},{P},{P})",(u,hash_pw(p),lv["canonical"]))
        user=db_exec(f"SELECT * FROM users WHERE username={P}",(u,),one=True)
        session.permanent=True
        session.update({'user_id':user['id'],'username':user['username'],'location':user['location']})
        r={"success":True,"redirect":url_for('dashboard'),"username":user['username'],"location":user['location']}
        if lv.get("warning"): r["warning"]=lv["warning"]
        return jsonify(r)
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            return jsonify({"success":False,"error":"Username already taken."})
        return jsonify({"success":False,"error":"Sign-up failed."})

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('login'))

# Removed forgot_password and reset_password endpoints as requested.

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    P=ph()
    acts=db_exec(f"SELECT * FROM user_activity WHERE user_id={P} ORDER BY created_at DESC LIMIT 5",
                 (session['user_id'],),many=True) or []
    return render_template('dashboard.html',username=session['username'],
                           location=session['location'],activities=acts)

# ---------------------------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------------------------
@app.route('/api/validate-location', methods=['POST'])
def api_validate_location():
    return jsonify(validate_location((request.get_json() or {}).get('city','')))

@app.route('/api/weather')
def api_weather():
    city=request.args.get('city',session.get('location','Delhi'))
    lv=validate_location(city)
    if not lv["valid"]: return jsonify({"error":lv["error"]}),400
    data=get_weather(lv["canonical"])
    if "error" in data: return jsonify(data),400
    season=infer_season(data['temp']); cond=data.get('condition','')
    advice=("Skip watering — rain expected" if cond in ['Rain','Drizzle','Thunderstorm']
            else "Water twice daily — extreme heat" if data['temp']>34
            else "Water every morning" if data['temp']>28
            else "Water every 2-3 days")
    data.update({'watering_advice':advice,'season_inferred':season,
                 'crop_suggestions':ai_recommend(data['temp'],data['humidity'],season)[:4]})
    uid=session.get('user_id')
    if uid:
        P=ph(); safe_exec(f"INSERT INTO user_inputs (user_id,city,temperature,humidity,season) VALUES ({P},{P},{P},{P},{P})",
                          (uid,lv["canonical"],data['temp'],data['humidity'],season))
    return jsonify(data)

@app.route('/api/crops')
def api_crops():
    q = request.args.get('q', '').strip()
    season = request.args.get('season', '').strip()
    container = request.args.get('container', '').strip()
    is_pg = USING_POSTGRES and DATABASE_URL
    like = "ILIKE" if is_pg else "LIKE"
    P = ph()
    
    sql = "SELECT DISTINCT * FROM crops WHERE 1=1"
    params = []
    if q:
        sql += f" AND (LOWER(name) {like} LOWER({P}) OR LOWER(category) {like} LOWER({P}))"
        params.extend([f"%{q}%", f"%{q}%"])
    if season: 
        sql += f" AND LOWER(season) {like} LOWER({P})"
        params.append(f"%{season}%")
    if container: 
        sql += f" AND LOWER(container_size) {like} LOWER({P})"
        params.append(f"%{container}%")
    sql += " ORDER BY name"
    
    rows = db_exec(sql, params, many=True) or []
    
    # Ensure there is always a fallback value returned for UX if partial search fails entirely
    if not rows and q:
        fallback_sql = f"SELECT DISTINCT * FROM crops WHERE LOWER(name) {like} LOWER({P}) ORDER BY name LIMIT 5"
        rows = db_exec(fallback_sql, [f"%{q[0]}%"], many=True) or []
        
    seen, result = [], []
    for r in rows:
        if r['name'] not in seen:
            seen.append(r['name'])
            vd = VEG.get(r['name'].lower(), {})
            r.update({
                'youtube_id': vd.get('youtube_id', ''),
                'watering_schedule': vd.get('watering_schedule', 'Every 2-3 days'),
                'image_url': f"https://placehold.co/400x300/e6f2eb/2d6a4f?text={r['name'].replace(' ', '+')}"
            })
            result.append(r)
            
    if q and len(q)>=2: log_search(session.get('user_id'), q, len(result))
    return jsonify(result)

@app.route('/api/crop/<int:crop_id>')
def api_crop_detail(crop_id):
    P=ph(); crop=db_exec(f"SELECT * FROM crops WHERE id={P}",(crop_id,),one=True)
    if not crop: return jsonify({"error":"Not found"}),404
    vd=VEG.get(crop['name'].lower(),{})
    crop.update({'youtube_id':vd.get('youtube_id',''),'watering_schedule':vd.get('watering_schedule',''),
                 'growth_stages':vd.get('growth_stages',{}),'fertilizer_schedule':vd.get('fertilizer_schedule','Every 3 weeks'),
                 'growth_tips':vd.get('growth_tips',''),'harvest_tips':vd.get('harvest_tips',''),'spacing_cm':vd.get('spacing_cm',30),
                 'image_url': f"https://placehold.co/400x300/e6f2eb/2d6a4f?text={crop['name'].replace(' ', '+')}"})
    if 'user_id' in session: log_activity(session['user_id'],'crop_view',{'crop':crop['name']})
    return jsonify(crop)

@app.route('/api/diseases')
def api_diseases():
    return jsonify(db_exec("SELECT DISTINCT * FROM diseases ORDER BY name",many=True) or [])

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    d = request.get_json() or {}
    temp = float(d.get('temp', 25))
    hum = float(d.get('humidity', 60))
    season = d.get('season', '')
    space = d.get('space', '')
    exp = d.get('experience', 'beginner')
    
    is_pg = USING_POSTGRES and DATABASE_URL
    like = "ILIKE" if is_pg else "LIKE"
    P = ph()
    
    s = season or infer_season(temp)
    
    # Fetch directly using robust ai_recommend logic over the DB
    base_res = ai_recommend(temp, hum, s)
    res = list(base_res)
    
    # Apply user filters
    if space: 
        res = [v for v in res if space.lower() in (v.get('container_size') or '').lower()]
    if exp == 'beginner': 
        res = [v for v in res if v.get('days_to_harvest') is not None and v.get('days_to_harvest') <= 80]
        
    # Robust fallback: if filters removed all results, fetch top matches ignoring filters
    if not res:
        res = base_res[:6]
        if not res:
            res = db_exec("SELECT * FROM crops ORDER BY name LIMIT 6", many=True) or []
            
    seen, out = set(), []
    for v in res:
        nm = v.get('name', '')
        if nm and nm not in seen:
            seen.add(nm)
            vd = VEG.get(nm.lower(), {})
            out.append({
                **v,
                "recommendation_extras": {
                    "fertilizer_suggestion": ','.join(v.get('organic_fertilizer', 'Compost').split(',')),
                    "fertilizer_schedule": vd.get('fertilizer_schedule', 'Every 3 weeks'),
                    "watering_schedule": vd.get('watering_schedule', 'Every 2-3 days'),
                    "growth_tips": vd.get('growth_tips', ''),
                    "harvest_tips": vd.get('harvest_tips', ''),
                    "spacing_cm": vd.get('spacing_cm', 30)
                },
                "youtube_id": vd.get('youtube_id', ''),
                "image_url": f"https://placehold.co/400x300/e6f2eb/2d6a4f?text={nm.replace(' ', '+')}"
            })
    return jsonify(out[:6])

@app.route('/api/growth-stages/<crop_name>')
def api_growth_stages(crop_name):
    vd=VEG.get(crop_name.lower())
    if not vd:
        P=ph(); is_pg=USING_POSTGRES and DATABASE_URL; like="ILIKE" if is_pg else "LIKE"
        vd=db_exec(f"SELECT * FROM crops WHERE name {like} {P}",(f"%{crop_name}%",),one=True)
        if not vd: return jsonify({"error":"Not found"}),404
    stages=vd.get('growth_stages') or {
        "Seed":f"Sow at correct depth; keep moist",
        "Germination":"Sprouts in 5-14 days",
        "Vegetative":f"Water {vd.get('watering_schedule','regularly')}; apply organic fertiliser",
        "Flowering":"Pollination stage; phosphorus-rich organic feed helps",
        "Harvest":f"Ready in ~{vd.get('days_to_harvest',60)} days"}
    return jsonify({"crop":vd.get('name',crop_name),"days_to_harvest":vd.get('days_to_harvest',60),
                    "watering_schedule":vd.get('watering_schedule','Every 2-3 days'),"stages":stages})

@app.route('/api/garden-layout', methods=['POST'])
def api_garden_layout():
    crops=(request.get_json() or {}).get('crops',[])
    if not crops: return jsonify({"error":"Provide crop list"}),400
    cset={c.lower() for c in crops}; advice=[]; conflicts=[]
    for cn in crops:
        vd=VEG.get(cn.lower(),{})
        a={"crop":vd.get('name',cn),"good_companions":vd.get('companion_plants',[]),
           "avoid_companions":vd.get('avoid_plants',[]),"spacing_cm":vd.get('spacing_cm',30),
           "container_size":vd.get('container_size','Medium')}
        advice.append(a)
        for av in vd.get('avoid_plants',[]):
            if av.lower() in cset:
                conflicts.append({"crop1":a['crop'],"crop2":av,"message":f"⚠️ {a['crop']} and {av} should NOT be planted together"})
    return jsonify({"layout_advice":advice,"conflicts_detected":conflicts})

@app.route('/upload/container', methods=['POST'])
def upload_container():
    if 'file' not in request.files: return jsonify({"error":"No file"}),400
    try: img=Image.open(request.files['file'].stream); w,h=img.size
    except Exception: return jsonify({"error":"Could not read image"}),400
    cr=cnn_container(img)
    if cr and cr[1] >= 50: size,conf=cr; method="CNN"
    else: size,conf=heuristic_container(img); method="CNN / Heuristic"
    
    if size.lower() == "mismatch" or size.lower() == "unknown":
        return jsonify({
            "container_size": "medium", 
            "confidence": conf, 
            "status": "uncertain", 
            "tips": "Invalid input detected. Please upload a clear container image.",
            "message": "Invalid input: Please upload a clearer container image",
            "recommended_crops": [],
            "dimensions": f"{w}x{h}px",
            "method": "Heuristic"
        })

    is_pg=USING_POSTGRES and DATABASE_URL; like="ILIKE" if is_pg else "LIKE"; P=ph()
    crops=db_exec(f"SELECT DISTINCT * FROM crops WHERE container_size {like} {P} ORDER BY name",(f"%{size}%",),many=True) or []
    seen,enhanced=set(),[]
    for c in crops:
        if c['name'] not in seen:
            seen.add(c['name']); vd=VEG.get(c['name'].lower(),{})
            enhanced.append({**c,"watering_schedule":vd.get('watering_schedule','Every 2-3 days'),
                             "sunlight_detail":vd.get('sunlight','Full Sun'),
                             "harvest_time":f"{c.get('days_to_harvest','?')} days",
                             "harvest_tips":vd.get('harvest_tips',''),"youtube_id":vd.get('youtube_id','')})
    random.shuffle(enhanced)
    uid=session.get('user_id')
    if uid: log_upload(uid,'container',f"img_{w}x{h}",f"{size} ({method})")
    return jsonify({
        "container_size": size.lower(),
        "confidence": conf,
        "method": method,
        "status": "success",
        "dimensions": f"{w}x{h}px",
        "recommended_crops": enhanced[:5],
        "tips": {"Small":"Herbs, radish, lettuce. Use 6-8 inch pots.",
                 "Medium":"Tomatoes, peppers. Use 12-14 inch containers.",
                 "Large":"Gourds, cucumbers. Use 18+ inch containers.",
                 "Mismatch":"Invalid container type detected.",
                 "Unknown":"Unable to calculate container bounds clearly."}.get(size, "")
    })

@app.route('/upload/disease', methods=['POST'])
def upload_disease():
    if 'file' not in request.files: return jsonify({"error":"No file"}),400
    try: img=Image.open(request.files['file'].stream)
    except Exception: return jsonify({"error":"Could not read image"}),400
    cr=cnn_disease(img)
    if cr: dn,conf=cr; method="CNN (MobileNetV2)"
    else: dn,conf=heuristic_disease(img); method="Heuristic (7-signal)"
    is_pg=USING_POSTGRES and DATABASE_URL; like="ILIKE" if is_pg else "LIKE"; P=ph()
    
    # Improved disease matching logic
    all_diseases = db_exec("SELECT * FROM diseases", many=True)
    disease = None
    
    # Extract base name without uncertain tags for DB matching
    search_name = dn.replace(' (Uncertain Prediction)', '')
    
    if all_diseases:
        for d in all_diseases:
            if d['name'].lower() in search_name.lower() or search_name.lower() in d['name'].lower():
                disease = dict(d)
                break
    
    # Dynamic fallback if not found in DB
    if not disease:
        disease = {
            "name": dn,
            "affected_plants": "Various affected plants",
            "symptoms": "Generic symptoms or non-specific stress indicators observed.",
            "causes": "Environmental factors, undiagnosed conditions, or non-ideal lighting.",
            "organic_treatment": "Trim affected leaves, ensure optimal watering, and isolate if necessary.",
            "prevention": "Maintain garden hygiene, monitor moisture levels, and provide appropriate sunlight.",
            "color_hint": "Y"
        }
        
    uid=session.get('user_id')
    if uid:
        log_upload(uid,'disease','leaf_image',f"{disease['name']} ({conf}%)")
        log_activity(uid,'disease_detection',{"disease":disease['name'],"confidence":conf,"method":method})
        
    response = dict(disease)
    response["confidence"] = round(conf, 2)
    response["status"] = "success"
    
    return jsonify(response)

@app.route('/api/shopping')
def api_shopping():
    crops=db_exec("SELECT DISTINCT name,seeds_cost,soil_cost,fertilizer_cost,days_to_harvest,organic_fertilizer,container_size,category FROM crops ORDER BY name",many=True) or []
    SOILM={"Leafy Vegetable":"Cocopeat + Vermicompost (60:40)","Fruit Vegetable":"Loamy + Compost + Perlite (50:30:20)",
           "Herb":"Well-drained + Cocopeat (70:30)","Root Vegetable":"Sandy loam + Compost (70:30)",
           "Vine Vegetable":"Rich Loamy + Cow Dung (60:40)","Brassica":"Rich soil + Vermicompost (60:40)",
           "Bulb":"Sandy loam + Wood ash (80:20)","Legume":"Sandy loam (no extra N)","Rhizome":"Moist loamy + Cow dung (60:40)"}
    POTN={"Small":"6-8 inch pot / 1-3L grow bag | Rs.40-80","Medium":"10-12 inch pot / 5-10L grow bag | Rs.80-150",
          "Large":"14-18 inch pot / 15-25L grow bag | Rs.150-300","Deep Medium":"12-inch deep pot / 10L | Rs.120-200"}
    seen,result=set(),[]
    for c in crops:
        if c['name'] in seen: continue
        seen.add(c['name']); vd=VEG.get(c['name'].lower(),{})
        sc=c.get('seeds_cost') or vd.get('seeds_cost',30)
        slc=c.get('soil_cost') or vd.get('soil_cost',80)
        fc=c.get('fertilizer_cost') or vd.get('fertilizer_cost',50)
        result.append({"crop":c['name'],"category":c.get('category','Vegetable'),
            "seeds":{"item":vd.get('seeds_qty',f"{c['name']} Seeds"),"cost":sc},
            "soil":{"item":SOILM.get(c.get('category',''),"Potting mix + Compost"),"qty":vd.get('soil_qty','3-5 kg'),"cost":slc},
            "fertilizer":{"items":(c.get('organic_fertilizer') or 'Compost').split(','),"schedule":vd.get('fertilizer_schedule','Every 3 weeks'),"cost":fc},
            "container":{"recommendation":POTN.get(c.get('container_size','Medium'),"Suitable container"),"cost":80},
            "watering_schedule":vd.get('watering_schedule','Every 2-3 days'),
            "harvest_time":f"{c.get('days_to_harvest','?')} days","youtube_id":vd.get('youtube_id',''),
            "total_est":sc+slc+fc+80,"quantity_summary":f"Seeds:{vd.get('seeds_qty','1 packet')} | Soil:{vd.get('soil_qty','3-5 kg')}"})
    return jsonify(result)

@app.route('/api/shop')
def api_shop():
    q = request.args.get('q', '').strip()
    is_pg=USING_POSTGRES and DATABASE_URL; like="ILIKE" if is_pg else "LIKE"; P=ph()
    sql = "SELECT DISTINCT name, seeds_cost FROM crops WHERE 1=1"
    params = []
    if q:
        sql += f" AND LOWER(name) {like} LOWER({P})"
        params.append(f"%{q}%")
    rows = db_exec(sql, params, many=True) or []
    
    out = []
    seen = set()
    for r in rows:
        cname = r['name']
        if cname in seen: continue
        seen.add(cname)
        cost = r.get('seeds_cost') or random.randint(30, 80)
        out.append({
            "crop": cname, 
            "price": f"₹{cost}",
            "buy_link": f"https://www.amazon.in/s?k={cname.replace(' ', '+')}+seeds+for+gardening",
            "image_url": f"https://placehold.co/400x300/e6f2eb/2d6a4f?text={cname.replace(' ', '+')}+Seeds"
        })
    return jsonify(out)

@app.route('/api/waste-to-best')
def api_waste_to_best():
    videos = [
        {"title": "Complete Guide to Kitchen Composting", "video_id": "qwKc13R-bMc"},
        {"title": "DIY Organic Liquid Fertilizer from Kitchen Waste", "video_id": "7hBnlajtfxc"},
        {"title": "How to Make Compost at Home for Beginners", "video_id": "qwKc13R-bMc"},
        {"title": "Turn Vegetable Scraps Into Black Gold", "video_id": "_6zeDbEkWdg"},
        {"title": "Fastest Home Composting Method", "video_id": "_K25WjjCBuw"},
        {"title": "Zero Cost Kitchen Waste Compost", "video_id": "a8Y7laHr57c"},
        {"title": "Organic Fertilizers at Home", "video_id": "qMosKNTL85s"},
        {"title": "Grow Vegetables in Pots Easy Guide", "video_id": "kNAd4BZv7c0"}
    ]
    for v in videos:
        v["embed_url"] = f"https://www.youtube.com/embed/{v['video_id']}?rel=0&modestbranding=1"
        v["thumbnail"] = f"https://img.youtube.com/vi/{v['video_id']}/0.jpg"
    return jsonify(videos)

@app.route('/api/youtube/<crop_name>')
def api_youtube(crop_name):
    vd=VEG.get(crop_name.lower(),{}); vid=vd.get('youtube_id','')
    if not vid: return jsonify({"error":"No video for this crop"}),404
    return jsonify({"crop":vd.get('name',crop_name),"youtube_id":vid,
                    "embed_url":f"https://www.youtube.com/embed/{vid}?rel=0&modestbranding=1",
                    "watch_url":f"https://www.youtube.com/watch?v={vid}"})

@app.route('/api/user-stats') 
def api_user_stats():
    if 'user_id' not in session: return jsonify({"error":"Login required"}),401
    uid=session['user_id']; P=ph()
    s=db_exec(f"SELECT COUNT(*) as cnt FROM search_history WHERE user_id={P}",(uid,),one=True)
    u=db_exec(f"SELECT COUNT(*) as cnt FROM uploads WHERE user_id={P}",(uid,),one=True)
    t=db_exec("SELECT COUNT(*) as cnt FROM crops",one=True)
    return jsonify({"searches_done":(s or {}).get('cnt',0),"images_uploaded":(u or {}).get('cnt',0),
                    "total_crops_db":(t or {}).get('cnt',0),"dataset_count":len(VEG)})

@app.route('/predict')
def predict():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('predict.html')

if __name__=='__main__':
    init_db()
    debug=os.environ.get('FLASK_ENV','development')!='production'
    port=int(os.environ.get('PORT',5000))
    print(f"AI Kitchen Garden v4.0 — http://localhost:{port}")
    app.run(debug=debug,host='0.0.0.0',port=port)
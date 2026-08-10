import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews.db')


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn(); c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews (
        id TEXT PRIMARY KEY, username TEXT, review_text TEXT NOT NULL,
        rating INTEGER, review_date TEXT, app_version TEXT,
        scraped_at TEXT DEFAULT CURRENT_TIMESTAMP, analyzed INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT NOT NULL,
        normalized_text TEXT, sentiment TEXT, sentiment_confidence REAL,
        is_complaint INTEGER DEFAULT 0,
        category TEXT, subcategory TEXT, classification_confidence REAL,
        reason TEXT, escalated INTEGER DEFAULT 0,
        layer1 TEXT, layer2 TEXT, layer3 TEXT, layer4 TEXT, layer5 TEXT,
        quality_param TEXT, analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (review_id) REFERENCES reviews(id))''')
    for col in ['is_complaint INTEGER DEFAULT 0',
                'flagged INTEGER DEFAULT 0',
                'flag_note TEXT',
                'flagged_at TEXT']:
        try: c.execute(f'ALTER TABLE analysis_results ADD COLUMN {col}')
        except Exception: pass
    c.execute('CREATE INDEX IF NOT EXISTS idx_rev_an ON reviews(analyzed)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_res_sent ON analysis_results(sentiment)')
    conn.commit(); conn.close()
    print(f"[DB] siap: {os.path.abspath(DB_PATH)}")


def insert_reviews(reviews):
    conn = get_conn(); c = conn.cursor(); n = 0
    for r in reviews:
        try:
            c.execute('''INSERT OR IGNORE INTO reviews
                (id, username, review_text, rating, review_date, app_version)
                VALUES (?,?,?,?,?,?)''',
                (r['id'], r.get('username'), r['review_text'], r.get('rating'),
                 r.get('review_date'), r.get('app_version')))
            if c.rowcount > 0: n += 1
        except Exception as e:
            print(f"[DB] skip {r.get('id')}: {e}")
    conn.commit(); conn.close(); return n


def get_unanalyzed_reviews(limit=50):
    conn = get_conn(); c = conn.cursor()
    c.execute('SELECT * FROM reviews WHERE analyzed=0 ORDER BY scraped_at DESC LIMIT ?', (limit,))
    rows = [dict(r) for r in c.fetchall()]; conn.close(); return rows


def save_analysis(review_id, res):
    conn = get_conn(); c = conn.cursor()
    c.execute('''INSERT INTO analysis_results
        (review_id, normalized_text, sentiment, sentiment_confidence, is_complaint,
         category, subcategory, classification_confidence, reason, escalated,
         layer1, layer2, layer3, layer4, layer5, quality_param)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (review_id, res.get('normalized'), res.get('sentiment'),
         res.get('sentiment_confidence'), 1 if res.get('is_complaint') else 0,
         res.get('category'), res.get('subcategory'),
         res.get('classification_confidence'), res.get('reason'),
         1 if res.get('escalated') else 0,
         res.get('layer1',''), res.get('layer2',''), res.get('layer3',''),
         res.get('layer4',''), res.get('layer5',''), res.get('quality_param','')))
    c.execute('UPDATE reviews SET analyzed=1 WHERE id=?', (review_id,))
    conn.commit(); conn.close()


def get_dashboard_stats():
    conn = get_conn(); c = conn.cursor()
    c.execute('SELECT COUNT(*) t FROM reviews'); total = c.fetchone()['t']
    c.execute('SELECT COUNT(*) t FROM analysis_results'); ana = c.fetchone()['t']
    c.execute('SELECT sentiment, COUNT(*) n FROM analysis_results GROUP BY sentiment')
    sent = {r['sentiment']: r['n'] for r in c.fetchall()}
    c.execute("SELECT category, COUNT(*) n FROM analysis_results WHERE is_complaint=1 GROUP BY category ORDER BY n DESC")
    cat = [{'category': r['category'], 'count': r['n']} for r in c.fetchall() if r['category']]
    c.execute('SELECT COUNT(*) n FROM analysis_results WHERE is_complaint=1'); comp = c.fetchone()['n']
    c.execute('SELECT COUNT(*) n FROM analysis_results WHERE escalated=1'); esc = c.fetchone()['n']
    c.execute('''SELECT AVG(r.rating) a FROM reviews r
                 JOIN analysis_results x ON r.id=x.review_id''')
    row = c.fetchone(); avg = round(row['a'], 2) if row['a'] else 0
    conn.close()
    return {'total_reviews': total, 'total_analyzed': ana, 'total_complaints': comp,
            'escalated_count': esc, 'avg_rating': avg,
            'sentiment_distribution': sent, 'category_distribution': cat}


def get_recent_results(limit=50, sentiment=None, category=None,
                       layer1=None, layer2=None, layer3=None,
                       layer4=None, layer5=None, rating=None):
    conn = get_conn(); c = conn.cursor()
    q = '''SELECT a.id AS result_id, a.review_id, r.username, r.review_text, r.rating, r.review_date,
        a.normalized_text, a.sentiment, a.sentiment_confidence, a.is_complaint,
        a.category, a.subcategory, a.classification_confidence, a.reason, a.escalated,
        a.layer1, a.layer2, a.layer3, a.layer4, a.layer5, a.quality_param, a.analyzed_at,
        a.flagged, a.flag_note, a.flagged_at
        FROM analysis_results a JOIN reviews r ON r.id=a.review_id WHERE 1=1'''
    p = []
    if sentiment: q += ' AND a.sentiment=?';  p.append(sentiment)
    if category:  q += ' AND a.category=?';   p.append(category)
    if layer1:    q += ' AND a.layer1=?';     p.append(layer1)
    if layer2:    q += ' AND a.layer2=?';     p.append(layer2)
    if layer3:    q += ' AND a.layer3=?';     p.append(layer3)
    if layer4:    q += ' AND a.layer4=?';     p.append(layer4)
    if layer5:    q += ' AND a.layer5=?';     p.append(layer5)
    if rating:    q += ' AND r.rating=?';     p.append(int(rating))
    q += ' ORDER BY a.analyzed_at DESC LIMIT ?'; p.append(limit)
    c.execute(q, p); rows = [dict(r) for r in c.fetchall()]; conn.close(); return rows


def get_all_results(sentiment=None, layer1=None):
    """Sama seperti get_recent_results tapi TANPA limit — dipakai untuk export Excel."""
    conn = get_conn(); c = conn.cursor()
    q = '''SELECT a.id AS result_id, a.review_id, r.username, r.review_text, r.rating, r.review_date,
        a.normalized_text, a.sentiment, a.sentiment_confidence, a.is_complaint,
        a.category, a.subcategory, a.classification_confidence, a.reason, a.escalated,
        a.layer1, a.layer2, a.layer3, a.layer4, a.layer5, a.quality_param, a.analyzed_at,
        a.flagged, a.flag_note, a.flagged_at
        FROM analysis_results a JOIN reviews r ON r.id=a.review_id WHERE 1=1'''
    p = []
    if sentiment: q += ' AND a.sentiment=?'; p.append(sentiment)
    if layer1:    q += ' AND a.layer1=?';    p.append(layer1)
    q += ' ORDER BY a.analyzed_at DESC'
    c.execute(q, p); rows = [dict(r) for r in c.fetchall()]; conn.close(); return rows


def set_flag(result_id, flagged, note=None):
    """Tandai/hapus tanda 'klasifikasi salah' untuk satu hasil analisis.
    Dipanggil dari tombol flag manual di frontend (Analysis Results)."""
    import datetime as _dt
    conn = get_conn(); c = conn.cursor()
    ts = _dt.datetime.now().isoformat() if flagged else None
    c.execute('UPDATE analysis_results SET flagged=?, flag_note=?, flagged_at=? WHERE id=?',
              (1 if flagged else 0, note, ts, result_id))
    conn.commit(); ok = c.rowcount > 0; conn.close(); return ok


if __name__ == '__main__':
    init_db()
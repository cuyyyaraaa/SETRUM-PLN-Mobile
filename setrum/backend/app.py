import sys, os, threading, time, queue
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, jsonify, request, send_from_directory, send_file, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from backend.database import (init_db, get_dashboard_stats, get_recent_results,
                              get_unanalyzed_reviews, insert_reviews, save_analysis,
                              set_flag, get_all_results)
from backend.agents.orchestrator import analyze_review

load_dotenv()
app = Flask(__name__, static_folder='../frontend/src', static_url_path='')
CORS(app)
PORT = int(os.getenv('FLASK_PORT', 5000))

_job = {'running': False, 'type': None, 'progress': 0, 'total': 0, 'logs': [], 'done': False, 'error': None}
_q = queue.Queue()


def _log(m):
    _job['logs'].append(m); _q.put(m); print(m)


def _reset(t, total=0):
    _job.update({'running': True, 'type': t, 'progress': 0, 'total': total,
                 'logs': [], 'done': False, 'error': None})


@app.route('/')
def index(): return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': '2.0.0'})


@app.route('/api/stats')
def stats():
    try: return jsonify({'success': True, 'data': get_dashboard_stats()})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/reviews')
def reviews_list():
    try:
        lim    = min(int(request.args.get('limit', 50)), 200)
        s      = request.args.get('sentiment')
        layer1 = request.args.get('layer1')
        layer2 = request.args.get('layer2')
        layer3 = request.args.get('layer3')
        layer4 = request.args.get('layer4')
        layer5 = request.args.get('layer5')
        rating = request.args.get('rating')
        d = get_recent_results(limit=lim, sentiment=s,
                               layer1=layer1, layer2=layer2, layer3=layer3,
                               layer4=layer4, layer5=layer5, rating=rating)
        return jsonify({'success': True, 'data': d, 'count': len(d)})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/flag/<int:result_id>', methods=['POST'])
def flag_result(result_id):
    """Tandai/lepas tanda 'klasifikasi salah' pada satu hasil analisis.
    Dipanggil saat user klik ikon flag manual di tabel Analysis Results."""
    try:
        b = request.get_json() or {}
        flagged = bool(b.get('flagged', True))
        note = (b.get('note') or '').strip() or None
        ok = set_flag(result_id, flagged, note)
        if not ok:
            return jsonify({'success': False, 'error': 'Hasil analisis tidak ditemukan'}), 404
        return jsonify({'success': True, 'result_id': result_id, 'flagged': flagged})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export/excel')
def export_excel():
    try:
        import pandas as pd, io
        sentiment = request.args.get('sentiment') or None
        layer1    = request.args.get('layer1') or None
        rows = get_all_results(sentiment=sentiment, layer1=layer1)
        if not rows:
            return jsonify({'success': False, 'error': 'Tidak ada data untuk diekspor'}), 404

        df = pd.DataFrame(rows)
        df['Flag Status'] = df['flagged'].apply(
            lambda v: 'Flagged - Salah Klasifikasi' if v else '')
        df['escalated'] = df['escalated'].apply(lambda v: 'Ya' if v else 'Tidak')

        keep = {
            'username': 'Username', 'review_text': 'Review Text', 'rating': 'Rating',
            'review_date': 'Review Date', 'sentiment': 'Sentiment',
            'sentiment_confidence': 'Confidence (%)', 'layer1': 'Layer 1',
            'layer2': 'Layer 2', 'layer3': 'Layer 3', 'layer4': 'Layer 4',
            'layer5': 'Layer 5', 'quality_param': 'Quality Parameter',
            'reason': 'Reason', 'escalated': 'Escalated', 'analyzed_at': 'Analyzed At',
            'Flag Status': 'Flag Status', 'flag_note': 'Flag Note',
        }
        cols = [c for c in keep if c in df.columns]
        out = df[cols].rename(columns=keep)
        if 'Confidence (%)' in out.columns:
            out['Confidence (%)'] = (out['Confidence (%)'].fillna(0) * 100).round(1)

        buf = io.BytesIO()
        out.to_excel(buf, index=False, engine='openpyxl', sheet_name='Hasil Analisis')
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                          download_name='SETRUM_Hasil_Analisis.xlsx',
                          mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/queue')
def q_count():
    try: return jsonify({'success': True, 'pending_count': len(get_unanalyzed_reviews(limit=5000))})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/job/status')
def job_status():
    return jsonify({**{k: _job[k] for k in ('running','type','progress','total','done','error')},
                    'pct': round(_job['progress']/max(_job['total'],1)*100),
                    'recent_logs': _job['logs'][-50:]})


def _parse_date(s, end=False):
    """'YYYY-MM-DD' -> datetime. end=True -> jam 23:59:59 (inklusif akhir hari)."""
    if not s:
        return None
    try:
        d = datetime.strptime(s, '%Y-%m-%d')
        return d.replace(hour=23, minute=59, second=59) if end else d
    except Exception:
        return None


@app.route('/api/scrape', methods=['POST'])
def scrape():
    if _job['running']: return jsonify({'success': False, 'error': 'Another job is running'}), 409
    b = request.get_json() or {}
    count = min(int(b.get('count', 200)), 2000)
    sort = b.get('sort', 'newest'); rating = int(b.get('rating', 0))
    date_from = b.get('date_from') or None
    date_to   = b.get('date_to') or None
    threading.Thread(target=_run_scrape, args=(count, sort, rating, date_from, date_to), daemon=True).start()
    return jsonify({'success': True, 'message': f'Scraping {count} reviews started'})


def _run_scrape(count, sort_mode, max_rating, date_from=None, date_to=None):
    _reset('scrape', count)
    dt_from = _parse_date(date_from)
    dt_to   = _parse_date(date_to, end=True)
    try:
        from google_play_scraper import reviews, Sort
        APP = 'com.icon.pln123'
        smap = {'newest': Sort.NEWEST, 'rating': Sort.RATING, 'relevance': Sort.MOST_RELEVANT}
        tok, scanned, matched, ins = None, 0, 0, 0
        period = f"{date_from or '...'} s/d {date_to or '...'}" if (dt_from or dt_to) else 'semua periode'
        _log(f'[Scraper] target {count} review LOLOS FILTER (sort={sort_mode}, max_rating={max_rating or "all"}, periode={period})')
        stop_early = False
        # Batas aman scan raw review, biar tidak looping lama kalau filternya memang
        # terlalu sempit (mis. minta 1000 review rating=1 tapi cuma ada 50 di real-nya).
        MAX_SCAN = max(count * 50, 5000)
        while matched < count and not stop_early and scanned < MAX_SCAN:
            res, tok = reviews(APP, lang='id', country='id',
                               sort=smap.get(sort_mode, Sort.NEWEST), count=200,
                               continuation_token=tok)
            if not res: break
            scanned += len(res)
            tr = []
            for r in res:
                rt = r.get('score', 0)
                if max_rating and rt > max_rating: continue
                d = r.get('at')
                if dt_from and d and d < dt_from:
                    # data terurut newest->oldest: begitu sudah lebih tua dari
                    # batas awal periode, sisanya pasti lebih tua lagi -> stop.
                    if sort_mode == 'newest':
                        stop_early = True
                    continue
                if dt_to and d and d > dt_to:
                    continue
                ct = r.get('content', '')
                if not ct or len(ct) < 10: continue
                rid = r.get('reviewId', '')
                if not rid:
                    # reviewId kosong -> JANGAN dimasukkan dgn id='' (nanti salah-anggap
                    # duplikat sesama review yg juga tidak punya reviewId). Skip & catat.
                    _log(f'[Scraper] WARNING: review tanpa reviewId dilewati (rating={rt}, teks="{ct[:40]}...")')
                    continue
                tr.append({'id': rid, 'username': r.get('userName','Anonim'),
                           'review_text': ct, 'rating': rt,
                           'review_date': d.isoformat() if d and hasattr(d,'isoformat') else str(d or ''),
                           'app_version': r.get('appVersion','')})
                matched += 1
                if matched >= count: break
            ins += insert_reviews(tr); _job['progress'] = matched
            _log(f'[Scraper] scan {scanned} raw | lolos filter {matched}/{count} (+{ins} baru disimpan)')
            if not tok: break
            time.sleep(1)
        if matched < count:
            _log(f'[Scraper] Berhenti: cuma {matched}/{count} review yang lolos filter setelah '
                 f'scan {scanned} review mentah. Kemungkinan data dengan kombinasi filter ini '
                 f'memang terbatas jumlahnya di Play Store.')
        _log(f'[Scraper] done. {ins} new reviews.'); _job['done'] = True
    except Exception as e:
        _job['error'] = str(e); _log(f'[Scraper] ERROR: {e}')
    finally:
        _job['running'] = False


@app.route('/api/analyze-all', methods=['POST'])
def analyze_all():
    if _job['running']: return jsonify({'success': False, 'error': 'Another job is running'}), 409
    b = request.get_json() or {}
    lim = min(int(b.get('limit', 100)), 5000)
    threading.Thread(target=_run_analyze, args=(lim,), daemon=True).start()
    return jsonify({'success': True, 'message': f'Analysis of {lim} reviews started'})


def _run_analyze(limit):
    pend = get_unanalyzed_reviews(limit=limit)
    _reset('analyze', len(pend))
    if not pend:
        _log('[Analyzer] no pending reviews'); _job['done'] = True; _job['running'] = False; return
    _log(f'[Analyzer] processing {len(pend)} reviews')
    ok = err = comp = 0
    for i, rv in enumerate(pend, 1):
        try:
            res = analyze_review(rv['review_text'], rating=rv.get('rating'))
            save_analysis(rv['id'], res); ok += 1
            if res.get('is_complaint'): comp += 1
            tag = 'KELUHAN' if res.get('is_complaint') else res['sentiment']
            _log(f"[{i}/{len(pend)}] {tag} | {res.get('category','-')} > {res.get('subcategory','-')}")
        except Exception as e:
            err += 1; _log(f'[{i}/{len(pend)}] ERROR: {str(e)[:60]}')
        _job['progress'] = i
        time.sleep(0.2)
    _log(f'[Analyzer] done. {ok} ok, {comp} negative complaints, {err} errors.')
    _job['done'] = True; _job['running'] = False


@app.route('/api/analyze', methods=['POST'])
def analyze_one():
    try:
        b = request.get_json()
        if not b or 'text' not in b: return jsonify({'success': False, 'error': 'text is required'}), 400
        t = b['text'].strip()
        if len(t) < 5: return jsonify({'success': False, 'error': 'text too short'}), 400
        rating = b.get('rating')
        return jsonify({'success': True, 'data': analyze_review(t, rating=rating, verbose=True)})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/layer-stats')
def layer_stats():
    """Drill-down keluhan: Layer1 -> Layer2 -> Layer3 (hanya ulasan negatif)."""
    try:
        from backend.database import get_conn
        conn = get_conn(); c = conn.cursor()
        c.execute("""SELECT
            COALESCE(NULLIF(layer1,''),'Umum') l1,
            COALESCE(NULLIF(layer2,''),'Umum') l2,
            COALESCE(NULLIF(layer3,''), NULLIF(subcategory,''),'Umum') l3,
            COUNT(*) cnt
            FROM analysis_results WHERE is_complaint=1
            GROUP BY l1,l2,l3 ORDER BY cnt DESC""")
        rows = c.fetchall(); conn.close()
        l1map, l2map, l3map = {}, {}, {}
        for r in rows:
            l1, l2, l3, cnt = r['l1'], r['l2'], r['l3'], r['cnt']
            l1map[l1] = l1map.get(l1, 0) + cnt
            l2map.setdefault(l1, {})
            l2map[l1][l2] = l2map[l1].get(l2, 0) + cnt
            k = f"{l1}||{l2}"
            l3map.setdefault(k, {})
            l3map[k][l3] = l3map[k].get(l3, 0) + cnt
        tolist = lambda d: [{'label': k, 'count': v} for k, v in sorted(d.items(), key=lambda x: -x[1])]
        return jsonify({'success': True,
                        'layer1': tolist(l1map),
                        'layer2': {k: tolist(v) for k, v in l2map.items()},
                        'layer3': {k: tolist(v) for k, v in l3map.items()}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/raw-reviews')
def raw_reviews():
    """Data mentah hasil scraping, berhalaman + filter."""
    try:
        from backend.database import get_conn
        conn = get_conn(); c = conn.cursor()
        page = max(int(request.args.get('page', 1)), 1)
        per = min(int(request.args.get('per_page', 25)), 200)
        off = (page - 1) * per
        rating = request.args.get('rating', '')
        search = request.args.get('search', '')
        analyzed = request.args.get('analyzed', '')
        q = 'SELECT id, username, review_text, rating, review_date, analyzed FROM reviews WHERE 1=1'
        cq = 'SELECT COUNT(*) t FROM reviews WHERE 1=1'
        p = []
        if rating: q += ' AND rating=?'; cq += ' AND rating=?'; p.append(int(rating))
        if search:
            q += ' AND review_text LIKE ?'; cq += ' AND review_text LIKE ?'; p.append(f'%{search}%')
        if analyzed in ('0', '1'):
            q += ' AND analyzed=?'; cq += ' AND analyzed=?'; p.append(int(analyzed))
        c.execute(cq, p); total = c.fetchone()['t']
        q += ' ORDER BY review_date DESC LIMIT ? OFFSET ?'
        c.execute(q, p + [per, off])
        rows = []
        for r in c.fetchall():
            d = dict(r)
            ds = (d.get('review_date') or '').replace('T', ' ')
            d['tanggal'] = ds[:10] if ds and ds != 'None' else '-'
            rows.append(d)
        conn.close()
        return jsonify({'success': True, 'data': rows, 'total': total, 'page': page,
                        'per_page': per, 'total_pages': max(1, -(-total // per))})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/stream')
def log_stream():
    def gen():
        for ln in _job['logs'][-50:]: yield f"data: {ln}\n\n"
        while True:
            try: yield f"data: {_q.get(timeout=30)}\n\n"
            except queue.Empty: yield "data: \n\n"
    return Response(stream_with_context(gen()), content_type='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    print('='*55, '\nSETRUM v2.0 — PLN Mobile Review Analyzer\n', '='*55)
    init_db()
    print(f'\nDashboard: http://localhost:{PORT}/\n')
    app.run(debug=False, port=PORT, host='0.0.0.0', threaded=True)
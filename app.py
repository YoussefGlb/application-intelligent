# FlaskGPT : assistant IA avec Flask + Groq (API compatible OpenAI) + SQLite

# Importe le type Iterator, qui représente un objet fournissant des valeurs une par une.
from collections.abc import Iterator

# Modules standards : variables d'environnement, JSON, base SQLite, temps, identifiants
import os
import re
import json
import math
import sqlite3
import time
import uuid
import unicodedata
from collections import Counter
from datetime import datetime, timezone

# Importe le client OpenAI et ses erreurs (Groq utilise la même API)
from openai import (OpenAI, APIError, APIConnectionError, AuthenticationError,
                    NotFoundError, RateLimitError)

# Permet de charger les variables du fichier .env.
from dotenv import load_dotenv

from flask import Flask, render_template, request, Response, jsonify, g

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("FLASKGPT_DB", os.path.join(BASE_DIR, "chats.db"))
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

DEFAULT_MODEL = "openai/gpt-oss-20b"
TITLE_MODEL = "llama-3.1-8b-instant"   # petit modèle rapide pour les titres
MAX_FILE_CHARS = 100_000               # taille max d'un fichier joint
SEPARATOR = "\x1e"                     # sépare le texte de la réponse des infos (tokens, erreurs)

# Liste utilisée si Groq ne répond pas à /models
FALLBACK_MODELS = [
    {"id": "openai/gpt-oss-20b", "context": 131072},
    {"id": "openai/gpt-oss-120b", "context": 131072},
    {"id": "llama-3.3-70b-versatile", "context": 131072},
    {"id": "llama-3.1-8b-instant", "context": 131072},
]

# Consignes ajoutées avant le prompt système choisi par l'utilisateur
BASE_INSTRUCTIONS = (
    "Tu es FlaskGPT, un assistant utile. Réponds en Markdown, dans la langue de l'utilisateur. "
    "Quand un schéma aide à comprendre (étapes, architecture, flux), ajoute un bloc ```mermaid valide."
)

#créer l'application web Flask et la stocke dans la variable app
app = Flask(__name__)

# Le client est créé au premier besoin : l'appli démarre même sans clé.
_client = None


def get_client():
    """Retourne le client Groq, ou None si la clé est absente du .env."""
    global _client
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    if _client is None:
        _client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Base de données SQLite
# ---------------------------------------------------------------------------
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    """Une connexion SQLite par requête HTTP."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def fts5_available():
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False


# Si SQLite n'a pas FTS5, on calcule BM25 nous-mêmes en Python
FTS5 = fts5_available()


def init_db():
    """Crée les tables si elles n'existent pas encore."""
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT 'Nouvelle conversation',
                pinned      INTEGER NOT NULL DEFAULT 0,
                model       TEXT,
                system      TEXT,
                temperature REAL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id  TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                role     TEXT NOT NULL,
                content  TEXT NOT NULL,
                files    TEXT,
                meta     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, position);

            -- RAG : documents de l'utilisateur découpés en extraits (chunks)
            CREATE TABLE IF NOT EXISTS documents (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                pages      INTEGER NOT NULL,
                chars      INTEGER NOT NULL,
                active     INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page     INTEGER NOT NULL,
                position INTEGER NOT NULL,
                text     TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        """)
        # Ancienne base : ajoute la colonne "rag" si elle n'existe pas encore
        columns = [row[1] for row in db.execute("PRAGMA table_info(chats)")]
        if "rag" not in columns:
            db.execute("ALTER TABLE chats ADD COLUMN rag INTEGER NOT NULL DEFAULT 0")
        # Index plein texte FTS5 (moteur BM25 intégré à SQLite), s'il est disponible
        if FTS5:
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, content='chunks', content_rowid='id',
                tokenize='unicode61 remove_diacritics 2')""")


init_db()


def chat_to_dict(row):
    return {
        "id": row["id"],
        "title": row["title"],
        "pinned": bool(row["pinned"]),
        "model": row["model"],
        "system": row["system"],
        "temperature": row["temperature"],
        "rag": bool(row["rag"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def find_chat(chat_id):
    return get_db().execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()


def read_temperature(value, default=0.7):
    try:
        return min(2.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
#on met un route en mentionnant la racine comme page d'accueil
@app.route('/')
def home():
    return render_template('index.html')


@app.get("/api/status")
def status():
    return jsonify(has_key=get_client() is not None, default_model=DEFAULT_MODEL)


# Cache de la liste des modèles (évite d'appeler Groq à chaque chargement)
_models_cache = {"time": 0, "data": None}


@app.get("/api/models")
def list_models():
    if _models_cache["data"] and time.time() - _models_cache["time"] < 600:
        return jsonify(models=_models_cache["data"])

    models = FALLBACK_MODELS
    client = get_client()
    if client is not None:
        try:
            exclude = ("whisper", "tts", "guard", "playai", "orpheus", "distil")
            live = []
            for m in client.models.list():
                extra = getattr(m, "model_extra", None) or {}
                if any(word in m.id.lower() for word in exclude) or extra.get("active") is False:
                    continue
                live.append({"id": m.id, "context": extra.get("context_window") or 8192})
            if live:
                # Les modèles connus en premier, puis les autres par ordre alphabétique
                known = [f["id"] for f in FALLBACK_MODELS]
                live.sort(key=lambda m: (known.index(m["id"]) if m["id"] in known else 99, m["id"]))
                models = live
        except Exception:
            pass

    _models_cache.update(time=time.time(), data=models)
    return jsonify(models=models)


# ---------------------------------------------------------------------------
# Conversations (SQLite)
# ---------------------------------------------------------------------------
@app.get("/api/chats")
def list_chats():
    """Liste des conversations ; ?q= pour chercher dans les titres et les messages."""
    q = (request.args.get("q") or "").strip()
    db = get_db()
    if not q:
        rows = db.execute(
            "SELECT * FROM chats ORDER BY pinned DESC, updated_at DESC").fetchall()
        return jsonify(chats=[chat_to_dict(r) for r in rows])

    like = f"%{q}%"
    rows = db.execute("""
        SELECT c.*,
               (SELECT m.content FROM messages m
                 WHERE m.chat_id = c.id AND m.content LIKE ?
                 ORDER BY m.position LIMIT 1) AS match
          FROM chats c
         WHERE c.title LIKE ?
            OR EXISTS (SELECT 1 FROM messages m WHERE m.chat_id = c.id AND m.content LIKE ?)
         ORDER BY c.pinned DESC, c.updated_at DESC
    """, (like, like, like)).fetchall()

    result = []
    for r in rows:
        item = chat_to_dict(r)
        if r["match"]:
            # Petit extrait autour du mot trouvé
            text = r["match"]
            i = max(0, text.lower().find(q.lower()))
            start = max(0, i - 30)
            item["snippet"] = ("…" if start else "") + text[start:i + len(q) + 50].replace("\n", " ")
        result.append(item)
    return jsonify(chats=result)


@app.post("/api/chats")
def create_chat():
    data = request.get_json(silent=True) or {}
    chat_id = uuid.uuid4().hex
    t = now()
    db = get_db()
    db.execute(
        "INSERT INTO chats (id, model, system, temperature, rag, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chat_id, str(data.get("model") or DEFAULT_MODEL)[:100], str(data.get("system") or "")[:4000],
         read_temperature(data.get("temperature")), 1 if data.get("rag") else 0, t, t))
    db.commit()
    return jsonify(chat_to_dict(find_chat(chat_id))), 201


@app.get("/api/chats/<chat_id>")
def get_chat(chat_id):
    row = find_chat(chat_id)
    if row is None:
        return jsonify(error="Conversation introuvable."), 404
    msgs = get_db().execute(
        "SELECT role, content, files, meta FROM messages WHERE chat_id = ? ORDER BY position",
        (chat_id,)).fetchall()
    chat = chat_to_dict(row)
    chat["messages"] = [{
        "role": m["role"],
        "content": m["content"],
        "files": json.loads(m["files"]) if m["files"] else [],
        "meta": json.loads(m["meta"]) if m["meta"] else None,
    } for m in msgs]
    return jsonify(chat)


@app.patch("/api/chats/<chat_id>")
def update_chat(chat_id):
    """Renommer, épingler, changer le modèle / prompt système / température."""
    if find_chat(chat_id) is None:
        return jsonify(error="Conversation introuvable."), 404
    data = request.get_json(silent=True) or {}
    fields, values = [], []
    if isinstance(data.get("title"), str) and data["title"].strip():
        fields.append("title = ?"); values.append(data["title"].strip()[:100])
    if "pinned" in data:
        fields.append("pinned = ?"); values.append(1 if data["pinned"] else 0)
    if isinstance(data.get("model"), str):
        fields.append("model = ?"); values.append(data["model"][:100])
    if isinstance(data.get("system"), str):
        fields.append("system = ?"); values.append(data["system"][:4000])
    if "temperature" in data:
        fields.append("temperature = ?"); values.append(read_temperature(data["temperature"]))
    if "rag" in data:
        fields.append("rag = ?"); values.append(1 if data["rag"] else 0)
    if fields:
        db = get_db()
        db.execute(f"UPDATE chats SET {', '.join(fields)} WHERE id = ?", (*values, chat_id))
        db.commit()
    return jsonify(chat_to_dict(find_chat(chat_id)))


@app.delete("/api/chats/<chat_id>")
def delete_chat(chat_id):
    db = get_db()
    db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    db.commit()
    return jsonify(ok=True)


@app.put("/api/chats/<chat_id>/messages")
def save_messages(chat_id):
    """Remplace tous les messages de la conversation (simple et fiable)."""
    if find_chat(chat_id) is None:
        return jsonify(error="Conversation introuvable."), 404
    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    if not isinstance(messages, list):
        return jsonify(error="'messages' doit être une liste."), 400

    db = get_db()
    db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        db.execute(
            "INSERT INTO messages (chat_id, position, role, content, files, meta) VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, i, m["role"], str(m.get("content") or ""),
             json.dumps(m.get("files") or [], ensure_ascii=False),
             json.dumps(m["meta"], ensure_ascii=False) if m.get("meta") else None))
    db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now(), chat_id))
    db.commit()
    return jsonify(ok=True)


@app.post("/api/chats/<chat_id>/title")
def generate_title(chat_id):
    """Donne un titre court à la conversation à partir de la première question."""
    db = get_db()
    first = db.execute(
        "SELECT content, files FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY position LIMIT 1",
        (chat_id,)).fetchone()
    if first is None:
        return jsonify(error="Aucun message."), 400

    question = first["content"].strip()
    if not question and first["files"]:
        files = json.loads(first["files"])
        question = "Fichier : " + ", ".join(f.get("name", "") for f in files)
    title = question.splitlines()[0][:50] if question else "Nouvelle conversation"

    client = get_client()
    if client is not None:
        try:
            r = client.chat.completions.create(
                model=TITLE_MODEL,
                messages=[
                    {"role": "system", "content": "Donne un titre très court (3 à 6 mots) pour cette question. "
                                                  "Pas de guillemets, pas de point final, même langue que la question."},
                    {"role": "user", "content": question[:1500]},
                ],
                temperature=0.3,
                max_tokens=20,
            )
            t = (r.choices[0].message.content or "").strip().strip('"«» ').splitlines()
            if t and t[0].strip():
                title = t[0].strip()[:60]
        except Exception:
            pass

    db.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
    db.commit()
    return jsonify(title=title)


@app.get("/api/chats/<chat_id>/export")
def export_chat(chat_id):
    """Télécharge la conversation au format Markdown."""
    row = find_chat(chat_id)
    if row is None:
        return jsonify(error="Conversation introuvable."), 404
    msgs = get_db().execute(
        "SELECT role, content, files FROM messages WHERE chat_id = ? ORDER BY position",
        (chat_id,)).fetchall()

    lines = [f"# {row['title']}", "",
             f"_Exporté depuis FlaskGPT le {datetime.now().strftime('%d/%m/%Y %H:%M')} — modèle : {row['model']}_", ""]
    for m in msgs:
        lines.append("## 🧑 Vous" if m["role"] == "user" else "## 🤖 Assistant")
        for f in json.loads(m["files"] or "[]"):
            lines.append(f"📎 _{f.get('name', 'fichier')}_")
        lines += ["", m["content"], ""]

    filename = "".join(c if c.isalnum() or c in " -_" else "_" for c in row["title"]).strip() or "conversation"
    return Response("\n".join(lines), mimetype="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{filename}.md"'})


# ---------------------------------------------------------------------------
# RAG : base de connaissances (documents de l'utilisateur)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 900        # taille d'un extrait (caractères)
CHUNK_OVERLAP = 150     # chevauchement entre deux extraits, pour ne pas couper une idée
TOP_K = 6               # nombre d'extraits envoyés au modèle

STOPWORDS = set("""
le la les un une des du de d l au aux et ou en dans sur pour par avec sans ce cet cette ces qui que quoi
dont est sont etre avoir a as ont il elle ils elles on nous vous je tu me te se mon ma mes ton ta tes son
sa ses leur leurs notre votre plus moins tres pas ne n y c s qu quel quelle quels quelles comment pourquoi
quand peux peut faire fait explique expliquer donne donner moi stp svp cours document
the a an and or of to in on for with without is are was were be been it this that these those what which
who how why when can could do does did me my your you i we they about from as at by not
""".split())


def normalize(text: str) -> str:
    """minuscules + sans accents : « Réseau » -> « reseau »"""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def keywords(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", normalize(text))
    seen, result = set(), []
    for w in words:
        if len(w) >= 2 and w not in STOPWORDS and w not in seen:
            seen.add(w)
            result.append(w)
    return result


def split_into_chunks(text: str) -> list[str]:
    """Découpe une page en extraits d'environ CHUNK_SIZE caractères, en coupant de préférence en fin de phrase."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if end < len(text):
            cut = max(text.rfind("\n", start + CHUNK_SIZE // 2, end), text.rfind(". ", start + CHUNK_SIZE // 2, end))
            if cut > start:
                end = cut + 1
        chunk = text[start:end].strip()
        if len(chunk) > 20:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def doc_to_dict(row, chunk_count=None):
    return {
        "id": row["id"], "name": row["name"], "pages": row["pages"], "chars": row["chars"],
        "active": bool(row["active"]), "created_at": row["created_at"], "chunks": chunk_count,
    }


@app.get("/api/documents")
def list_documents():
    rows = get_db().execute("""
        SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS n
          FROM documents d ORDER BY d.created_at DESC""").fetchall()
    return jsonify(documents=[doc_to_dict(r, r["n"]) for r in rows],
                   engine="BM25 (SQLite FTS5)" if FTS5 else "BM25 (Python)")


@app.post("/api/documents")
def add_document():
    """Reçoit le texte d'un document (une entrée par page), le découpe et l'indexe."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name") or "document").strip()[:200]
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages or not all(isinstance(p, str) for p in pages):
        return jsonify(error="'pages' doit être une liste de textes."), 400
    pages = pages[:1000]
    if sum(len(p) for p in pages) > 5_000_000:
        return jsonify(error="Document trop volumineux (5 millions de caractères max)."), 413

    doc_id = uuid.uuid4().hex
    db = get_db()
    db.execute("INSERT INTO documents (id, name, pages, chars, created_at) VALUES (?, ?, ?, ?, ?)",
               (doc_id, name, len(pages), sum(len(p) for p in pages), now()))
    count = 0
    for page_number, page_text in enumerate(pages, start=1):
        for chunk in split_into_chunks(page_text):
            cur = db.execute("INSERT INTO chunks (doc_id, page, position, text) VALUES (?, ?, ?, ?)",
                             (doc_id, page_number, count, chunk))
            if FTS5:
                db.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (cur.lastrowid, chunk))
            count += 1
    if count == 0:
        db.rollback()
        return jsonify(error="Aucun texte exploitable dans ce document (PDF scanné ?)."), 400
    db.commit()
    row = db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return jsonify(doc_to_dict(row, count)), 201


@app.patch("/api/documents/<doc_id>")
def update_document(doc_id):
    data = request.get_json(silent=True) or {}
    db = get_db()
    if "active" in data:
        db.execute("UPDATE documents SET active = ? WHERE id = ?", (1 if data["active"] else 0, doc_id))
        db.commit()
    return jsonify(ok=True)


@app.delete("/api/documents/<doc_id>")
def delete_document(doc_id):
    db = get_db()
    if FTS5:
        # Avec une table FTS5 "external content", il faut retirer chaque extrait de l'index
        for c in db.execute("SELECT id, text FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall():
            db.execute("INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', ?, ?)", (c["id"], c["text"]))
    db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    db.commit()
    return jsonify(ok=True)


def expand_query(question: str, previous: str = "") -> list[str]:
    """Demande à un petit modèle des mots-clés et synonymes (FR + EN) pour mieux retrouver les passages."""
    client = get_client()
    if client is None:
        return []
    try:
        r = client.chat.completions.create(
            model=TITLE_MODEL,
            messages=[
                {"role": "system", "content":
                    "Tu aides un moteur de recherche par mots-clés. À partir de la question, donne 8 mots-clés "
                    "ou termes techniques (synonymes, traductions FR/EN, notions liées) qui apparaîtraient dans "
                    "un cours contenant la réponse. Réponds uniquement par les mots-clés séparés par des virgules."},
                {"role": "user", "content": (f"Question précédente : {previous[:500]}\n" if previous else "") +
                                            f"Question : {question[:1500]}"},
            ],
            temperature=0.2,
            max_tokens=80,
        )
        return keywords(r.choices[0].message.content or "")[:16]
    except Exception:
        return []


def bm25_python(db, terms, limit):
    """BM25 calculé en Python (utilisé seulement si SQLite n'a pas FTS5)."""
    rows = db.execute("""SELECT c.id, c.doc_id, c.page, c.text, d.name FROM chunks c
                         JOIN documents d ON d.id = c.doc_id WHERE d.active = 1""").fetchall()
    if not rows:
        return []
    docs = [Counter(re.findall(r"[a-z0-9]+", normalize(r["text"]))) for r in rows]
    n = len(docs)
    avg_len = sum(sum(d.values()) for d in docs) / n
    k1, b = 1.5, 0.75
    scored = []
    for row, d in zip(rows, docs):
        length = sum(d.values())
        score = 0.0
        for t in terms:
            if t in d:
                df = sum(1 for x in docs if t in x)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * d[t] * (k1 + 1) / (d[t] + k1 * (1 - b + b * length / avg_len))
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda x: -x[0])
    return [{"chunk_id": r["id"], "doc_id": r["doc_id"], "doc": r["name"], "page": r["page"],
             "text": r["text"], "score": round(s, 2)} for s, r in scored[:limit]]


def retrieve(question: str, previous: str = "", limit: int = TOP_K):
    """Étape « R » du RAG : retrouve les extraits les plus pertinents pour la question."""
    base_terms = keywords(question)
    extra_terms = [t for t in expand_query(question, previous) if t not in base_terms]
    terms = (base_terms + extra_terms)[:30]
    if not terms:
        return terms, []

    db = get_db()
    if not FTS5:
        return terms, bm25_python(db, terms, limit)

    match = " OR ".join(f'"{t}"' for t in terms)
    rows = db.execute("""
        SELECT c.id, c.doc_id, c.page, c.text, d.name, bm25(chunks_fts) AS score
          FROM chunks_fts
          JOIN chunks c ON c.id = chunks_fts.rowid
          JOIN documents d ON d.id = c.doc_id
         WHERE chunks_fts MATCH ? AND d.active = 1
         ORDER BY score
         LIMIT ?""", (match, limit)).fetchall()
    # bm25() de SQLite renvoie un score négatif : plus il est petit, plus c'est pertinent
    return terms, [{"chunk_id": r["id"], "doc_id": r["doc_id"], "doc": r["name"], "page": r["page"],
                    "text": r["text"], "score": round(-r["score"], 2)} for r in rows]


@app.get("/api/documents/search")
def search_documents():
    """Permet de tester la recherche depuis l'interface (et de voir ce que le modèle recevra)."""
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify(terms=[], results=[])
    start = time.perf_counter()
    terms, results = retrieve(q)
    return jsonify(terms=terms, results=results, time=round(time.perf_counter() - start, 2))


def build_rag_prompt(results) -> str:
    """Étape « A » du RAG : on ajoute les extraits trouvés au prompt système."""
    if not results:
        return ("\n\nL'utilisateur a activé la recherche dans ses documents, mais aucun extrait pertinent "
                "n'a été trouvé. Dis-le clairement, puis réponds avec tes connaissances générales en le signalant.")
    blocks = [f"[{i}] (document : {r['doc']}, page {r['page']})\n{r['text']}" for i, r in enumerate(results, start=1)]
    return ("\n\nTu réponds à partir des extraits des documents de l'utilisateur ci-dessous. "
            "Cite tes sources avec leur numéro entre crochets, par exemple [1] ou [2][3], juste après l'information. "
            "N'invente pas de numéro. Si les extraits ne contiennent pas la réponse, dis-le clairement puis "
            "réponds avec tes connaissances générales en le précisant.\n\n"
            "=== EXTRAITS ===\n" + "\n\n".join(blocks) + "\n=== FIN DES EXTRAITS ===")


# ---------------------------------------------------------------------------
# Envoi d'un message au modèle (réponse en streaming)
# ---------------------------------------------------------------------------
# Reçoit la conversation envoyée par JavaScript.
@app.route('/prompt', methods=['POST'])
def prompt():
    # Vérifie que la requête contient des données au format JSON.
    if not request.is_json:
        return jsonify(error="Le format JSON est requis."), 415

    # Lit le JSON ; retourne None si son contenu est invalide.
    data = request.get_json(silent=True)

    # Vérifie que le JSON représente un objet.
    if not isinstance(data, dict):
        return jsonify(error="Le JSON doit être un objet valide."), 400

    # Récupère les messages, ou None si la clé est absente.
    messages = data.get("messages")

    # Vérifie que les messages constituent une liste non vide.
    if not isinstance(messages, list) or not messages:
        return jsonify(error="'messages' doit être une liste non vide."), 400

    # Transforme les messages en dictionnaires avec leurs rôles.
    try:
        conversation = build_conversation_dict(messages=messages)
    except ValueError as e:
        return jsonify(error=str(e)), 400

    if get_client() is None:
        return jsonify(error="La clé GROQ_API_KEY est absente du fichier .env."), 500

    # Paramètres choisis dans l'interface
    model = data.get("model") if isinstance(data.get("model"), str) and data.get("model") else DEFAULT_MODEL
    temperature = read_temperature(data.get("temperature"))
    system = data.get("system") if isinstance(data.get("system"), str) else ""
    system_prompt = BASE_INSTRUCTIONS + ("\n\n" + system.strip()[:4000] if system.strip() else "")

    # RAG : recherche des extraits utiles dans les documents de l'utilisateur
    extra_meta = None
    if data.get("rag"):
        start = time.perf_counter()
        questions = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        question = questions[-1] if questions else ""
        previous = questions[-2] if len(questions) > 1 else ""
        terms, results = retrieve(question or conversation[-1]["content"][:2000], previous)
        system_prompt += build_rag_prompt(results)
        extra_meta = {
            "rag": {"terms": terms, "time": round(time.perf_counter() - start, 2)},
            "sources": [{"n": i, "doc": r["doc"], "page": r["page"], "score": r["score"], "text": r["text"]}
                        for i, r in enumerate(results, start=1)],
        }

    full_conversation = [{"role": "system", "content": system_prompt}] + conversation

    # Renvoie progressivement la réponse sous forme de texte brut.
    return Response(
        event_stream(full_conversation, model=model, temperature=temperature, extra_meta=extra_meta),
        content_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def read_usage(chunk):
    """Groq envoie les tokens utilisés dans le dernier fragment (x_groq.usage)."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        extra = getattr(chunk, "model_extra", None) or {}
        x_groq = extra.get("x_groq") or getattr(chunk, "x_groq", None)
        if isinstance(x_groq, dict):
            usage = x_groq.get("usage")
        elif x_groq is not None:
            usage = getattr(x_groq, "usage", None)
    if usage is None:
        return None
    if not isinstance(usage, dict):
        usage = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    return usage


def estimate_tokens(text: str) -> int:
    # Approximation : environ 4 caractères par token
    return max(1, round(len(text) / 4))


# Génère progressivement les fragments de texte de la réponse.
def event_stream(conversation: list[dict], model: str = DEFAULT_MODEL,
                 temperature: float = 0.7, extra_meta: dict | None = None) -> Iterator[str]:
    start = time.perf_counter()
    first_token = None
    usage = None
    produced = ""
    response = None
    try:
        # Envoie la conversation au modèle et active la réception progressive.
        response = get_client().chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature,
            stream=True,
        )

        # Parcourt les fragments reçus depuis l'API.
        for chunk in response:
            usage = read_usage(chunk) or usage

            # Ignore les fragments qui ne contiennent aucun choix.
            if not chunk.choices:
                continue

            # Récupère le texte du fragment, ou une chaîne vide.
            text = chunk.choices[0].delta.content or ""

            # Transmet le texte à Flask, qui l'envoie au navigateur.
            if text:
                if first_token is None:
                    first_token = time.perf_counter()
                produced += text
                yield text

    except AuthenticationError:
        yield SEPARATOR + json.dumps({"error": "Clé API Groq invalide. Vérifie GROQ_API_KEY dans le fichier .env."})
        return
    except RateLimitError:
        yield SEPARATOR + json.dumps({"error": "Limite Groq atteinte. Attends un peu ou choisis un autre modèle."})
        return
    except NotFoundError:
        yield SEPARATOR + json.dumps({"error": f"Le modèle « {model} » n'existe pas (ou plus) chez Groq."})
        return
    except APIConnectionError:
        yield SEPARATOR + json.dumps({"error": "Impossible de joindre Groq. Vérifie ta connexion internet."})
        return
    except APIError as e:
        yield SEPARATOR + json.dumps({"error": f"Erreur Groq : {e.message}"})
        return
    except Exception as e:
        yield SEPARATOR + json.dumps({"error": f"Erreur serveur : {e}"})
        return
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    # À la fin : infos sur les tokens et la vitesse
    end = time.perf_counter()
    meta = {"model": model, "duration": round(end - start, 2), "estimated": usage is None}
    if usage:
        meta["prompt_tokens"] = usage.get("prompt_tokens") or 0
        meta["completion_tokens"] = usage.get("completion_tokens") or 0
    else:
        meta["prompt_tokens"] = sum(estimate_tokens(m["content"]) for m in conversation)
        meta["completion_tokens"] = estimate_tokens(produced)
    meta["total_tokens"] = meta["prompt_tokens"] + meta["completion_tokens"]
    generation_time = end - (first_token or start)
    if generation_time > 0:
        meta["tokens_per_s"] = round(meta["completion_tokens"] / generation_time, 1)
    if extra_meta:
        meta.update(extra_meta)
    yield SEPARATOR + json.dumps(meta)


# Transforme les messages en une liste de dictionnaires contenant le rôle et le contenu.
def build_conversation_dict(messages: list) -> list[dict]:
    # Ancien format du TP : une liste de textes -> rôles alternés user / assistant
    if all(isinstance(m, str) for m in messages):
        if any(not m.strip() for m in messages):
            raise ValueError("Chaque message doit être un texte non vide.")
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": message}
            for i, message in enumerate(messages)
        ]

    # Nouveau format : chaque message indique son rôle (plus de décalage user/assistant)
    conversation = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            raise ValueError("Chaque message doit avoir un rôle 'user' ou 'assistant'.")
        content = m.get("content") or ""
        if not isinstance(content, str):
            raise ValueError("Le contenu d'un message doit être un texte.")

        # Les fichiers joints sont ajoutés au texte envoyé au modèle
        parts = []
        for f in m.get("files") or []:
            if isinstance(f, dict) and isinstance(f.get("text"), str):
                name = str(f.get("name") or "fichier")[:200]
                parts.append(f"Fichier joint « {name} » :\n````\n{f['text'][:MAX_FILE_CHARS]}\n````")
        if content.strip():
            parts.append(content)
        if not parts:
            continue
        conversation.append({"role": m["role"], "content": "\n\n".join(parts)})

    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError("Le dernier message doit venir de l'utilisateur.")
    return conversation


#démarrer le serveur lorsque tu lances directement le fichier
if __name__ == "__main__":
    app.run(debug=True)

// =====================================================================
// FlaskGPT — interface du chat (JavaScript sans framework)
// =====================================================================

// Caractère qui sépare le texte de la réponse des infos envoyées à la fin (tokens, erreur)
const SEPARATOR = "\x1e";
const MAX_FILE_CHARS = 100000;
const MAX_FILES = 5;

// Personnalités prédéfinies (prompt système)
const PERSONAS = [
    { name: "🤖 Assistant général", prompt: "" },
    { name: "📐 Prof de maths", prompt: "Tu es un professeur de mathématiques patient. Explique étape par étape avec des exemples simples, et vérifie les calculs." },
    { name: "🌍 Traducteur FR ↔ EN", prompt: "Tu es un traducteur. Si le texte est en français, traduis-le en anglais ; sinon traduis-le en français. Donne la traduction, puis une courte note seulement si une nuance est importante." },
    { name: "🐍 Développeur Python", prompt: "Tu es un développeur Python senior. Donne du code propre, commenté et testé, puis explique brièvement tes choix." },
    { name: "📊 Data scientist", prompt: "Tu es un data scientist expérimenté. Propose des analyses claires avec pandas / scikit-learn, et explique les résultats simplement." },
    { name: "🎤 Recruteur (entretien)", prompt: "Tu es un recruteur tech. Simule un entretien : pose une question à la fois, attends ma réponse, puis donne un feedback honnête avant la question suivante." },
    { name: "✍️ Correcteur de texte", prompt: "Tu corriges l'orthographe, la grammaire et le style du texte donné. Rends le texte corrigé puis liste les principales corrections." },
];

// État de l'application
const state = {
    chats: [],            // liste des conversations (barre latérale)
    chat: null,           // conversation ouverte
    messages: [],         // messages de la conversation ouverte
    models: [],           // modèles disponibles chez Groq
    settings: loadDefaults(),
    files: [],            // fichiers joints en attente d'envoi
    streaming: false,
    controller: null,     // permet d'arrêter la réponse
    hasKey: true,
    documents: [],        // documents de la base de connaissances (RAG)
};

// Raccourci pour sélectionner un élément
const $ = (selector, root = document) => root.querySelector(selector);

// =====================================================================
// Petits utilitaires
// =====================================================================
function loadDefaults() {
    const base = { model: "openai/gpt-oss-20b", system: "", temperature: 0.7, rag: false };
    try {
        return { ...base, ...JSON.parse(localStorage.getItem("flaskgpt:defaults") || "{}") };
    } catch (e) {
        return base;
    }
}

function saveDefaults(settings) {
    try { localStorage.setItem("flaskgpt:defaults", JSON.stringify(settings)); } catch (e) {}
}

// Appel à l'API Flask qui renvoie du JSON
async function api(url, options = {}) {
    const response = await fetch(url, {
        ...options,
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        body: options.body && typeof options.body !== "string" ? JSON.stringify(options.body) : options.body,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Erreur ${response.status}`);
    return data;
}

// Formate un nombre à la française : 12 345
const fmt = (n) => Number(n || 0).toLocaleString("fr-FR");

// Estimation rapide : environ 4 caractères = 1 token
const estimateTokens = (text) => Math.round((text || "").length / 4);

// "openai/gpt-oss-20b" -> "gpt-oss-20b"
const shortModel = (id) => (id || "").split("/").pop();

const isDark = () => document.documentElement.classList.contains("dark");

let toastTimer;
function toast(message) {
    const el = $("#toast");
    el.textContent = message;
    el.classList.remove("opacity-0");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("opacity-0"), 2200);
}

async function copyText(text) {
    try {
        await navigator.clipboard.writeText(text);
    } catch (e) {
        // Méthode de secours si le presse-papier n'est pas accessible
        const area = document.createElement("textarea");
        area.value = text;
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        area.remove();
    }
    toast("Copié dans le presse-papier ✓");
}

// Charge un script externe une seule fois (ex. : pdf.js seulement si on joint un PDF)
const loadedScripts = {};
function loadScript(src) {
    if (!loadedScripts[src]) {
        loadedScripts[src] = new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.src = src;
            s.onload = resolve;
            s.onerror = () => reject(new Error("Impossible de charger " + src));
            document.head.appendChild(s);
        });
    }
    return loadedScripts[src];
}

function contextWindow(modelId) {
    const m = state.models.find((x) => x.id === modelId);
    return (m && m.context) || 131072;
}

// =====================================================================
// Markdown -> HTML (+ coloration du code, schémas Mermaid)
// =====================================================================
let converter;
function markdownToHtml(text) {
    if (!converter) {
        converter = new showdown.Converter({
            tables: true, strikethrough: true, tasklists: true,
            openLinksInNewWindow: true, ghCodeBlocks: true, emoji: true,
        });
    }
    // DOMPurify retire tout HTML dangereux venant de la réponse
    return DOMPurify.sanitize(converter.makeHtml(text), { ADD_ATTR: ["target"] });
}

const COPY_ICON = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';

let mermaidCount = 0;

// Améliore le HTML d'une réponse terminée : barre de code, couleurs, schémas
async function enhanceMessage(container) {
    const diagrams = [];

    container.querySelectorAll("pre > code").forEach((code) => {
        const pre = code.parentElement;
        const langClass = [...code.classList].find((c) => c.startsWith("language-"));
        const lang = langClass ? langClass.slice(9) : (code.classList[0] || "");

        // Bloc ```mermaid -> schéma
        if (lang === "mermaid") {
            const box = document.createElement("div");
            box.className = "mermaid-box";
            const node = document.createElement("div");
            node.textContent = code.textContent;
            box.appendChild(node);
            pre.replaceWith(box);
            diagrams.push(node);
            return;
        }

        // Coloration syntaxique
        if (window.hljs) {
            try { hljs.highlightElement(code); } catch (e) {}
        }

        // Barre avec le langage et un bouton "Copier"
        const block = document.createElement("div");
        block.className = "code-block";
        const head = document.createElement("div");
        head.className = "code-head";
        const label = document.createElement("span");
        label.textContent = lang || "code";
        const button = document.createElement("button");
        button.type = "button";
        button.innerHTML = COPY_ICON + "<span>Copier</span>";
        button.addEventListener("click", () => copyText(code.textContent));
        head.append(label, button);
        pre.replaceWith(block);
        block.append(head, pre);
    });

    if (diagrams.length && window.mermaid) {
        mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: isDark() ? "dark" : "default" });
        for (const node of diagrams) {
            const source = node.textContent;
            const id = "mermaid-" + (++mermaidCount);
            try {
                const { svg } = await mermaid.render(id, source);
                node.innerHTML = svg;
            } catch (e) {
                // Schéma invalide : on montre le code à la place
                document.getElementById("d" + id)?.remove();
                document.getElementById(id)?.remove();
                const pre = document.createElement("pre");
                pre.textContent = source;
                const note = document.createElement("p");
                note.className = "text-left text-xs text-red-500";
                note.textContent = "⚠️ Schéma Mermaid invalide, voici son code :";
                node.replaceWith(note, pre);
            }
        }
    }
}

// =====================================================================
// Affichage des messages
// =====================================================================
const output = () => $("#gpt-output");

// Crée un nouveau bloc de message à partir du modèle HTML (user ou assistant).
function createMessageElement(role) {
    const template = $(role === "user" ? "#user-template" : "#assistant-template");
    const clone = template.content.firstElementChild.cloneNode(true);
    clone.dataset.role = role;
    output().appendChild(clone);
    return clone;
}

function fileChip(file, removable, index) {
    const chip = document.createElement("span");
    chip.className = "inline-flex max-w-full items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200";
    const icon = file.name.toLowerCase().endsWith(".pdf") ? "📕" : "📄";
    const name = document.createElement("span");
    name.className = "truncate";
    name.textContent = `${icon} ${file.name}`;
    const size = document.createElement("span");
    size.className = "text-slate-400";
    size.textContent = `≈ ${fmt(estimateTokens(file.text))} tk${file.truncated ? " (coupé)" : ""}`;
    chip.append(name, size);
    if (removable) {
        const x = document.createElement("button");
        x.type = "button";
        x.className = "ml-0.5 text-slate-400 hover:text-red-500";
        x.textContent = "✕";
        x.title = "Retirer";
        x.addEventListener("click", () => {
            state.files.splice(index, 1);
            renderFileChips();
            updateInputTokens();
        });
        chip.appendChild(x);
    }
    return chip;
}

// Ajoute un message de l'utilisateur à la conversation et retourne son élément HTML.
function appendMessage(message) {
    const el = createMessageElement("user");
    // innerText : le message est affiché comme du texte, jamais interprété comme du HTML
    $(".message", el).innerText = message.content;
    if (!message.content) $(".message", el).classList.add("hidden");
    const files = $(".files", el);
    (message.files || []).forEach((f) => files.appendChild(fileChip(f, false)));
    $(".copy-msg", el).addEventListener("click", () => copyText(message.content));
    return el;
}

function metaText(meta) {
    if (!meta) return "";
    const parts = [];
    const approx = meta.estimated ? "≈ " : "";
    if (meta.prompt_tokens != null) parts.push(`↑ ${approx}${fmt(meta.prompt_tokens)}`);
    if (meta.completion_tokens != null) parts.push(`↓ ${approx}${fmt(meta.completion_tokens)} tokens`);
    if (meta.duration) parts.push(`${String(meta.duration).replace(".", ",")} s`);
    if (meta.tokens_per_s) parts.push(`${fmt(Math.round(meta.tokens_per_s))} tok/s`);
    if (meta.model) parts.push(shortModel(meta.model));
    if (meta.stopped) parts.push("⏹ arrêtée");
    return parts.join(" · ");
}

// Crée le bloc d'une réponse de l'IA (vide au départ, avec les points animés)
function createAssistantView() {
    const root = createMessageElement("assistant");
    const view = {
        root,
        body: $(".message", root),
        typing: $(".typing", root),
        errorBox: $(".error-box", root),
        meta: $(".meta", root),
        sources: $(".sources", root),
        text: "",
    };
    if (state.settings.rag) $(".typing-label", root).textContent = "Recherche dans tes documents…";
    $(".copy-msg", root).addEventListener("click", () => copyText(view.text));
    $(".regen", root).addEventListener("click", regenerate);
    return view;
}

async function showAssistantMessage(message, view = createAssistantView()) {
    view.text = message.content;
    view.typing.classList.add("hidden");
    view.body.innerHTML = markdownToHtml(message.content);
    view.meta.textContent = metaText(message.meta);
    await enhanceMessage(view.body);
    renderSources(view, message.meta);
    return view;
}

// ---------- Sources du RAG ----------
function renderSources(view, meta) {
    const sources = (meta && meta.sources) || [];
    view.sources.replaceChildren();
    view.sources.classList.toggle("hidden", !meta || !meta.rag);
    if (!meta || !meta.rag) return;

    const title = document.createElement("div");
    title.className = "mb-1.5 text-xs font-semibold text-slate-500 dark:text-slate-400";
    title.textContent = sources.length
        ? `📚 ${sources.length} extrait(s) trouvé(s) dans tes documents`
        : "📚 Aucun extrait pertinent trouvé dans tes documents";
    view.sources.appendChild(title);

    const list = document.createElement("div");
    list.className = "flex flex-col gap-1.5";
    sources.forEach((src) => {
        const item = document.createElement("details");
        item.className = "source rounded-lg border border-slate-200 bg-slate-50 text-sm transition dark:border-slate-700 dark:bg-slate-900";
        item.dataset.n = src.n;
        const summary = document.createElement("summary");
        summary.className = "flex cursor-pointer items-center gap-2 px-3 py-1.5";
        const badge = document.createElement("span");
        badge.className = "grid size-5 shrink-0 place-items-center rounded bg-indigo-600 text-[11px] font-bold text-white";
        badge.textContent = src.n;
        const name = document.createElement("span");
        name.className = "min-w-0 flex-1 truncate";
        name.textContent = `${src.doc} · page ${src.page}`;
        const score = document.createElement("span");
        score.className = "text-xs text-slate-400";
        score.textContent = `score ${String(src.score).replace(".", ",")}`;
        summary.append(badge, name, score);
        const text = document.createElement("p");
        text.className = "border-t border-slate-200 px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap text-slate-600 dark:border-slate-700 dark:text-slate-300";
        text.textContent = src.text;
        item.append(summary, text);
        list.appendChild(item);
    });
    view.sources.appendChild(list);
    linkCitations(view, sources);
}

// Transforme les [1], [2] du texte en petits boutons qui ouvrent la source
function linkCitations(view, sources) {
    if (!sources.length) return;
    const valid = new Set(sources.map((s) => String(s.n)));
    const walker = document.createTreeWalker(view.body, NodeFilter.SHOW_TEXT, {
        acceptNode: (node) => (node.parentElement.closest("pre, code, .mermaid-box") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    const nodes = [];
    while (walker.nextNode()) if (/\[\d+\]/.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);

    nodes.forEach((node) => {
        const parts = node.nodeValue.split(/(\[\d+\])/);
        const fragment = document.createDocumentFragment();
        parts.forEach((part) => {
            const m = part.match(/^\[(\d+)\]$/);
            if (m && valid.has(m[1])) {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "cite";
                btn.textContent = m[1];
                btn.title = "Voir la source " + m[1];
                btn.addEventListener("click", () => {
                    const item = view.sources.querySelector(`.source[data-n="${m[1]}"]`);
                    if (!item) return;
                    item.open = true;
                    item.scrollIntoView({ behavior: "smooth", block: "nearest" });
                    item.classList.add("ring-2", "ring-indigo-400");
                    setTimeout(() => item.classList.remove("ring-2", "ring-indigo-400"), 1200);
                });
                fragment.appendChild(btn);
            } else if (part) {
                fragment.appendChild(document.createTextNode(part));
            }
        });
        node.replaceWith(fragment);
    });
}

function showError(view, text) {
    view.typing.classList.add("hidden");
    view.errorBox.textContent = "⚠️ " + text;
    view.errorBox.classList.remove("hidden");
    view.root.dataset.error = "1";
    $(".copy-msg", view.root).classList.add("hidden");
}

async function renderConversation() {
    output().innerHTML = "";
    $("#welcome").classList.toggle("hidden", state.messages.length > 0);
    for (const m of state.messages) {
        if (m.role === "user") appendMessage(m);
        else await showAssistantMessage(m);
    }
    updateRegenButtons();
    updateTokenBar();
    scrollToBottom(true);
}

// Le bouton "Régénérer" n'apparaît que sur la dernière réponse
function updateRegenButtons() {
    document.querySelectorAll("#gpt-output .regen").forEach((b) => b.classList.add("hidden"));
    const last = output().lastElementChild;
    if (last && last.dataset.role === "assistant" && !state.streaming) {
        $(".regen", last).classList.remove("hidden");
    }
}

function isNearBottom() {
    const s = $("#scroller");
    return s.scrollHeight - s.scrollTop - s.clientHeight < 150;
}

function scrollToBottom(force = false) {
    const s = $("#scroller");
    if (force || isNearBottom()) s.scrollTop = s.scrollHeight;
}

// =====================================================================
// Compteur de tokens
// =====================================================================
function updateTokenBar() {
    const answers = state.messages.filter((m) => m.role === "assistant" && m.meta);
    $("#token-bar").classList.toggle("hidden", answers.length === 0);
    if (!answers.length) return;

    // Total consommé = somme des tokens de chaque échange
    const total = answers.reduce((sum, m) => sum + (m.meta.total_tokens || 0), 0);
    const approx = answers.some((m) => m.meta.estimated) ? "≈ " : "";
    $("#total-tokens").textContent = approx + fmt(total);

    // Contexte occupé = taille du dernier échange (question + historique + réponse)
    const last = answers[answers.length - 1].meta;
    const used = (last.prompt_tokens || 0) + (last.completion_tokens || 0);
    const max = contextWindow(state.settings.model);
    const percent = Math.min(100, (used / max) * 100);
    const fill = $("#context-fill");
    fill.style.width = Math.max(percent, 1) + "%";
    fill.classList.remove("bg-emerald-500", "bg-amber-500", "bg-red-500");
    fill.classList.add(percent < 50 ? "bg-emerald-500" : percent < 80 ? "bg-amber-500" : "bg-red-500");
    $("#context-text").textContent = `${percent < 1 ? percent.toFixed(1).replace(".", ",") : Math.round(percent)} % (${fmt(used)} / ${fmt(max)})`;
}

// Estimation en direct pendant la saisie
function updateInputTokens() {
    const text = $("#prompt").value;
    const fileTokens = state.files.reduce((s, f) => s + estimateTokens(f.text), 0);
    const tokens = estimateTokens(text) + fileTokens;
    const el = $("#input-tokens");
    if (!tokens) {
        el.textContent = "";
    } else {
        const max = contextWindow(state.settings.model);
        el.textContent = `≈ ${fmt(tokens)} tokens` + (fileTokens ? " (fichiers inclus)" : "") + (tokens > max ? " ⚠️ trop long pour ce modèle" : "");
    }
    $("#send-btn").disabled = !text.trim() && !state.files.length;
}

// =====================================================================
// Envoi et réception (streaming)
// =====================================================================
// Récupère les messages de la conversation (avec leur rôle) pour les envoyer à Flask.
function getChatHistory() {
    return state.messages.map((m) => ({
        role: m.role,
        content: m.content,
        files: (m.files || []).map((f) => ({ name: f.name, text: f.text })),
    }));
}

// Envoie la conversation à Flask et retourne un lecteur de la réponse progressive.
async function fetchPromptResponse(signal) {
    const response = await fetch("/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            messages: getChatHistory(),
            model: state.settings.model,
            system: state.settings.system,
            temperature: state.settings.temperature,
            rag: state.settings.rag,
        }),
        signal,
    });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `Erreur ${response.status}`);
    }
    // Retourne un lecteur permettant de lire la réponse fragment par fragment.
    return response.body.getReader();
}

// Lit la réponse progressive et affiche le texte converti en HTML.
async function readResponseChunks(reader, view) {
    // Décode les octets reçus en texte UTF-8.
    const decoder = new TextDecoder("utf-8");
    let raw = "";
    let frame = 0;

    // Redessine au plus une fois par image pour rester fluide
    const draw = () => {
        frame = 0;
        if (!view.text) return;
        view.typing.classList.add("hidden");
        view.body.innerHTML = markdownToHtml(view.text);
        scrollToBottom();
    };

    try {
        while (true) {
            // done indique la fin du flux ; value contient les octets reçus.
            const { done, value } = await reader.read();
            if (done) break;
            raw += decoder.decode(value, { stream: true });
            view.text = raw.split(SEPARATOR)[0];
            if (!frame) frame = requestAnimationFrame(draw);
        }
        raw += decoder.decode();
    } finally {
        // Annule un dessin en attente (sinon il effacerait la mise en forme finale)
        if (frame) cancelAnimationFrame(frame);
        frame = 0;
        // Libère le lecteur, même si une erreur survient.
        try { reader.releaseLock(); } catch (e) {}
    }

    view.text = raw.split(SEPARATOR)[0];
    draw();

    // Infos envoyées après le séparateur : tokens ou message d'erreur
    let meta = null;
    let error = null;
    const index = raw.indexOf(SEPARATOR);
    if (index >= 0) {
        try {
            const info = JSON.parse(raw.slice(index + 1));
            if (info.error) error = info.error;
            else meta = info;
        } catch (e) {}
    }
    return { text: view.text, meta, error };
}

function setStreaming(on) {
    state.streaming = on;
    $("#send-btn").classList.toggle("hidden", on);
    $("#stop-btn").classList.toggle("hidden", !on);
    $("#stop-btn").classList.toggle("grid", on);
    if (!on) updateInputTokens();
}

// Demande une réponse au modèle pour la conversation actuelle
async function streamAnswer() {
    const view = createAssistantView();
    view.root.classList.add("streaming");
    scrollToBottom(true);
    setStreaming(true);
    updateRegenButtons();

    const controller = new AbortController();
    state.controller = controller;
    let result = { text: "", meta: null, error: null };
    let stopped = false;

    try {
        const reader = await fetchPromptResponse(controller.signal);
        result = await readResponseChunks(reader, view);
    } catch (error) {
        if (error.name === "AbortError") {
            stopped = true;
            result.text = view.text;
        } else {
            result.error = error.message || "Impossible de récupérer la réponse.";
        }
    } finally {
        state.controller = null;
        setStreaming(false);
        view.root.classList.remove("streaming");
    }

    if (result.text) {
        const meta = result.meta || {
            estimated: true,
            completion_tokens: estimateTokens(result.text),
            total_tokens: estimateTokens(result.text),
            model: state.settings.model,
        };
        if (stopped) meta.stopped = true;
        const message = { role: "assistant", content: result.text, meta };
        state.messages.push(message);
        await showAssistantMessage(message, view);
        await saveMessages();
    } else if (stopped) {
        view.root.remove();
    }

    if (result.error) {
        // Le message d'erreur est affiché mais pas enregistré dans l'historique
        if (result.text) showError(createAssistantView(), result.error);
        else showError(view, result.error);
    }

    updateRegenButtons();
    updateTokenBar();
    scrollToBottom();
}

async function sendMessage(text) {
    if (state.streaming) return;
    const content = text.trim();
    const files = state.files.slice();
    if (!content && !files.length) return;

    // Retire les anciens messages d'erreur (ils ne font pas partie de l'historique)
    document.querySelectorAll('#gpt-output [data-error="1"]').forEach((e) => e.remove());

    try {
        // Première question : on crée la conversation dans SQLite
        if (!state.chat) {
            state.chat = await api("/api/chats", { method: "POST", body: state.settings });
            setCurrentChatUi();
        }
    } catch (e) {
        toast("Erreur : " + e.message);
        return;
    }

    const message = { role: "user", content, files };
    state.messages.push(message);
    $("#prompt").value = "";
    autoResize();
    state.files = [];
    renderFileChips();
    updateInputTokens();

    $("#welcome").classList.add("hidden");
    appendMessage(message);
    scrollToBottom(true);
    await saveMessages();

    const isFirst = state.messages.filter((m) => m.role === "user").length === 1;
    await streamAnswer();
    if (isFirst) generateTitle();
    refreshChatList();
}

// Refait la dernière réponse
async function regenerate() {
    if (state.streaming) return;
    document.querySelectorAll('#gpt-output [data-error="1"]').forEach((e) => e.remove());
    const last = output().lastElementChild;
    if (state.messages.length && state.messages[state.messages.length - 1].role === "assistant") {
        state.messages.pop();
        if (last && last.dataset.role === "assistant") last.remove();
    }
    if (!state.messages.length) return;
    await streamAnswer();
}

function stopStreaming() {
    if (state.controller) state.controller.abort();
}

async function saveMessages() {
    if (!state.chat) return;
    try {
        await api(`/api/chats/${state.chat.id}/messages`, { method: "PUT", body: { messages: state.messages } });
    } catch (e) {
        toast("Sauvegarde impossible : " + e.message);
    }
}

async function generateTitle() {
    if (!state.chat) return;
    const id = state.chat.id;
    try {
        const { title } = await api(`/api/chats/${id}/title`, { method: "POST" });
        if (state.chat && state.chat.id === id) {
            state.chat.title = title;
            $("#chat-title").textContent = title;
            document.title = `${title} — FlaskGPT`;
        }
        refreshChatList();
    } catch (e) {}
}

// =====================================================================
// Conversations (barre latérale)
// =====================================================================
let searchQuery = "";

async function refreshChatList() {
    try {
        const url = searchQuery ? `/api/chats?q=${encodeURIComponent(searchQuery)}` : "/api/chats";
        const { chats } = await api(url);
        state.chats = chats;
        renderChatList();
    } catch (e) {
        toast("Impossible de charger les conversations");
    }
}

const ICONS = {
    pin: '<svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 17v5M9 3h6l-1 6 4 4H6l4-4-1-6Z"/></svg>',
    pinned: '<svg width="15" height="15" fill="currentColor" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 17v5M9 3h6l-1 6 4 4H6l4-4-1-6Z"/></svg>',
    edit: '<svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16v4Z"/></svg>',
    trash: '<svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
};

function chatItem(chat) {
    const li = document.createElement("li");
    const active = state.chat && state.chat.id === chat.id;
    li.className = "group relative flex items-center rounded-lg text-sm " +
        (active ? "bg-indigo-100 text-indigo-900 dark:bg-indigo-950 dark:text-indigo-100" : "hover:bg-slate-200 dark:hover:bg-slate-800");

    const open = document.createElement("button");
    open.className = "min-w-0 flex-1 px-3 py-2 text-left";
    const title = document.createElement("div");
    title.className = "truncate";
    title.textContent = chat.title;
    open.appendChild(title);
    if (chat.snippet) {
        const snippet = document.createElement("div");
        snippet.className = "truncate text-xs text-slate-500 dark:text-slate-400";
        snippet.textContent = chat.snippet;
        open.appendChild(snippet);
    }
    open.addEventListener("click", () => openChat(chat.id));
    open.addEventListener("dblclick", () => startRename(li, chat));

    const tools = document.createElement("div");
    tools.className = "flex shrink-0 items-center gap-0.5 pr-1.5 text-slate-500 " +
        (chat.pinned ? "" : "opacity-0 group-hover:opacity-100 focus-within:opacity-100");
    const makeTool = (html, label, handler, extra = "") => {
        const b = document.createElement("button");
        b.className = "rounded p-1 hover:bg-white hover:text-slate-800 dark:hover:bg-slate-700 dark:hover:text-white " + extra;
        b.innerHTML = html;
        b.title = label;
        b.addEventListener("click", (e) => { e.stopPropagation(); handler(); });
        return b;
    };
    const onlyOnHover = chat.pinned ? "hidden group-hover:block" : "";
    tools.append(
        makeTool(chat.pinned ? ICONS.pinned : ICONS.pin, chat.pinned ? "Désépingler" : "Épingler",
            () => togglePin(chat), chat.pinned ? "text-indigo-600 dark:text-indigo-400" : ""),
        makeTool(ICONS.edit, "Renommer", () => startRename(li, chat), onlyOnHover),
        makeTool(ICONS.trash, "Supprimer", () => deleteChat(chat), "hover:text-red-600! " + onlyOnHover),
    );
    li.append(open, tools);
    return li;
}

function renderChatList() {
    const pinned = state.chats.filter((c) => c.pinned);
    const others = state.chats.filter((c) => !c.pinned);
    const searching = Boolean(searchQuery);

    $("#pinned-section").classList.toggle("hidden", searching || pinned.length === 0);
    $("#pinned-list").replaceChildren(...(searching ? [] : pinned.map(chatItem)));
    const list = searching ? state.chats : others;
    $("#chat-list").replaceChildren(...list.map(chatItem));
    $("#recent-title").textContent = searching ? `Résultats (${state.chats.length})` : "Récentes";
    $("#recent-title").classList.toggle("hidden", !searching && others.length === 0 && pinned.length > 0);

    const empty = $("#chat-list-empty");
    empty.textContent = searching ? "Aucun résultat." : "Aucune conversation pour l'instant.";
    empty.classList.toggle("hidden", list.length > 0 || (!searching && pinned.length > 0));
}

async function togglePin(chat) {
    try {
        await api(`/api/chats/${chat.id}`, { method: "PATCH", body: { pinned: !chat.pinned } });
        toast(chat.pinned ? "Conversation désépinglée" : "Conversation épinglée 📌");
        refreshChatList();
    } catch (e) {
        toast("Erreur : " + e.message);
    }
}

function startRename(li, chat) {
    const input = document.createElement("input");
    input.value = chat.title;
    input.className = "m-1 w-full rounded-md border border-indigo-400 bg-white px-2 py-1 text-sm outline-none dark:bg-slate-800";
    li.replaceChildren(input);
    input.focus();
    input.select();

    let done = false;
    const finish = async (save) => {
        if (done) return;
        done = true;
        const title = input.value.trim();
        if (save && title && title !== chat.title) {
            try {
                await api(`/api/chats/${chat.id}`, { method: "PATCH", body: { title } });
                if (state.chat && state.chat.id === chat.id) {
                    state.chat.title = title;
                    $("#chat-title").textContent = title;
                }
            } catch (e) {
                toast("Erreur : " + e.message);
            }
        }
        refreshChatList();
    };
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") finish(true);
        if (e.key === "Escape") finish(false);
    });
    input.addEventListener("blur", () => finish(true));
}

async function deleteChat(chat) {
    if (!confirm(`Supprimer la conversation « ${chat.title} » ?`)) return;
    try {
        await api(`/api/chats/${chat.id}`, { method: "DELETE" });
        if (state.chat && state.chat.id === chat.id) newChat();
        toast("Conversation supprimée");
        refreshChatList();
    } catch (e) {
        toast("Erreur : " + e.message);
    }
}

function setCurrentChatUi() {
    const chat = state.chat;
    $("#chat-title").textContent = chat ? chat.title : "Nouvelle conversation";
    document.title = chat ? `${chat.title} — FlaskGPT` : "FlaskGPT — Assistant IA";
    const exportBtn = $("#export-btn");
    exportBtn.href = chat ? `/api/chats/${chat.id}/export` : "#";
    exportBtn.classList.toggle("pointer-events-none", !chat);
    exportBtn.classList.toggle("opacity-40", !chat);
    history.replaceState(null, "", chat ? `#${chat.id}` : location.pathname);
    renderChatList();
}

async function openChat(id) {
    if (state.streaming) stopStreaming();
    try {
        const chat = await api(`/api/chats/${id}`);
        state.chat = chat;
        state.messages = chat.messages || [];
        state.settings = {
            model: chat.model || state.settings.model,
            system: chat.system || "",
            temperature: chat.temperature ?? 0.7,
            rag: Boolean(chat.rag),
        };
        updateModelBadge();
        setCurrentChatUi();
        await renderConversation();
        closeSidebar();
    } catch (e) {
        toast("Conversation introuvable");
        newChat();
    }
}

function newChat() {
    if (state.streaming) stopStreaming();
    state.chat = null;
    state.messages = [];
    state.settings = loadDefaults();
    updateModelBadge();
    setCurrentChatUi();
    renderConversation();
    closeSidebar();
    $("#prompt").focus();
}

// =====================================================================
// Fichiers joints (texte, code, CSV, PDF)
// =====================================================================
// Lit le texte de chaque page d'un PDF avec pdf.js (directement dans le navigateur)
async function readPdfPages(file, onProgress = () => {}, maxPages = 1000) {
    await loadScript("https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js");
    pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
    const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
    const total = Math.min(pdf.numPages, maxPages);
    const pages = [];
    for (let i = 1; i <= total; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        pages.push(content.items.map((item) => item.str + (item.hasEOL ? "\n" : "")).join(""));
        onProgress(i, total);
    }
    return pages;
}

async function readPdf(file) {
    const pages = await readPdfPages(file, () => {}, 200);
    return pages.map((text, i) => `--- Page ${i + 1} ---\n${text}`).join("\n\n");
}

function readNotebook(text) {
    // Fichier Jupyter : on garde seulement le contenu des cellules
    const nb = JSON.parse(text);
    return (nb.cells || []).map((cell) => {
        const src = Array.isArray(cell.source) ? cell.source.join("") : cell.source || "";
        return cell.cell_type === "code" ? "```python\n" + src + "\n```" : src;
    }).join("\n\n");
}

async function addFiles(fileList) {
    for (const file of fileList) {
        if (state.files.length >= MAX_FILES) {
            toast(`${MAX_FILES} fichiers maximum`);
            break;
        }
        if (file.size > 20 * 1024 * 1024) {
            toast(`${file.name} : trop gros (20 Mo max)`);
            continue;
        }
        let text;
        try {
            const lower = file.name.toLowerCase();
            if (lower.endsWith(".pdf")) {
                toast(`Lecture de ${file.name}…`);
                text = await readPdf(file);
            } else if (lower.endsWith(".ipynb")) {
                text = readNotebook(await file.text());
            } else {
                text = await file.text();
                if (text.includes("\u0000")) throw new Error("binaire");
            }
        } catch (e) {
            toast(`Impossible de lire ${file.name} (format non supporté)`);
            continue;
        }
        if (!text.trim()) {
            toast(`${file.name} ne contient pas de texte (PDF scanné ?)`);
            continue;
        }
        const truncated = text.length > MAX_FILE_CHARS;
        state.files.push({ name: file.name, text: truncated ? text.slice(0, MAX_FILE_CHARS) : text, truncated });
    }
    renderFileChips();
    updateInputTokens();
    $("#prompt").focus();
}

function renderFileChips() {
    const box = $("#file-chips");
    box.replaceChildren(...state.files.map((f, i) => fileChip(f, true, i)));
    box.classList.toggle("hidden", state.files.length === 0);
    box.classList.toggle("flex", state.files.length > 0);
}

// =====================================================================
// Paramètres : modèle, personnalité, prompt système, température
// =====================================================================
function updateModelBadge() {
    $("#model-badge-text").textContent = shortModel(state.settings.model);
    $("#rag-toggle").setAttribute("aria-pressed", state.settings.rag ? "true" : "false");
}

function openSettings() {
    const select = $("#model-select");
    const ids = state.models.map((m) => m.id);
    if (!ids.includes(state.settings.model)) ids.unshift(state.settings.model);
    select.replaceChildren(...ids.map((id) =>
        new Option(shortModel(id) + (id.includes("/") ? `  (${id.split("/")[0]})` : ""), id)));
    select.value = state.settings.model;
    showModelContext();

    const persona = $("#persona-select");
    persona.replaceChildren(...PERSONAS.map((p, i) => new Option(p.name, i)), new Option("✏️ Personnalisé", "custom"));
    $("#system-input").value = state.settings.system || "";
    syncPersona();

    $("#temp-input").value = state.settings.temperature;
    $("#temp-value").textContent = Number(state.settings.temperature).toFixed(1);

    $("#settings").returnValue = "";
    $("#settings").showModal();
}

function showModelContext() {
    $("#model-context").textContent = `Fenêtre de contexte : ${fmt(contextWindow($("#model-select").value))} tokens`;
}

function syncPersona() {
    const text = $("#system-input").value.trim();
    const index = PERSONAS.findIndex((p) => p.prompt === text);
    $("#persona-select").value = index >= 0 ? String(index) : "custom";
}

async function saveSettings() {
    state.settings = {
        model: $("#model-select").value,
        system: $("#system-input").value.trim(),
        temperature: Number($("#temp-input").value),
        rag: state.settings.rag,
    };
    saveDefaults(state.settings);
    updateModelBadge();
    updateTokenBar();
    updateInputTokens();
    if (state.chat) {
        try {
            await api(`/api/chats/${state.chat.id}`, { method: "PATCH", body: state.settings });
        } catch (e) {
            toast("Erreur : " + e.message);
            return;
        }
    }
    toast("Paramètres enregistrés ✓");
}

// =====================================================================
// RAG : base de connaissances (mes documents)
// =====================================================================
async function loadDocuments() {
    try {
        const data = await api("/api/documents");
        state.documents = data.documents;
        state.ragEngine = data.engine;
    } catch (e) {
        state.documents = [];
    }
    renderDocuments();
}

function renderDocuments() {
    const docs = state.documents;
    const active = docs.filter((d) => d.active).length;
    $("#docs-count").textContent = docs.length;
    $("#docs-empty").classList.toggle("hidden", docs.length > 0);
    $("#docs-list").classList.toggle("hidden", docs.length === 0);
    const chunks = docs.reduce((s, d) => s + (d.chunks || 0), 0);
    $("#docs-stats").textContent = docs.length ? `${active}/${docs.length} actifs · ${fmt(chunks)} extraits · ${state.ragEngine || ""}` : "";

    $("#docs-list").replaceChildren(...docs.map((doc) => {
        const li = document.createElement("li");
        li.className = "flex items-center gap-3 px-3 py-2.5";

        const check = document.createElement("input");
        check.type = "checkbox";
        check.checked = doc.active;
        check.className = "size-4 accent-indigo-600";
        check.title = "Utiliser ce document dans les réponses";
        check.addEventListener("change", async () => {
            await api(`/api/documents/${doc.id}`, { method: "PATCH", body: { active: check.checked } });
            doc.active = check.checked;
            renderDocuments();
        });

        const info = document.createElement("div");
        info.className = "min-w-0 flex-1";
        const name = document.createElement("div");
        name.className = "truncate text-sm font-medium" + (doc.active ? "" : " text-slate-400 line-through");
        name.textContent = (doc.name.toLowerCase().endsWith(".pdf") ? "📕 " : "📄 ") + doc.name;
        const details = document.createElement("div");
        details.className = "text-xs text-slate-500";
        const date = new Date(doc.created_at).toLocaleDateString("fr-FR");
        details.textContent = `${fmt(doc.pages)} page(s) · ${fmt(doc.chunks)} extraits · ≈ ${fmt(Math.round(doc.chars / 4))} tokens · ajouté le ${date}`;
        info.append(name, details);

        const del = document.createElement("button");
        del.className = "rounded p-1.5 text-slate-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950";
        del.innerHTML = ICONS.trash;
        del.title = "Supprimer";
        del.addEventListener("click", async () => {
            if (!confirm(`Supprimer « ${doc.name} » de ta base de connaissances ?`)) return;
            await api(`/api/documents/${doc.id}`, { method: "DELETE" });
            toast("Document supprimé");
            loadDocuments();
        });

        li.append(check, info, del);
        return li;
    }));
}

function setDocsProgress(percent, text) {
    $("#docs-progress").classList.toggle("hidden", percent === null);
    if (percent === null) return;
    $("#docs-progress-fill").style.width = percent + "%";
    $("#docs-progress-text").textContent = text;
}

async function uploadDocuments(fileList) {
    for (const file of fileList) {
        const lower = file.name.toLowerCase();
        try {
            let pages;
            setDocsProgress(0, `Lecture de ${file.name}…`);
            if (lower.endsWith(".pdf")) {
                pages = await readPdfPages(file, (i, total) =>
                    setDocsProgress(Math.round((i / total) * 80), `${file.name} : page ${i}/${total}`));
            } else if (lower.endsWith(".ipynb")) {
                pages = [readNotebook(await file.text())];
            } else {
                const text = await file.text();
                if (text.includes("\u0000")) throw new Error("Format non supporté");
                pages = [text];
            }
            setDocsProgress(90, `Découpage et indexation de ${file.name}…`);
            const doc = await api("/api/documents", { method: "POST", body: { name: file.name, pages } });
            setDocsProgress(100, `✓ ${file.name} : ${doc.chunks} extraits indexés`);
            toast(`${file.name} ajouté (${doc.chunks} extraits)`);
        } catch (e) {
            setDocsProgress(null);
            toast(`${file.name} : ${e.message}`);
        }
        await loadDocuments();
    }
    setTimeout(() => setDocsProgress(null), 2500);

    // Active automatiquement le mode RAG après le premier ajout
    if (state.documents.length && !state.settings.rag) setRag(true);
}

async function testDocsSearch(query) {
    const info = $("#docs-search-info");
    const list = $("#docs-results");
    if (!query.trim()) return;
    info.textContent = "Recherche…";
    list.replaceChildren();
    try {
        const data = await api(`/api/documents/search?q=${encodeURIComponent(query)}`);
        info.textContent = `Mots-clés (dont ceux ajoutés par l'IA) : ${data.terms.join(", ") || "—"} · ${data.results.length} résultat(s) en ${String(data.time).replace(".", ",")} s`;
        const terms = data.terms;
        list.replaceChildren(...data.results.map((r, i) => {
            const li = document.createElement("li");
            li.className = "rounded-xl border border-slate-200 p-3 text-xs dark:border-slate-700";
            const head = document.createElement("div");
            head.className = "mb-1 flex justify-between font-medium text-slate-700 dark:text-slate-200";
            head.innerHTML = `<span></span><span class="font-normal text-slate-400"></span>`;
            head.children[0].textContent = `[${i + 1}] ${r.doc} · page ${r.page}`;
            head.children[1].textContent = `score BM25 ${String(r.score).replace(".", ",")}`;
            const body = document.createElement("p");
            body.className = "leading-relaxed whitespace-pre-wrap text-slate-600 dark:text-slate-300";
            body.innerHTML = highlightTerms(r.text, terms);
            li.append(head, body);
            return li;
        }));
    } catch (e) {
        info.textContent = "Erreur : " + e.message;
    }
}

// Surligne les mots-clés trouvés (le texte est échappé avant)
function highlightTerms(text, terms) {
    const escaped = text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
    const strip = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
    return escaped.replace(/[\p{L}\p{N}]+/gu, (word) =>
        terms.includes(strip(word)) ? `<mark class="rounded bg-amber-200 px-0.5 dark:bg-amber-700 dark:text-white">${word}</mark>` : word);
}

async function setRag(on) {
    if (on && !state.documents.length) {
        toast("Ajoute d'abord un document 📚");
        $("#docs-dialog").showModal();
        return;
    }
    state.settings.rag = on;
    saveDefaults(state.settings);
    updateModelBadge();
    if (state.chat) {
        try { await api(`/api/chats/${state.chat.id}`, { method: "PATCH", body: { rag: on } }); } catch (e) {}
    }
    toast(on ? "📚 Réponses basées sur tes documents" : "Mode documents désactivé");
}

// =====================================================================
// Thème, barre latérale (mobile), zone de saisie
// =====================================================================
function toggleTheme() {
    const dark = !isDark();
    document.documentElement.classList.toggle("dark", dark);
    try { localStorage.setItem("flaskgpt:theme", dark ? "dark" : "light"); } catch (e) {}
    // Redessine les schémas avec les bonnes couleurs
    if (!state.streaming && document.querySelector(".mermaid-box")) renderConversation();
}

function openSidebar() {
    $("#sidebar").classList.remove("-translate-x-full");
    $("#sidebar-overlay").classList.remove("hidden");
}

function closeSidebar() {
    $("#sidebar").classList.add("-translate-x-full");
    $("#sidebar-overlay").classList.add("hidden");
}

function autoResize() {
    const area = $("#prompt");
    area.style.height = "auto";
    area.style.height = Math.min(area.scrollHeight, 208) + "px";
}

// =====================================================================
// Démarrage
// =====================================================================
// Attend que le HTML soit chargé avant d'exécuter ce code.
document.addEventListener("DOMContentLoaded", async () => {
    const form = $("#prompt-form");
    const area = $("#prompt");

    // Exécute cette fonction lors de l'envoi du formulaire.
    form.addEventListener("submit", (event) => {
        // Empêche l'envoi classique du formulaire et le rechargement de la page.
        event.preventDefault();
        sendMessage(area.value);
    });

    // Entrée = envoyer, Maj+Entrée = nouvelle ligne
    area.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            sendMessage(area.value);
        }
    });
    area.addEventListener("input", () => { autoResize(); updateInputTokens(); });

    $("#stop-btn").addEventListener("click", stopStreaming);
    $("#new-chat").addEventListener("click", newChat);
    $("#theme-toggle").addEventListener("click", toggleTheme);
    $("#open-sidebar").addEventListener("click", openSidebar);
    $("#close-sidebar").addEventListener("click", closeSidebar);
    $("#sidebar-overlay").addEventListener("click", closeSidebar);
    $("#model-badge").addEventListener("click", openSettings);
    $("#open-settings-side").addEventListener("click", openSettings);

    // Exemples de l'écran d'accueil
    document.querySelectorAll(".example").forEach((btn) =>
        btn.addEventListener("click", () => sendMessage(btn.dataset.prompt)));

    // Recherche (avec un petit délai pour ne pas appeler le serveur à chaque touche)
    let searchTimer;
    $("#search").addEventListener("input", (e) => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => {
            searchQuery = e.target.value.trim();
            refreshChatList();
        }, 250);
    });

    // Fichiers : bouton et glisser-déposer
    $("#attach-btn").addEventListener("click", () => $("#file-input").click());
    $("#file-input").addEventListener("change", (e) => {
        addFiles([...e.target.files]);
        e.target.value = "";
    });
    const drop = $("#drop-zone");
    ["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (e) => {
        e.preventDefault();
        drop.classList.add("ring-4", "ring-indigo-200");
    }));
    ["dragleave", "drop"].forEach((type) =>
        drop.addEventListener(type, () => drop.classList.remove("ring-4", "ring-indigo-200")));
    drop.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) addFiles([...e.dataTransfer.files]);
    });

    // Paramètres
    $("#model-select").addEventListener("change", showModelContext);
    $("#persona-select").addEventListener("change", (e) => {
        if (e.target.value !== "custom") $("#system-input").value = PERSONAS[Number(e.target.value)].prompt;
        else $("#system-input").focus();
    });
    $("#system-input").addEventListener("input", syncPersona);
    $("#temp-input").addEventListener("input", (e) => {
        $("#temp-value").textContent = Number(e.target.value).toFixed(1);
    });
    $("#settings").addEventListener("close", () => {
        if ($("#settings").returnValue === "save") saveSettings();
    });

    // RAG
    $("#rag-toggle").addEventListener("click", () => setRag(!state.settings.rag));
    $("#open-docs").addEventListener("click", () => { closeSidebar(); $("#docs-dialog").showModal(); });
    $("#close-docs").addEventListener("click", () => $("#docs-dialog").close());
    $("#docs-browse").addEventListener("click", () => $("#docs-input").click());
    $("#docs-input").addEventListener("change", (e) => {
        uploadDocuments([...e.target.files]);
        e.target.value = "";
    });
    const docsDrop = $("#docs-drop");
    ["dragenter", "dragover"].forEach((type) => docsDrop.addEventListener(type, (e) => {
        e.preventDefault();
        docsDrop.classList.add("border-indigo-400", "bg-indigo-50", "dark:bg-indigo-950");
    }));
    ["dragleave", "drop"].forEach((type) =>
        docsDrop.addEventListener(type, () => docsDrop.classList.remove("border-indigo-400", "bg-indigo-50", "dark:bg-indigo-950")));
    docsDrop.addEventListener("drop", (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) uploadDocuments([...e.dataTransfer.files]);
    });
    $("#docs-search-form").addEventListener("submit", (e) => {
        e.preventDefault();
        testDocsSearch($("#docs-search").value);
    });

    // Raccourcis clavier
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            newChat();
        }
        if (e.key === "Escape" && state.streaming) stopStreaming();
    });

    // Chargement initial
    updateModelBadge();
    updateInputTokens();
    try {
        const [status, models] = await Promise.all([api("/api/status"), api("/api/models")]);
        state.hasKey = status.has_key;
        state.models = models.models || [];
        $("#no-key-warning").classList.toggle("hidden", state.hasKey);
    } catch (e) {
        toast("Le serveur ne répond pas");
    }
    await Promise.all([refreshChatList(), loadDocuments()]);

    const id = location.hash.slice(1);
    if (id) await openChat(id);
    else newChat();
});

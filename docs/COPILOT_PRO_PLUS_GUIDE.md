# 🚀 Guida Copilot Pro Plus + GitHub Pro

> **Configurazione ottimizzata per NICO Project**  
> **Ultimo aggiornamento**: 2025-12-29

---

## 📊 Cosa Hai a Disposizione

### Copilot Pro Plus ($39/mese)
| Feature | Limite |
|---------|--------|
| Premium Requests | 1500/mese |
| Claude Opus 4.5 | ✅ |
| Claude Sonnet 4/4.5 | ✅ |
| Gemini 2.5/3 Pro | ✅ |
| GPT-5, GPT-5.1, GPT-5.2 | ✅ |
| Copilot Coding Agent | ✅ |
| GitHub Spark | ✅ |
| MCP Servers | ✅ |
| Custom Agents | ✅ |

### GitHub Pro ($4/mese)
- Unlimited private repos
- 3000 Actions minutes/month
- GitHub Pages
- Protected branches
- Required reviewers

---

## 🎯 Come Usare i Custom Agents

### Agents Disponibili

| Agent | Quando Usarlo | Comando |
|-------|---------------|---------|
| **Planner** | Pianificare prima di implementare | `@Planner` |
| **Code Reviewer** | Review qualità/sicurezza | `@Code Reviewer` |
| **Full Stack Dev** | Implementazione end-to-end | `@Full Stack Dev` |
| **Test Generator** | Generare test | `@Test Generator` |

### Workflow Consigliato

```
1. @Planner "Aggiungi feature X"
   ↓ [Crea piano dettagliato]
   
2. Click "🚀 Implementa Piano"
   ↓ [Passa a Full Stack Dev]
   
3. Click "🔍 Review Codice"
   ↓ [Passa a Code Reviewer]
   
4. Click "🧪 Genera Test"
   ↓ [Passa a Test Generator]
```

### Selezione Agent

1. Apri Chat (`Cmd+Shift+I`)
2. Click sul dropdown agent (default: "Agent")
3. Seleziona l'agent desiderato
4. Scrivi il prompt

---

## 📝 Prompt Files (Comandi /)

Prompt riusabili per task comuni:

| Comando | Descrizione |
|---------|-------------|
| `/new-api-endpoint` | Template endpoint FastAPI |
| `/new-react-component` | Template componente React |
| `/surrealdb-query` | Esempi query SurrealDB |

### Come Usarli

1. In chat, digita `/`
2. Seleziona il prompt dalla lista
3. Il template viene inserito nel contesto

---

## 🔧 MCP Servers Configurati

| Server | Funzione |
|--------|----------|
| `github` | Accesso a issues, PR, repo |
| `filesystem` | Operazioni file |
| `fetch` | HTTP requests |

### Attivare MCP

1. Apri Command Palette (`Cmd+Shift+P`)
2. `MCP: List Servers`
3. Start i server necessari

### Usare Tool MCP

```
Usa #github per cercare issues nel repo
Usa #fetch per prendere contenuto da URL
```

---

## 💡 Modelli Consigliati per Task

| Task | Modello | Motivo |
|------|---------|--------|
| Pianificazione | Claude Sonnet 4 | Veloce + ragionamento |
| Implementazione | Claude Sonnet 4.5 | Codice accurato |
| Review complessa | Claude Opus 4.5 | Max reasoning |
| Debugging | GPT-5 | Buon context |
| Documentazione | Gemini 2.5 Pro | Ottimo per testo |

### Cambiare Modello

1. In Chat view, click sul nome modello
2. Seleziona dal dropdown
3. Il modello cambia per quella sessione

---

## ⚡ Shortcut Essenziali

| Shortcut | Azione |
|----------|--------|
| `Cmd+Shift+I` | Apri Chat |
| `Cmd+I` | Inline Chat |
| `Cmd+Enter` | Invia prompt |
| `Tab` | Accetta suggestion |
| `Esc` | Chiudi inline suggestion |

---

## 🎯 Best Practices

### 1. Usa Subagents per Task Complessi

```
"Usa un subagent per cercare tutti i file che usano SurrealDB"
```

### 2. Sfrutta #codebase

```
"#codebase Come funziona l'autenticazione?"
```

### 3. Usa Checkpoints

Durante una sessione lunga:
- I checkpoints salvano lo stato
- Puoi tornare indietro se qualcosa va storto
- Attivati automaticamente ogni N modifiche

### 4. Auto-Approve Safe Commands

I comandi sicuri sono auto-approvati:
- `ls`, `cat`, `echo`, `pwd`
- `git status`, `git branch`, `git log`
- `pytest`, `npm run`
- `docker ps`

Comandi **sempre** manuali:
- `rm`, `rmdir`
- `pip install`
- Qualsiasi `sudo`

### 5. Handoffs tra Agents

Dopo ogni risposta, guarda i bottoni:
- 🚀 **Implementa** → passa a dev
- 🔍 **Review** → passa a reviewer
- 📋 **Pianifica** → passa a planner

---

## 🐛 Troubleshooting

### "Agent non trova file"
```bash
# Build workspace index
Cmd+Shift+P → "Build Remote Workspace Index"
```

### "MCP server non parte"
```bash
# Verifica npx disponibile
which npx

# Restart server
Cmd+Shift+P → "MCP: List Servers" → Stop → Start
```

### "Modello non disponibile"
Alcuni modelli richiedono Copilot Pro Plus. Verifica il tuo piano su GitHub.

---

## 📁 File Configurazione

| File | Scopo |
|------|-------|
| `.vscode/settings.json` | VS Code settings |
| `.vscode/mcp.json` | MCP servers config |
| `.github/agents/*.agent.md` | Custom agents |
| `.github/prompts/*.prompt.md` | Prompt riusabili |
| `.github/copilot-instructions.md` | Istruzioni globali |

---

## 🔄 Sync Settings

Per sincronizzare su altri device:

1. `Cmd+Shift+P` → "Settings Sync: Turn On"
2. Abilita "MCP Servers" nella sync
3. I settings si sincronizzano automaticamente

---

## 📈 Monitorare Usage

### Premium Requests
- Vai su https://github.com/settings/copilot
- Sezione "Usage"
- Vedi requests rimanenti

### Consiglio
- Claude Opus usa più requests
- Per task semplici usa Sonnet
- Per batch di modifiche usa Agent mode (conta come 1 request)

---

**Buon coding! 🎉**

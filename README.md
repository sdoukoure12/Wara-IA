# 🦁 Wara-IA — Assistant Intelligent Libre

> **Wara** = lion (en dioula) · **IA** = Intelligence Artificielle pensif & actif

Wara-IA est un assistant conversationnel **gratuit**, **open-source** et **local**, alimenté par le modèle [DialoGPT](https://huggingface.co/microsoft/DialoGPT-small) de Microsoft via la bibliothèque [Hugging Face Transformers](https://github.com/huggingface/transformers).  
Aucune clé API, aucun abonnement — tout tourne sur votre machine.

---

## 📁 Structure du projet

```
Wara-IA/
├── wara_ia.py          # Le script principal — interface conversationnelle
├── requirements.txt    # Dépendances Python (transformers, torch…)
├── README.md           # Documentation et guide d'installation
└── .gitignore          # Exclusions Git (venv, __pycache__, secrets…)
```

---

## ✨ Fonctionnalités

- 🤖 **Conversation en langage naturel** via DialoGPT (modèle libre)
- 🧠 **Mémoire de contexte** — l'assistant se souvient des derniers échanges
- 🔌 **Mode hors-ligne** — réponses par mots-clés si le modèle n'est pas disponible
- 🎨 **Interface colorée** dans le terminal (codes ANSI)
- ⚡ **Fonctionne sans GPU** — inférence CPU incluse

---

## 🚀 Installation

### Prérequis

| Outil   | Version minimale |
|---------|-----------------|
| Python  | 3.8+            |
| pip     | 21+             |

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/sdoukoure12/Wara-IA.git
cd Wara-IA
```

### Étape 2 — Créer un environnement virtuel (recommandé)

```bash
# Créer l'environnement
python -m venv venv

# Activer l'environnement
# Sur Linux / macOS :
source venv/bin/activate

# Sur Windows (PowerShell) :
.\venv\Scripts\Activate.ps1

# Sur Windows (CMD) :
venv\Scripts\activate.bat
```

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

> ℹ️ La première installation télécharge les poids du modèle (~350 Mo).  
> Ils sont mis en cache dans `~/.cache/huggingface/` pour les prochains lancements.

---

## ▶️ Lancer Wara-IA

```bash
python wara_ia.py
```

---

## 💬 Utilisation

```
Vous  > Bonjour !
Wara  > Bonjour ! Comment puis-je vous aider aujourd'hui ?

Vous  > aide
Wara  > Commandes disponibles :
          • 'quitter' / 'exit'  — terminer la session
          • 'effacer' / 'reset' — réinitialiser l'historique
          • 'aide'    / 'help'  — afficher cette aide

Vous  > quitter
Wara  > À bientôt ! Que le lion vous accompagne. 🦁
```

---

## ⚙️ Configuration avancée

Vous pouvez modifier les constantes en haut de `wara_ia.py` :

| Constante          | Défaut                          | Description                               |
|--------------------|---------------------------------|-------------------------------------------|
| `MODEL_NAME`       | `microsoft/DialoGPT-small`      | Modèle Hugging Face utilisé               |
| `MAX_NEW_TOKENS`   | `200`                           | Longueur max de la réponse (en tokens)    |
| `TEMPERATURE`      | `0.75`                          | Créativité des réponses (0 = déterministe)|
| `MAX_HISTORY_TURNS`| `5`                             | Nombre de tours mémorisés                 |

**Modèles disponibles (du plus léger au plus précis) :**
```bash
# Léger (~350 Mo) — par défaut
MODEL_NAME = "microsoft/DialoGPT-small"

# Moyen (~850 Mo) — meilleures réponses
MODEL_NAME = "microsoft/DialoGPT-medium"

# Grand (~1.8 Go) — qualité maximale
MODEL_NAME = "microsoft/DialoGPT-large"
```

---

## 🛡️ Licence

Ce projet est distribué sous licence [MIT](LICENSE).

---

## 🤝 Contribuer

Les contributions sont les bienvenues !

```bash
# 1. Forker le dépôt sur GitHub
# 2. Cloner votre fork
git clone https://github.com/VOTRE_NOM/Wara-IA.git
cd Wara-IA

# 3. Créer une branche de fonctionnalité
git checkout -b feature/ma-nouvelle-fonctionnalite

# 4. Pousser et ouvrir une Pull Request
git push origin feature/ma-nouvelle-fonctionnalite
```

---

*Fait avec ❤️ et la force du 🦁 — Wara-IA, pensif & actif.*

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════╗
║                        W A R A  -  I A                          ║
║             Assistant Intelligent Libre (Wara = lion)           ║
║                    Version 1.0  —  Pensif & Actif               ║
╚══════════════════════════════════════════════════════════════════╝

Description :
    Wara-IA est un assistant conversationnel basé sur des modèles de
    langage pré-entraînés (DialoGPT via Hugging Face Transformers).
    Il fonctionne entièrement en local, sans abonnement payant.

Auteur  : sdoukoure12
Licence : MIT
"""

# ──────────────────────────────────────────────────────────────────
# Importations standard
# ──────────────────────────────────────────────────────────────────
import sys                  # Gestion du système (exit, version Python…)
import os                   # Opérations sur le système d'exploitation
import time                 # Mesure du temps de réponse
import textwrap             # Mise en forme du texte dans le terminal

# ──────────────────────────────────────────────────────────────────
# Vérification de la version Python (≥ 3.8 requise)
# ──────────────────────────────────────────────────────────────────
if sys.version_info < (3, 8):
    sys.exit("❌  Wara-IA requiert Python 3.8 ou supérieur.")

# ──────────────────────────────────────────────────────────────────
# Importations tierces — gérées via requirements.txt
# ──────────────────────────────────────────────────────────────────
try:
    import torch                                # Calcul tensoriel (CPU/GPU)
    from transformers import (
        AutoModelForCausalLM,                   # Chargement du modèle de génération
        AutoTokenizer,                          # Tokeniseur associé au modèle
        pipeline,                               # API haut niveau pour l'inférence
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────
# Constantes de configuration
# ──────────────────────────────────────────────────────────────────

# Modèle conversationnel léger et gratuit (Microsoft DialoGPT-small)
# Vous pouvez remplacer par "microsoft/DialoGPT-medium" pour de meilleures réponses
MODEL_NAME = "microsoft/DialoGPT-small"

# Nombre maximum de tours de conversation conservés en mémoire
MAX_HISTORY_TURNS = 5

# Estimation du nombre de tokens par tour de conversation (entrée + réponse)
# DialoGPT tokenise en moyenne ~20 mots/tour ; on prend une marge confortable
ESTIMATED_TOKENS_PER_TURN = 100

# Longueur maximale de la réponse générée (en tokens)
MAX_NEW_TOKENS = 200

# Température de génération : plus elle est élevée, plus les réponses sont créatives
TEMPERATURE = 0.75

# Codes couleur ANSI pour l'affichage terminal
COLOR_RESET  = "\033[0m"
COLOR_LION   = "\033[38;5;214m"   # Orange — couleur du lion (Wara)
COLOR_USER   = "\033[38;5;39m"    # Bleu ciel — texte utilisateur
COLOR_BOT    = "\033[38;5;118m"   # Vert — réponse de l'IA
COLOR_ERROR  = "\033[38;5;196m"   # Rouge — messages d'erreur
COLOR_INFO   = "\033[38;5;245m"   # Gris — informations système
COLOR_BOLD   = "\033[1m"


# ──────────────────────────────────────────────────────────────────
# Réponses de secours (mode hors-ligne sans transformers)
# Ces réponses sont utilisées quand la bibliothèque n'est pas installée
# ──────────────────────────────────────────────────────────────────
FALLBACK_RESPONSES = {
    "bonjour": "Bonjour ! Je suis Wara, votre assistant IA. Comment puis-je vous aider ?",
    "hello":   "Hello! I'm Wara, your AI assistant. How can I help you?",
    "qui es-tu": "Je suis Wara-IA, un assistant intelligent libre. Mon nom signifie 'lion' — je suis pensif et actif !",
    "aide":    "Commandes disponibles :\n  • 'quitter' ou 'exit' — terminer la session\n  • 'effacer' — réinitialiser l'historique\n  • 'aide' — afficher cette aide",
    "default": "Je n'ai pas bien compris. Pouvez-vous reformuler ? (tapez 'aide' pour de l'aide)",
}


# ──────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────────────────────────

def print_banner() -> None:
    """Affiche la bannière de bienvenue de Wara-IA."""
    banner = f"""
{COLOR_LION}{COLOR_BOLD}
  ██╗    ██╗ █████╗ ██████╗  █████╗       ██╗ █████╗
  ██║    ██║██╔══██╗██╔══██╗██╔══██╗      ██║██╔══██╗
  ██║ █╗ ██║███████║██████╔╝███████║      ██║███████║
  ██║███╗██║██╔══██║██╔══██╗██╔══██║ ██   ██║██╔══██║
  ╚███╔███╔╝██║  ██║██║  ██║██║  ██║ ╚█████╔╝██║  ██║
   ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚════╝ ╚═╝  ╚═╝
{COLOR_RESET}
{COLOR_INFO}  Assistant Intelligent Libre  •  Wara = lion  •  v1.0{COLOR_RESET}
{COLOR_INFO}  ─────────────────────────────────────────────────────{COLOR_RESET}
{COLOR_INFO}  Tapez {COLOR_BOLD}'aide'{COLOR_RESET}{COLOR_INFO} pour l'aide  •  {COLOR_BOLD}'quitter'{COLOR_RESET}{COLOR_INFO} pour sortir{COLOR_RESET}
"""
    print(banner)


def print_message(role: str, message: str) -> None:
    """
    Affiche un message formaté dans le terminal.

    Args:
        role    : 'user' ou 'wara'
        message : le texte à afficher
    """
    width = min(os.get_terminal_size().columns - 4, 80)

    if role == "user":
        prefix = f"{COLOR_USER}{COLOR_BOLD}Vous  >{COLOR_RESET} "
        wrapped = textwrap.fill(message, width=width, subsequent_indent="        ")
        print(f"\n{prefix}{COLOR_USER}{wrapped}{COLOR_RESET}")
    else:
        prefix = f"{COLOR_LION}{COLOR_BOLD}Wara  >{COLOR_RESET} "
        wrapped = textwrap.fill(message, width=width, subsequent_indent="        ")
        print(f"{prefix}{COLOR_BOT}{wrapped}{COLOR_RESET}\n")


def fallback_response(user_input: str) -> str:
    """
    Génère une réponse simple basée sur des mots-clés.
    Utilisé quand le modèle Transformers n'est pas disponible.

    Args:
        user_input : texte saisi par l'utilisateur

    Returns:
        str : réponse correspondante ou réponse par défaut
    """
    text = user_input.lower().strip()
    for keyword, response in FALLBACK_RESPONSES.items():
        if keyword in text:
            return response
    return FALLBACK_RESPONSES["default"]


# ──────────────────────────────────────────────────────────────────
# Classe principale : WaraIA
# ──────────────────────────────────────────────────────────────────

class WaraIA:
    """
    Classe principale de l'assistant Wara-IA.

    Attributs :
        model_name  : identifiant du modèle Hugging Face utilisé
        tokenizer   : tokeniseur du modèle
        model       : modèle de langage chargé
        chat_history: historique encodé des tokens de conversation
        use_ai      : booléen indiquant si le modèle IA est disponible
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """
        Initialise l'assistant en chargeant le modèle si disponible.

        Args:
            model_name : nom du modèle Hugging Face à utiliser
        """
        self.model_name   = model_name
        self.tokenizer    = None
        self.model        = None
        self.chat_history = None   # Historique des tokens (tenseur PyTorch)
        self.use_ai       = False  # Passe à True si le modèle se charge

        self._load_model()         # Tentative de chargement du modèle

    # ──────────────────────────────────────────────
    # Chargement du modèle
    # ──────────────────────────────────────────────
    def _load_model(self) -> None:
        """
        Charge le tokeniseur et le modèle DialoGPT depuis Hugging Face.
        En cas d'échec, bascule sur le mode réponses de secours.
        """
        if not TRANSFORMERS_AVAILABLE:
            print(f"{COLOR_ERROR}⚠  Bibliothèques 'transformers' et/ou 'torch' introuvables.{COLOR_RESET}")
            print(f"{COLOR_INFO}   Mode hors-ligne activé — réponses basées sur des mots-clés.{COLOR_RESET}")
            print(f"{COLOR_INFO}   Installez les dépendances : pip install -r requirements.txt{COLOR_RESET}\n")
            return

        print(f"{COLOR_INFO}⏳  Chargement du modèle {self.model_name} …{COLOR_RESET}")
        try:
            # Téléchargement / lecture en cache du tokeniseur
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Chargement du modèle en mémoire (CPU par défaut)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()  # Mode inférence (désactive le dropout)

            self.use_ai = True
            print(f"{COLOR_BOT}✔  Modèle chargé avec succès !{COLOR_RESET}\n")

        except Exception as error:
            # En cas d'erreur réseau ou mémoire insuffisante
            print(f"{COLOR_ERROR}✘  Impossible de charger le modèle : {error}{COLOR_RESET}")
            print(f"{COLOR_INFO}   Mode hors-ligne activé.\n{COLOR_RESET}")

    # ──────────────────────────────────────────────
    # Génération de réponse avec le modèle IA
    # ──────────────────────────────────────────────
    def _generate_ai_response(self, user_input: str) -> str:
        """
        Génère une réponse conversationnelle via DialoGPT.

        Args:
            user_input : message de l'utilisateur

        Returns:
            str : réponse générée par le modèle
        """
        # Encodage du message utilisateur + token de fin de séquence
        new_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors="pt"       # Format tenseur PyTorch
        )

        # Concaténation avec l'historique de conversation
        if self.chat_history is not None:
            bot_input_ids = torch.cat([self.chat_history, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Génération de la réponse (sans gradient pour économiser la mémoire)
        with torch.no_grad():
            self.chat_history = self.model.generate(
                bot_input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,             # Échantillonnage stochastique
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Décodage du dernier token généré uniquement (évite de répéter le contexte)
        response = self.tokenizer.decode(
            self.chat_history[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )

        # Limitation de l'historique pour éviter les dépassements de contexte
        max_length = MAX_HISTORY_TURNS * ESTIMATED_TOKENS_PER_TURN
        if self.chat_history.shape[-1] > max_length:
            self.chat_history = self.chat_history[:, -max_length:]

        return response.strip() if response.strip() else "…"

    # ──────────────────────────────────────────────
    # Interface publique de réponse
    # ──────────────────────────────────────────────
    def respond(self, user_input: str) -> str:
        """
        Point d'entrée principal pour obtenir une réponse.

        Args:
            user_input : message saisi par l'utilisateur

        Returns:
            str : réponse de l'assistant
        """
        if self.use_ai:
            return self._generate_ai_response(user_input)
        else:
            return fallback_response(user_input)

    # ──────────────────────────────────────────────
    # Réinitialisation de l'historique
    # ──────────────────────────────────────────────
    def reset_history(self) -> None:
        """Efface l'historique de conversation pour repartir à zéro."""
        self.chat_history = None
        print(f"{COLOR_INFO}  (Historique effacé){COLOR_RESET}\n")


# ──────────────────────────────────────────────────────────────────
# Boucle principale d'interaction
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Lance la boucle interactive de l'assistant Wara-IA.
    L'utilisateur saisit ses messages en ligne de commande.
    """
    print_banner()                # Affichage du logo ASCII
    wara = WaraIA()               # Instanciation et chargement du modèle

    # Commandes spéciales reconnues par le programme
    EXIT_COMMANDS    = {"quitter", "exit", "quit", "bye", "au revoir"}
    RESET_COMMANDS   = {"effacer", "reset", "clear"}
    HELP_COMMANDS    = {"aide", "help", "?"}

    # ── Boucle de conversation ────────────────────────────────────
    while True:
        try:
            # Lecture de l'entrée utilisateur
            user_input = input(f"{COLOR_USER}Vous  > {COLOR_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            # Gestion de Ctrl+C ou fin de flux (ex. redirection depuis un fichier)
            print(f"\n{COLOR_INFO}(Interruption détectée — au revoir !){COLOR_RESET}")
            break

        # ── Ignorer les entrées vides ─────────────────────────────
        if not user_input:
            continue

        # ── Commandes spéciales ────────────────────────────────────
        if user_input.lower() in EXIT_COMMANDS:
            print(f"\n{COLOR_LION}{COLOR_BOLD}Wara  > {COLOR_RESET}{COLOR_BOT}À bientôt ! Que le lion vous accompagne. 🦁{COLOR_RESET}\n")
            break

        if user_input.lower() in RESET_COMMANDS:
            wara.reset_history()
            continue

        if user_input.lower() in HELP_COMMANDS:
            help_text = (
                "Commandes disponibles :\n"
                "  • 'quitter' / 'exit'  — terminer la session\n"
                "  • 'effacer' / 'reset' — réinitialiser l'historique\n"
                "  • 'aide'    / 'help'  — afficher cette aide\n"
            )
            print_message("wara", help_text)
            continue

        # ── Génération de la réponse ──────────────────────────────
        start = time.time()
        response = wara.respond(user_input)
        elapsed = time.time() - start

        # Affichage de la réponse avec le temps de traitement
        print_message("wara", response)
        print(f"{COLOR_INFO}  [{elapsed:.2f}s]{COLOR_RESET}")


# ──────────────────────────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

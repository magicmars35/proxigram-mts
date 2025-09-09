# PROXIGRAM - MTS
Transcripteur & Résumeur de Réunions Gradio (FFmpeg v8 Whisper + LM Studio)

Une petite application Gradio qui **enregistre depuis le microphone de votre navigateur** ou **importe un fichier audio**, le transcrit **localement** à l’aide de **FFmpeg v8 et du filtre `whisper`** (avec des modèles Whisper.cpp), puis envoie la transcription à un **LLM local (LM Studio)** pour produire des **comptes rendus de réunion** propres et structurés au format Markdown.

> **Points forts**
>
> * Enregistrement ou importation d’un seul clic (Gradio 4.x `Audio`).
> * Capture robuste : **pré‑roll** (pour éviter de manquer les premiers mots), **VAD** optionnel et **normalisation** de l’entrée en WAV 16 kHz mono.
> * Sortie flexible : **texte / srt / json**.
> * **CRUD** intégré pour les modèles d’invite, enregistrés dans `prompt_templates.json` et utilisés comme **prompt système** pour LM Studio.
> * **Options d’interface** persistantes (langue 🇬🇧/🇫🇷 et thème clair/sombre) sauvegardées dans `ui_settings.json`.
> * Nouvel onglet **Transcriptions** avec un CRUD complet pour revisiter les transcriptions et leurs résumés.

---

## Architecture (aperçu)

1. **Interface Gradio** (microphone/import → composant `Audio`).
2. **FFmpeg 8 + filtre `whisper`** (avec le modèle **ggml** de Whisper.cpp) → fichier de transcription.
   * **Pré‑roll** optionnel (ajoute un court silence au début) et **VAD**.
   * L’audio est normalisé en **WAV 16 kHz mono** pour plus de fiabilité.
3. **LM Studio** (API compatible OpenAI) → compte rendu de réunion structuré en Markdown. Ollama fonctionne aussi.

---

## Prérequis

* **Python** 3.10+
* **FFmpeg 8.0+** compilé avec le **filtre `whisper`** (nécessite Whisper.cpp). Vérifiez avec :
  * Linux/macOS : `ffmpeg -hide_banner -filters | grep whisper`
  * Windows : `ffmpeg -hide_banner -filters | findstr whisper`
* **LM Studio** en mode serveur local (compatible OpenAI) ou Ollama, généralement sur `http://localhost:1234`.
* Fichier(s) de modèle **Whisper.cpp ggml**, par ex. `ggml-large-v3-turbo.bin`.

Packages Python :

```bash
gradio>=4.44.1
uvicorn>=0.30
starlette>=0.37
anyio>=4.4
h11>=0.14
httpx>=0.27
httpcore>=1.0
python-dotenv
```

> **Confidentialité** : tout le traitement audio se fait localement via FFmpeg ; le résumé est généré par votre instance locale de LM Studio.

---

## Installation

1. **Cloner** le dépôt et y entrer.
   ```bash
   git clone https://github.com/magicmars35/AutoTranscriptReport
   cd AutoTranscriptReport
   ```

2. (Optionnel mais recommandé) **Créer un environnement virtuel** et l’activer.

   Pour Windows :

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Installer les dépendances** :

   ```bash
   pip install -r requirements.txt
   ```
4. **Placer un modèle Whisper.cpp** dans `./models/`, par ex. `./models/ggml-large-v3-turbo.bin`. Les modèles sont disponibles ici : https://huggingface.co/ggerganov/whisper.cpp/tree/main
5. **S’assurer que FFmpeg 8** prend en charge le filtre `whisper` (voir Prérequis).
6. **Lancer LM Studio** (ou Ollama) en mode serveur (onglet Developer → Start server). L’URL de base par défaut est `http://localhost:1234` et le chemin de l’API est généralement `/v1/chat/completions`.

---

## Configuration (.env)

Créez un fichier `.env` à la racine du projet si vous souhaitez remplacer les valeurs par défaut :

```ini
# LM Studio
LMSTUDIO_BASE_URL=http://localhost:1234
LMSTUDIO_API_PATH=/v1/chat/completions
LMSTUDIO_MODEL=Qwen2.5-7B-Instruct
LMSTUDIO_API_KEY=lm-studio

# FFmpeg + Whisper
FFMPEG_BIN=ffmpeg
WHISPER_MODEL_PATH=./models/ggml-large-v3-turbo.bin
WHISPER_LANGUAGE=fr

# Stockage des modèles d’invite
TEMPLATES_PATH=prompt_templates.json
```

Les valeurs par défaut sont les mêmes que dans l’exemple ci-dessus. Vous pouvez aussi modifier les constantes en haut du fichier Python.

---

## Exécution

```bash
python app.py
```

Gradio affichera une URL locale (ex. `http://127.0.0.1:7860`).

---

## Utilisation de l’application

1. **Enregistrer ou Importer** : utilisez l’unique widget `Audio` pour enregistrer depuis le micro du navigateur ou importer un fichier existant (`.wav`, `.mp3`, `.m4a`, etc.).
2. **Choisir le format de sortie** : `srt` (par défaut) pour vérifier visuellement les horodatages ; `text` pour du texte brut ; `json` pour un usage programmatique.
3. **(Optionnel) Paramètres avancés** :
   * **Pré‑roll (ms)** : ajoute un court silence au début pour ne pas couper les premiers mots (essayez 250–500 ms).
   * **Queue (ms)** : fenêtre de mise en mémoire tampon pour le VAD ; des valeurs plus grandes peuvent stabiliser la segmentation.
   * **VAD (Silero)** : n’activez que si vous fournissez un **modèle VAD Silero** en format ggml ; sinon laissez‑le désactivé pour préserver les pauses naturelles.
4. Cliquez sur **Transcrire** → FFmpeg exécute le filtre `whisper` et écrit le fichier de transcription dans `./transcripts/`.
5. Consultez la transcription/SRT/JSON affichée dans l’interface.
6. Choisissez ou modifiez un **Modèle d’Invite** et cliquez sur **Résumer** → LM Studio renvoie un **compte rendu de réunion** structuré en Markdown.
7. (Optionnel) Utilisez l’onglet **Options** pour changer la langue de l’interface (anglais/français) ou le thème (clair/sombre). Les choix sont enregistrés dans `ui_settings.json`.

---

## Modèles d’Invite (CRUD)

* Les modèles sont enregistrés dans **`prompt_templates.json`** à la racine du projet.
* Chaque entrée associe un **nom ➝ contenu du prompt**.
* L’interface offre des boutons pour **Recharger**, **Enregistrer** et **Supprimer** les modèles. Le contenu du modèle sélectionné est envoyé comme prompt système.
* Vous pouvez également éditer manuellement `prompt_templates.json` lorsque l’application est arrêtée.

**Exemple de `prompt_templates.json` :**

```json
{
  "default": "Vous êtes un assistant spécialisé dans les comptes rendus de réunion...",
  "brief": "Rédige un résumé concis centré sur les décisions et actions."
}
```

> L’application tronque le contenu du modèle à une longueur sûre avant de l’envoyer à LM Studio.

---

## Paramètres de l’interface

* L’onglet **Options** permet de choisir la langue de l’interface (anglais ou français) et le thème (clair ou sombre).
* Les libellés pour chaque langue sont stockés dans **`strings/<lang>.json`** pour faciliter la traduction.
* Les sélections sont enregistrées dans **`ui_settings.json`** afin que vos préférences soient restaurées au prochain lancement.
* Supprimez ou modifiez ce fichier pour réinitialiser les paramètres de l’interface.

---

## Choix du modèle

Placez votre fichier de modèle **Whisper.cpp ggml** choisi sous `./models` et pointez `WHISPER_MODEL_PATH` dessus. Recommandations :

* **`ggml-large-v3-turbo.bin`** (~1,5 Go) : excellent compromis qualité/vitesse, multilingue.
* `ggml-large-v3.bin` (~2,9 Go) : meilleure qualité, plus lourd.
* `ggml-medium.bin` (~1,5 Go) : bonne qualité, plus léger que large.
* `ggml-small.bin` (~466 Mo) : rapide et correct pour le français.

Tous les modèles sont multilingues sauf si le nom de fichier se termine par `.en`.
Les modèles sont disponibles ici : https://huggingface.co/ggerganov/whisper.cpp/tree/main

---

## Conseils & Dépannage

* **Vérifiez la disponibilité du filtre `whisper`** : si `ffmpeg -filters` n’affiche pas `whisper`, votre build n’est pas compatible. Installez/compilez FFmpeg 8 avec le support Whisper.
* **Chemins Windows** : le parseur du filtre FFmpeg n’aime pas les lettres de lecteur (`D:`) ni les antislashs. Cette appli écrit les transcriptions avec des **chemins relatifs au format POSIX** (slashs `/`) pour éviter cela.
* **Problèmes de micro / premiers mots manquants** : augmentez le **pré‑roll** (par ex. 300–800 ms). Laissez **VAD désactivé** sauf nécessité. L’appli convertit aussi automatiquement les entrées en **WAV 16 kHz mono** pour la cohérence.
* **Audio partiellement transcrit** : essayez de désactiver VAD, d’augmenter `queue (ms)` et de vérifier que votre entrée n’est pas corrompue.
* **Erreurs LM Studio (404/connexion refusée)** : assurez-vous que le serveur local de LM Studio est en cours d’exécution et que `LMSTUDIO_BASE_URL` et `LMSTUDIO_API_PATH` correspondent à votre version.
* **Lent sur CPU** : préférez `large-v3-turbo` ou `small`; les variantes quantifiées peuvent aider.

---

## Feuille de route (idées)

* Export en **DOCX/PDF** du compte rendu final.
* **Diarisation** / étiquettes de locuteurs.
* Post‑traitement multilingue et traduction.
* Mode batch et dossiers surveillés.

---

## Licence

MIT

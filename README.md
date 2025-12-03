# 🧪 Modrinth Compatibility & Modpack Helper

A small web companion for **building modpacks with friends** and avoiding version hell.

Paste a list of **Modrinth mod links or slugs**, choose a **loader**, and the app:

- Finds which **Minecraft version** has the **highest compatibility** with your mods.
- Shows which mods are **compatible**, **incompatible**, or **only available for other modloaders**.
- Lets you **export a ready-to-import modpack** (`.zip` / `.mrpack-style`) for **Prism Launcher** / **Modrinth Launcher**.

> 💡 Think of it as:  
> *“I want to merge everyone’s favourite mods and quickly find a version that actually works.”*

---

## ✨ Features

- 🔍 **Smart compatibility analysis**
  - Paste multiple Modrinth links/slugs.
  - Choose your modloader: **Fabric**, **Quilt**, **NeoForge** (Forge support depends on external APIs/servers).
  - For each Minecraft version, see:
    - `X of Y mods compatible` (+ percentage)
    - Detailed lists of:
      - ✅ Compatible mods  
      - ❌ Not compatible for that version  
      - ⚠️ Available, but only for **other loaders**

- 🔗 **Integrated Modrinth search**
  - Search mods **directly inside the app** (no need to leave the page).
  - Add or remove mods from your list with a single click.
  - Automatically detects mods you’ve already added and offers **“Remove”** instead of **“Add”**.

- 🧮 **Mod counter**
  - Always shows how many mods are currently selected, e.g.:
    - `47 mods to analyze`

- 🌍 **Multi-language**
  - UI fully translated to several languages (including English and Spanish).
  - All core texts (buttons, warnings, titles, themes) go through the translation system.

- 🎨 **Multi-theme UI**
  - Multiple visual themes:
    - Default
    - Dark / OLED-friendly
    - Sepia dark and more
  - Consistent styling across themes (buttons, highlights, chips).

- 📦 **Export to modpack**
  - For each version you can:
    - **Export a modpack file** (`.zip` with `modrinth.index.json`, compatible with `.mrpack` format).
  - Tested with:
    - ✅ **Prism Launcher**
    - ✅ **Modrinth Launcher**
  - Typical flow:
    1. Build your mod list in the web app.
    2. Export the pack for the version+loader you liked.
    3. Import into Prism/Modrinth.
    4. Hit **“Check / Verify dependencies”** in the launcher to pull in any remaining deps.

- 📱 **Web-based**
  - Runs in the browser.
  - Usable from PC **and** mobile (via the deployed URL).

---

## 🧠 How it works (high level)

1. You provide a list of **Modrinth mods** (links or identifiers).
2. The app queries the **Modrinth API** for:
   - Available versions
   - Supported game versions
   - Supported loaders
3. For each **Minecraft version + loader** combination, it:
   - Checks which mods have **compatible builds**.
   - Counts:
     - Compatible
     - Incompatible
     - “Other loader only”
4. The UI then presents:
   - A ranked list of versions with compatibility percentages.
   - Detailed lists of mods per category.
   - Buttons to export a **Modrinth-style modpack** with the compatible mods for that version.

---

## 🚀 Live Demo

> 🔗 **Live URL:**  
> `<https://tu-url-de-render.onrender.com>`  
> *(Replace this with your actual Render URL.)*

You can open it from:

- Your **PC** to build and export packs.
- Your **phone** to draft mod lists and compatibility before you’re at your gaming PC.

---

## 🛠️ Tech Stack

- **Backend:** Python, [Flask](https://flask.palletsprojects.com/)
- **Frontend:** HTML, CSS, vanilla JavaScript
- **HTTP / API:** `requests` (+ optionally `httpx` for optimized API calls)
- **Packing:** Generates `.zip` files with `modrinth.index.json` inside (compatible with `.mrpack` structure)
- **Deployment:** [Render](https://render.com/) free tier (auto-deploy from GitHub)

---

## 🧩 Supported Launchers

The exported packs are intended for:

- ✅ **Prism Launcher**
- ✅ **Modrinth Launcher**

Other launchers **may** work if they support Modrinth-style modpacks, but are not the main target.

> ❌ **CurseForge Launcher** does **not** support Modrinth modpacks (by design of their ecosystem).

---

## 📦 Running locally

If you want to run this on your own machine by cloning the repo and excuting inside the folder:

#Only the first time
pip install -r requirements.txt

#To execute the program
py .\app.py

And open http://127.0.0.1:5000/ in your browser

Close the program tab and CMD to close the program

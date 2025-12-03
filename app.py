import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple
from io import BytesIO
import zipfile
import json
import xml.etree.ElementTree as ET
from functools import lru_cache

import requests
from flask import Flask, jsonify, render_template, send_file

app = Flask(__name__)

MODRINTH_VERSION_ENDPOINT = "https://api.modrinth.com/v2/project/{project_id}/version"
MODRINTH_SEARCH_ENDPOINT = "https://api.modrinth.com/v2/search"
MODRINTH_SEARCH_ENDPOINT = "https://api.modrinth.com/v2/search"
MODRINTH_VERSION_BY_ID = "https://api.modrinth.com/v2/version/{version_id}"


@lru_cache(maxsize=512)
def _get_modrinth_versions_payload(project_id: str):
    """Descarga y cachea el payload de versiones de un proyecto Modrinth."""
    resp = requests.get(MODRINTH_VERSION_ENDPOINT.format(project_id=project_id), timeout=15)
    if resp.status_code != 200:
        raise requests.RequestException(f"status {resp.status_code}")
    return resp.json()


@lru_cache(maxsize=1024)
def _get_version_by_id(version_id: str):
    resp = requests.get(MODRINTH_VERSION_BY_ID.format(version_id=version_id), timeout=15)
    if resp.status_code != 200:
        raise requests.RequestException(f"status {resp.status_code}")
    return resp.json()


def parse_projects(raw_input: str) -> List[Dict]:
    """
    Extract Modrinth projects from text.
    Supports Modrinth URLs or plain slugs separated by commas/newlines.
    """
    tokens = re.split(r"[,\n\r]+", raw_input or "")
    projects = []
    mr_pattern = re.compile(r"modrinth\.com/(?:mod)/([^/?#]+)", re.IGNORECASE)

    for token in tokens:
        candidate = token.strip()
        if not candidate:
            continue

        slug = candidate
        mr_match = mr_pattern.search(candidate)
        if mr_match:
            slug = mr_match.group(1)
        elif candidate.lower().startswith("mr:"):
            slug = candidate[3:]
        else:
            # ignore non-Modrinth formats
            continue

        url = f"https://modrinth.com/mod/{slug}"
        display_name = slug.replace("-", " ").strip().title() or slug
        projects.append({"id": slug, "label": slug, "url": url, "display_name": display_name})

    # Preserve order but remove duplicates
    seen = set()
    unique_projects = []
    for proj in projects:
        label = proj["label"]
        if label not in seen:
            unique_projects.append(proj)
            seen.add(label)
    return unique_projects


def fetch_modrinth_versions(project_id: str) -> Tuple[Dict[str, Set[str]], str]:
    """Fetch all game versions for a Modrinth project with loaders per version."""
    try:
        versions_payload = _get_modrinth_versions_payload(project_id)
    except requests.RequestException as exc:
        return {}, f"{project_id}: error de red ({exc})"
    except ValueError:
        return {}, f"{project_id}: respuesta inválida de Modrinth"

    game_versions: Dict[str, Set[str]] = {}
    for version in versions_payload:
        loaders = set(version.get("loaders", []))
        for gv in version.get("game_versions", []):
            game_versions.setdefault(gv, set()).update(loaders)

    if not game_versions:
        return {}, f"{project_id}: sin versiones de Minecraft encontradas"

    return game_versions, ""


def fetch_latest_version_file(project_id: str, mc_version: str, loader: str) -> Tuple[Dict, str]:
    """
    Get the latest version file metadata for a given project / mc version / loader.
    Returns a dict with keys: filename, url, hashes (sha1/sha512), version_number.
    """
    try:
        version = _pick_best_version(project_id, mc_version, loader)
    except requests.RequestException as exc:
        return {}, f"{project_id}: error de red ({exc})"
    except ValueError:
        return {}, f"{project_id}: respuesta inválida de Modrinth"

    if not version:
        return {}, ""

    file_entry = _pick_main_file(version)
    if not file_entry:
        return {}, ""
    hashes = file_entry.get("hashes", {})
    return (
        {
            "filename": file_entry.get("filename") or f"{project_id}.jar",
            "url": file_entry.get("url") or "",
            "hashes": {k: v for k, v in hashes.items() if k in {"sha1", "sha512", "sha256"} and v},
            "version_number": version.get("version_number") or "",
            "project_id": project_id,
            "file_size": file_entry.get("size") or 0,
        },
        "",
    )


@lru_cache(maxsize=64)
def fetch_latest_loader_version(loader: str, mc_version: str) -> str:
    """
    Try to resolve the latest loader version compatible with the given MC version.
    Falls back to "latest" if unavailable.
    """
    loader = loader.lower()

    def clean(ver: str) -> str:
        if not ver:
            return ""
        if isinstance(ver, str) and ver.lower() == "latest":
            return ""
        return str(ver)
    try:
        if loader == "fabric":
            resp = requests.get(f"https://meta.fabricmc.net/v2/versions/loader/{mc_version}", timeout=10)
            if resp.ok:
                data = resp.json()
                if data:
                    return clean(data[0]["loader"]["version"])
        elif loader == "quilt":
            resp = requests.get(f"https://meta.quiltmc.org/v3/versions/loader/{mc_version}", timeout=10)
            if resp.ok:
                data = resp.json()
                if data:
                    return clean(data[0]["loader"]["version"])
        elif loader == "neoforge":
            try:
                resp = requests.get(f"https://meta.neoforged.net/v2/versions/neoforge/{mc_version}", timeout=10)
                if resp.ok:
                    data = resp.json()
                    if data:
                        version = clean(data[0].get("version"))
                        if version:
                            return version
            except requests.RequestException:
                pass
            # Fallback: parse maven metadata and pick highest matching MC prefix
            try:
                meta_resp = requests.get("https://maven.neoforged.net/releases/net/neoforged/neoforge/maven-metadata.xml", timeout=10)
                if meta_resp.ok:
                    root = ET.fromstring(meta_resp.text)
                    versions = [v.text for v in root.findall(".//version") if v.text]
                    target = version_tuple(mc_version, length=3)
                    target_major = target[1]  # from MC 1.x.y -> x is at index 1
                    target_minor = target[2]  # y is at index 2

                    def neo_parts(ver: str):
                        vt = version_tuple(ver, length=3)
                        return vt

                    filtered = [
                        v for v in versions
                        if neo_parts(v)[0] == target_major and neo_parts(v)[1] == target_minor
                    ]
                    if filtered:
                        filtered.sort(key=version_tuple, reverse=True)
                        return clean(filtered[0])
                    if versions:
                        versions.sort(key=version_tuple, reverse=True)
                        return clean(versions[0])
            except requests.RequestException:
                pass
            except ET.ParseError:
                pass
        elif loader == "forge":
            # Primary source: promotions_slim.json
            try:
                resp = requests.get("https://maven.minecraftforge.net/net/minecraftforge/forge/promotions_slim.json", timeout=10)
                if resp.ok:
                    data = resp.json()
                    promos = data.get("promos", {})
                    for key in (f"{mc_version}-recommended", f"{mc_version}-latest"):
                        if key in promos:
                            val = promos[key]
                            if isinstance(val, str):
                                if val.startswith(f"{mc_version}-"):
                                    return clean(val.split("-", 1)[1])
                                return clean(val)
            except requests.RequestException:
                pass

            # Fallback: parse maven metadata for the MC branch
            try:
                meta_resp = requests.get("https://maven.minecraftforge.net/net/minecraftforge/forge/maven-metadata.xml", timeout=10)
                if meta_resp.ok:
                    root = ET.fromstring(meta_resp.text)
                    versions = [v.text for v in root.findall(".//version") if v.text]
                    prefixed = [v for v in versions if v.startswith(f"{mc_version}-")]
                    if prefixed:
                        # sort by the forge part after the first dash
                        def forge_part(ver: str) -> Tuple[int, ...]:
                            part = ver.split("-", 1)[1] if "-" in ver else ver
                            return version_tuple(part)

                        prefixed.sort(key=forge_part, reverse=True)
                        best = prefixed[0]
                        return clean(best.split("-", 1)[1] if "-" in best else best)
            except requests.RequestException:
                pass
            except ET.ParseError:
                pass
    except requests.RequestException:
        return ""
    except (ValueError, KeyError, IndexError):
        return ""
    return ""

def _pick_best_version(project_id: str, mc_version: str, loader: str):
    """Select the latest version payload matching mc_version and loader."""
    versions_payload = _get_modrinth_versions_payload(project_id)
    candidates = []
    for version in versions_payload:
        gvs = set(version.get("game_versions", []))
        loaders = set(version.get("loaders", []))
        if mc_version in gvs and loader in loaders:
            candidates.append(version)
    if not candidates:
        return None
    candidates.sort(key=lambda v: v.get("date_published", "") or v.get("version_number", ""), reverse=True)
    return candidates[0]


def _pick_main_file(version: Dict) -> Dict:
    files = version.get("files") or []
    if not files:
        return {}
    primaries = [f for f in files if f.get("primary")]
    if primaries:
        return primaries[0]
    required_res = [f for f in files if f.get("file_type") == "required-resource"]
    if required_res:
        return required_res[0]
    jars = [f for f in files if (f.get("filename") or "").endswith(".jar")]
    if jars:
        return jars[0]
    return files[0]


def _supports(version: Dict, mc_version: str, loader: str) -> bool:
    gvs = set(version.get("game_versions", []))
    loaders = set(version.get("loaders", []))
    return mc_version in gvs and loader in loaders


def resolve_required_dependencies(root_versions: List[Dict], mc_version: str, loader: str, client: str = "required"):
    visited = set()
    to_process = list(root_versions)
    resolved = []
    unresolved = []

    while to_process:
        version = to_process.pop()
        vid = version.get("id")
        if vid and vid in visited:
            continue
        if vid:
            visited.add(vid)
        resolved.append(version)

        deps = version.get("dependencies") or []
        for dep in deps:
            if dep.get("dependency_type") != "required":
                continue
            env = dep.get("env") or {}
            client_env = env.get("client", "required")
            if client_env != "required":
                continue
            dep_version = None
            try:
                if dep.get("version_id"):
                    dep_version = _get_version_by_id(dep["version_id"])
                elif dep.get("project_id"):
                    dep_version = _pick_best_version(dep["project_id"], mc_version, loader)
            except requests.RequestException:
                dep_version = None
            if not dep_version or not _supports(dep_version, mc_version, loader):
                unresolved.append(dep.get("version_id") or dep.get("project_id") or "unknown")
                continue
            to_process.append(dep_version)

    return resolved, unresolved


def loader_dependency_key(loader: str) -> str:
    loader = loader.lower()
    if loader in {"forge", "neoforge"}:
        return loader
    if loader == "fabric":
        return "fabric-loader"
    if loader == "quilt":
        return "quilt-loader"
    return loader


def search_modrinth(query: str, limit: int = 20) -> Tuple[List[Dict], str]:
    params = {"query": query, "limit": limit}
    try:
        resp = requests.get(MODRINTH_SEARCH_ENDPOINT, params=params, timeout=15)
    except requests.RequestException as exc:
        return [], f"error de red ({exc})"

    if resp.status_code != 200:
        return [], f"error {resp.status_code} al buscar en Modrinth"

    try:
        payload = resp.json()
    except ValueError:
        return [], "respuesta inválida de Modrinth"

    hits = payload.get("hits", [])
    results = []
    for hit in hits:
        authors = []
        for author in hit.get("author_list", []):
            name = author.get("name")
            if name:
                authors.append(name)
        results.append(
            {
                "source": "modrinth",
                "project_id": hit.get("project_id"),
                "slug": hit.get("slug"),
                "title": hit.get("title"),
                "description": hit.get("description", ""),
                "downloads": hit.get("downloads", 0),
                "author": ", ".join(authors) if authors else hit.get("author", ""),
                "url": f"https://modrinth.com/mod/{hit.get('slug') or hit.get('project_id')}",
            }
        )
    return results, ""


def search_curseforge(query: str, limit: int = 8) -> Tuple[List[Dict], str]:
    params = {"gameId": 432, "query": query, "pageSize": limit}
    try:
        resp = requests.get(CURSEFORGE_SEARCH_ENDPOINT, params=params, timeout=15)
    except requests.RequestException as exc:
        return [], f"error de red ({exc})"

    if resp.status_code != 200:
        return [], f"error {resp.status_code} al buscar en CurseForge"

    try:
        payload = resp.json()
    except ValueError:
        return [], "respuesta inválida de CurseForge"

    data = payload.get("data", []) if isinstance(payload, dict) else []
    results: List[Dict] = []
    for hit in data:
        results.append(
            {
                "source": "curseforge",
                "slug": str(hit.get("slug") or hit.get("id")),
                "cf_id": hit.get("id"),
                "title": hit.get("name") or hit.get("slug") or hit.get("id"),
                "description": hit.get("summary", ""),
                "downloads": hit.get("downloadCount", 0) or hit.get("downloads", 0),
                "author": (hit.get("authors") or [{}])[0].get("name", "") if hit.get("authors") else hit.get("author", ""),
                "url": f"https://www.curseforge.com/minecraft/mc-mods/{hit.get('slug') or hit.get('id')}",
            }
        )

    return results[:limit], ""


def parse_version_filters(raw_input: str) -> List[str]:
    tokens = re.split(r"[,\n\r]+", raw_input or "")
    return [t.strip() for t in tokens if t.strip()]


def is_snapshot(version: str) -> bool:
    # Regla simplificada: toda versión que no empieza por "1." se considera snapshot
    return not version.startswith("1.")


def is_prerelease(version: str) -> bool:
    # Pre-release si contiene un guion tras el número (ej: 1.20.1-rc1)
    return "-" in version


def version_tuple(version: str, length: int = 3) -> Tuple[int, ...]:
    parts = re.split(r"[.\-]", version)
    nums: List[int] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            nums.append(int(part))
        else:
            m = re.match(r"(\d+)", part)
            if m:
                nums.append(int(m.group(1)))
            break
        if len(nums) >= length:
            break
    while len(nums) < length:
        nums.append(0)
    return tuple(nums[:length])


def matches_expression(version: str, expression: str) -> bool:
    expr = expression.strip()
    if not expr:
        return False
    if expr.endswith(".x"):
        prefix = expr[:-2]
        return version == prefix or version.startswith(prefix + ".")
    if "-" in expr:
        start, end = expr.split("-", 1)
        vt = version_tuple(version)
        return version_tuple(start) <= vt <= version_tuple(end)
    return version == expr


def should_include_version(
    version: str, filters: List[str], mode: str, include_snapshots: bool, include_prereleases: bool
) -> bool:
    if not include_snapshots and is_snapshot(version):
        return False
    if not include_prereleases and is_prerelease(version):
        return False
    if not filters:
        return True

    matches = any(matches_expression(version, f) for f in filters)
    if mode == "only":
        return matches
    if mode == "all_except":
        return not matches
    return True


def compute_compatibility(
    projects: List[Dict],
    version_filters: List[str] | None = None,
    filter_mode: str = "all_except",
    include_snapshots: bool = True,
    include_prereleases: bool = True,
    loader: str = "fabric",
) -> Dict:
    supported_loaders = {"fabric", "forge", "neoforge", "quilt"}
    if loader not in supported_loaders:
        loader = "fabric"

    compatibility: Dict[str, Dict[str, Set[str]]] = {}
    errors: List[str] = []
    project_urls: Dict[str, str] = {proj["label"]: proj["url"] for proj in projects}
    project_display: Dict[str, str] = {proj["label"]: proj.get("display_name", proj["label"]) for proj in projects}

    # Fetch in parallel to improve response times.
    max_workers = min(8, len(projects)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_modrinth_versions, proj["id"]): proj for proj in projects}
        for future in as_completed(futures):
            proj = futures[future]
            label = proj["label"]
            try:
                versions, err = future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}: error de ejecución ({exc})")
                continue

            if err:
                errors.append(err)
                continue
            compatibility[label] = versions

    total_projects = len(projects)
    counter: Counter = Counter()
    version_to_mods: Dict[str, Set[str]] = {}
    version_to_other_loader: Dict[str, Dict[str, Set[str]]] = {}

    for label, versions in compatibility.items():
        for gv, loaders in versions.items():
            if loader in loaders:
                counter[gv] += 1
                version_to_mods.setdefault(gv, set()).add(label)
            else:
                for l in loaders:
                    if l == loader or l not in supported_loaders:
                        continue
                    version_to_other_loader.setdefault(gv, {}).setdefault(l, set()).add(label)

    results = []
    filters = [v.strip() for v in (version_filters or []) if v.strip()]

    all_versions = set(counter.keys()) | set(version_to_other_loader.keys())

    for gv in sorted(all_versions):
        if not should_include_version(gv, filters, filter_mode, include_snapshots, include_prereleases):
            continue

        compatible_labels = sorted(version_to_mods.get(gv, set()))
        other_loader_map = {
            l: sorted(version_to_other_loader.get(gv, {}).get(l, set()))
            for l in supported_loaders
            if l != loader and version_to_other_loader.get(gv, {}).get(l)
        }
        incompatible_labels = sorted(
            proj["label"]
            for proj in projects
            if proj["label"] not in compatible_labels
            and all(proj["label"] not in other_loader_map.get(l, []) for l in other_loader_map)
        )
        compatible_mods = [
            {"label": project_display.get(lbl, lbl), "url": project_urls.get(lbl, "")} for lbl in compatible_labels
        ]
        incompatible_mods = [
            {"label": project_display.get(lbl, lbl), "url": project_urls.get(lbl, "")}
            for lbl in incompatible_labels
        ]
        other_loader_mods = {
            l: [
                {"label": project_display.get(lbl, lbl), "url": project_urls.get(lbl, "")}
                for lbl in labels
            ]
            for l, labels in other_loader_map.items()
        }
        count = len(compatible_labels)
        results.append(
            {
                "version": gv,
                "count": count,
                "total": total_projects,
                "percent": round((count / total_projects) * 100, 1) if total_projects else 0,
                "compatible_mods": compatible_mods,
                "incompatible_mods": incompatible_mods,
                "other_loader_mods": other_loader_mods,
                "is_snapshot": is_snapshot(gv),
                "is_prerelease": is_prerelease(gv),
            }
        )

    # Determine recommended version: highest count, tie-breaker by version string.
    recommended_version = None
    if results:
        recommended_version = sorted(results, key=lambda item: (-item["count"], item["version"]))[0]["version"]

    return {"results": results, "recommended": recommended_version, "errors": errors}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    from flask import request

    body = request.get_json(force=True, silent=True) or {}
    raw_urls = body.get("urls", "")
    raw_version_filters = body.get("versionFilters", "")
    filter_mode = body.get("versionFilterMode", "all_except")
    include_snapshots = bool(body.get("includeSnapshots", True))
    include_prereleases = bool(body.get("includePrereleases", True))
    loader = body.get("loader", "fabric")

    if filter_mode not in {"all_except", "only"}:
        filter_mode = "all_except"

    projects = parse_projects(raw_urls)

    if not projects:
        return jsonify({"error": "Proporciona al menos un enlace o ID de Modrinth."}), 400

    version_filters = parse_version_filters(raw_version_filters)

    result = compute_compatibility(
        projects,
        version_filters=version_filters,
        filter_mode=filter_mode,
        include_snapshots=include_snapshots,
        include_prereleases=include_prereleases,
        loader=loader,
    )
    return jsonify(result)


@app.route("/api/search")
def search():
    from flask import request

    query = (request.args.get("q") or "").strip()
    if not query:
        return jsonify({"error": "Falta parámetro q"}), 400

    results, err = search_modrinth(query)
    if err:
        return jsonify({"error": err}), 502
    return jsonify({"results": results, "errors": []})


@app.route("/api/export", methods=["POST"])
def export_zip():
    from flask import request

    body = request.get_json(force=True, silent=True) or {}
    raw_urls = body.get("urls", "")
    mc_version = body.get("mcVersion")
    loader = (body.get("loader", "fabric") or "fabric").lower()

    if not mc_version:
        return jsonify({"error": "Falta versión de Minecraft"}), 400

    projects = parse_projects(raw_urls)
    if not projects:
        return jsonify({"error": "Proporciona al menos un enlace o ID de Modrinth."}), 400

    version_payloads: List[Dict] = []
    errors: List[str] = []
    max_workers = min(8, len(projects)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_pick_best_version, proj["id"], mc_version, loader): proj
            for proj in projects
        }
        for future in as_completed(futures):
            proj = futures[future]
            try:
                version = future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{proj['label']}: error de ejecución ({exc})")
                continue
            if not version:
                errors.append(f"{proj['label']}: sin versión para {mc_version}/{loader}")
            else:
                version_payloads.append(version)

    if not version_payloads:
        return jsonify({"error": "No se encontraron archivos para exportar.", "errors": errors}), 400

    all_versions, unresolved = resolve_required_dependencies(version_payloads, mc_version, loader)
    if unresolved:
        errors.extend([f"Dependencia no resuelta: {u}" for u in unresolved])

    loader_version = fetch_latest_loader_version(loader, mc_version)
    if not loader_version:
        return jsonify({"error": "No se pudo resolver la versión específica del modloader."}), 400
    dep_key = loader_dependency_key(loader)

    index = {
        "formatVersion": 1,
        "game": "minecraft",
        "versionId": f"generated-{mc_version}-{loader}",
        "name": f"Modpack {mc_version} ({loader})",
        "files": [],
        "dependencies": {
            "minecraft": mc_version,
            dep_key: loader_version,
        },
    }

    files: List[Dict] = []
    for v in all_versions:
        file_entry = _pick_main_file(v)
        if not file_entry:
            continue
        hashes = file_entry.get("hashes", {})
        files.append(
            {
                "path": f"mods/{file_entry.get('filename') or 'mod.jar'}",
                "downloads": [file_entry.get("url") or ""],
                "hashes": {k: val for k, val in hashes.items() if k in {"sha1", "sha512", "sha256"} and val},
                "env": {"client": "required", "server": "required"},
                "fileSize": file_entry.get("size") or 0,
                "version": v.get("version_number", ""),
                "project_id": v.get("project_id", ""),
            }
        )

    if not files:
        return jsonify({"error": "No se encontraron archivos para exportar.", "errors": errors}), 400

    for f in files:
        index["files"].append(
            {
                "path": f["path"],
                "downloads": f["downloads"],
                "hashes": f["hashes"],
                "env": f["env"],
                "fileSize": f["fileSize"],
                "version": f["version"],
                "project_id": f["project_id"],
            }
        )

    mem = BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("modrinth.index.json", json.dumps(index, ensure_ascii=False, indent=2))
    mem.seek(0)
    return send_file(
        mem,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"modpack-{mc_version}-{loader}.zip",
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

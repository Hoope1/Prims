AGENTS.md — Leitfaden für AI‑Agents in diesem Python‑Repo (kombinierter Qualitäts‑Stack)

> Zweck: Dieses Dokument macht KI‑Agenten (und neue Teammitglieder) sofort produktiv. Es definiert Struktur, Konventionen, Tests, PR‑Vorgaben und programmatische Quality Gates — abgestimmt auf die folgende Tool‑Kombination, die zusammen sinnvoll ist:

Ruff · mypy · pytest + coverage.py · bandit · pre‑commit · interrogate · pip‑audit · radon + xenon · (scalene & py‑spy optional)

Dependency‑Management: Standard über pyproject.toml/Setuptools (Extras), optional poetry oder pip‑tools (nicht beides parallel im selben Workflow).



Technischer Kontext

Sprache/Runtime: Python ≥ 3.9 (getestet bis 3.13)

Projektgröße: ~800 LOC, wissenschaftlicher Code (NumPy/DEAP möglich)

Build/Config: pyproject.toml (Single Source of Truth)

Package‑Layout: src‑layout



---

1) Project Structure

/                    # Repo-Root
├─ pyproject.toml    # zentrale Tool-Konfiguration (Ruff, mypy, pytest, …)
├─ .pre-commit-config.yaml
├─ Makefile or justfile
├─ README.md
├─ AGENTS.md         # dieses Dokument
├─ src/              # Produktivcode (Top-Level-Package: `myproject/`)
│  └─ myproject/
│     ├─ __init__.py
│     ├─ core/      # Kernlogik, Algorithmen
│     ├─ io/        # Laden/Speichern, CLI/Parsing
│     ├─ utils/     # Hilfsfunktionen
│     └─ models/    # Datenstrukturen/Typen
├─ scripts/          # Ausführbare Skripte (dünne Wrapper um `src/`)
├─ tests/            # Pytest-Suite (Unit, Integration, Property-based)
│  ├─ conftest.py
│  └─ test_*.py
├─ docs/             # optional: Sphinx/MkDocs
└─ data/             # kleine Beispiel-/Testdaten (keine Geheimnisse)

Regeln

src‑Layout verpflichtend: Produktionscode nur unter src/myproject/....

Skripte sind dünn: keine Logik in scripts/; importiere aus src/.

Tests spiegeln Struktur: tests/test_<modul>.py korrespondiert zu src/....



---

2) Coding Conventions

2.1 Stil & Formatierung

Formatter & Linter: Ruff ist maßgeblich (Format + Lint + isort + pyupgrade + pydocstyle‑Checks).
Kein Black/isort parallel.

Zeilenlänge: 100 Zeichen.

Strings: doppelte Anführungszeichen.

Dokstrings: NumPy‑Stil (Parameters, Returns, Raises, Examples).


2.2 Typisierung

Type Hints verpflichtend für neue/öffentliche APIs.

mypy im strikten Modus; NumPy‑Plugin aktiv bei numerischem Code.


2.3 Benennung, Fehler, Logging

Mathe/DS‑Kurzformen wie i, j, k, x, y, z, n, m, df, X, Y, T erlaubt.

Spezifische Exceptions; keine nackten except:.

Logs statt Prints in Produktivcode; Prints in scripts/ ok.


2.4 Suppressions

# noqa, # type: ignore[...], # nosec nur gezielt und mit Begründung in derselben Zeile.



---

3) Testing Requirements

Framework: pytest + coverage.py (Ziel ≥ 80% Branch‑Coverage).

Struktur: Unit‑Tests primär; Integration mit @pytest.mark.integration.

Property‑based: Hypothesis für algorithmische/numerische Funktionen.

Performance: Unit‑Suite < 30s lokal; teure Tests @pytest.mark.slow.


Kernbefehle

pytest -q
pytest --cov=src --cov-report=term-missing:skip-covered --cov-fail-under=80
pytest -m "not slow"


---

4) PR Guidelines

Pflicht

1. Lokal grün: make all/just all ohne Fehler.


2. Tests enthalten: neue/änderte Logik hat Unit‑Tests; Coverage fällt nicht.


3. Type‑Safe: mypy fehlerfrei; begründete # type: ignore[...] direkt an der Stelle.


4. Lint/Format: ruff format --check . & ruff check . sauber.


5. Security: bandit ohne High‑Severity; pip-audit ohne kritische CVEs.


6. Docs/Dokstrings: NumPy‑Dokstrings für öffentliche API; README/Changelog bei Bedarf.



PR‑Beschreibung (Kurzvorlage)

Motivation · Lösung/Design · Tests · Breaking Changes (Ja/Nein) · Risiken/Follow‑ups



---

5) Programmatic Checks (Quality Gates)

> Diese Gates laufen per pre‑commit lokal und per CI bei PRs.



5.1 Lokale Hooks (pre‑commit)

Format & Lint: ruff format, ruff --fix

Types: mypy (NumPy‑Plugin)

Security: bandit & pip-audit

Docstrings: interrogate --fail-under=80

Sonstige: trailing-whitespace, end-of-file-fixer, check-yaml|json|toml


Aktivieren:

pip install pre-commit
pre-commit install
pre-commit run --all-files

5.2 CI‑Pipeline (GitHub Actions o. ä.)

Stages & Gates

1. Format‑Check – ruff format --check .


2. Lint – ruff check .


3. Type Check – mypy src/ tests/


4. Security – bandit -r src/ -c pyproject.toml & pip-audit


5. Tests + Coverage – pytest --cov=src --cov-fail-under=80


6. Complexity Gate – radon cc -nc src/ & xenon --max-absolute B --max-modules B --max-average A src/


7. Docstring‑Coverage – interrogate --fail-under=80



PR wird rot, wenn ein Gate fehlschlägt.


---

6) Tooling‑Details (Kurzreferenz)

6.1 Ruff (Format + Lint)

ruff format .
ruff format --check .
ruff check .
ruff check --fix .
ruff rule E501

Aktive Regelsets: E,W,F,I,B,C4,UP,ARG,SIM,NPY,RUF. Per‑File‑Ignores: tests/** (z. B. S101, ARG), __init__.py darf F401.

6.2 mypy (strict)

Daemon: dmypy run -- src/

Overrides für 3rd‑Party; NumPy‑Plugin aktiv.


6.3 pytest + coverage.py

--strict-markers --strict-config

HTML‑Report in htmlcov/


6.4 Security

bandit: B101 (assert) in Tests ok, produktiv vermeiden.

pip-audit: PyPA Advisory DB.


6.5 Komplexität & toter Code

radon misst; xenon erzwingt (B/B/A).

vulture optional (kein Gate), für toten Code.


6.6 Profiling (on‑demand)

scalene (CPU/Memory/GPU) · py‑spy (sampling, production‑safe).

Keine Artefakte einchecken; Ergebnisse lokal sichten.



---

7) Quickstart (für Menschen & AI‑Agenten)

# 1) Dev‑Setup
pip install -e ".[dev]"
pre-commit install

# 2) Alles auf einmal
make all     # oder: just all

# 3) Häufige Kommandos
ruff format . && ruff check --fix .
mypy src/ tests/
pytest --cov=src
bandit -r src/ -c pyproject.toml && pip-audit
# Komplexität lokal prüfen + erzwingen
radon cc -nc src/ && xenon --max-absolute B --max-modules B --max-average A src/


---

8) Konfig‑Ausschnitte (aus pyproject.toml)

[tool.ruff]
line-length = 100
target-version = "py39"
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E","W","F","I","B","C4","UP","ARG","SIM","NPY","RUF"]
ignore = ["E501","E731"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/**" = ["S101","ARG"]

[tool.ruff.format]
quote-style = "double"
docstring-code-format = true

[tool.mypy]
python_version = "3.9"
files = ["src", "tests"]
plugins = ["numpy.typing.mypy_plugin"]
strict_optional = true
warn_unused_ignores = true
no_implicit_reexport = true
show_error_codes = true

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
addopts = [
  "-ra",
  "--strict-markers",
  "--strict-config",
  "--cov=src",
  "--cov-report=term-missing:skip-covered",
  "--cov-fail-under=80",
]

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
exclude_lines = [
  "pragma: no cover",
  "if TYPE_CHECKING:",
  "raise NotImplementedError",
]

[tool.bandit]
exclude_dirs = ["tests", ".venv", "build"]
skips = ["B101"]

[tool.radon]
exclude = "tests/*,docs/*"
show_complexity = true
average = true


---

9) Makefile

.PHONY: help format lint test coverage clean install dev security complexity deadcode docstring-coverage docs profile profile-cpu profile-live all ci

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Install dev dependencies"
	@echo "  make format      - Format code with ruff"
	@echo "  make lint        - Run all linters"
	@echo "  make test        - Run tests"
	@echo "  make coverage    - Generate coverage report"
	@echo "  make security    - Run security checks"
	@echo "  make complexity  - Check code complexity (radon + xenon)"
	@echo "  make docs        - Generate documentation"
	@echo "  make clean       - Remove artifacts"
	@echo "  make all         - Run format, lint, test"

install:
	pip install -e .

dev:
	pip install -e ".[dev,profiling,docs]"
	pre-commit install

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	mypy src/

test:
	pytest

coverage:
	pytest --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

security:
	bandit -r src/ -c pyproject.toml
	pip-audit

complexity:
	radon cc -nc src/
	radon mi src/
	xenon --max-absolute B --max-modules B --max-average A src/

deadcode:
	vulture --min-confidence 80 src/

docstring-coverage:
	interrogate src/

docs:
	sphinx-build -b html docs docs/_build

profile:
	@echo "Use: scalene your_script.py"

profile-cpu:
	py-spy record -o profile.svg -- python -m your_entrypoint

profile-live:
	py-spy top --pid $$PID  # set PID before running

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete

all: format lint test

ci: lint test security complexity docstring-coverage


---

10) CI/CD (GitHub Actions)

# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Format check
        run: ruff format --check .
      
      - name: Lint
        run: ruff check .
      
      - name: Type check
        run: mypy src/
      
      - name: Security
        run: |
          bandit -r src/ -c pyproject.toml
          pip-audit
      
      - name: Tests
        run: pytest --cov=src --cov-report=xml

      - name: Complexity (radon + xenon)
        run: |
          radon cc -nc src/
          xenon --max-absolute B --max-modules B --max-average A src/

      - name: Docstring coverage (interrogate)
        run: interrogate --fail-under=80 src/
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml


---

11) Dependency‑Management

Standard: Setuptools via pyproject.toml + optionale Extras (.[dev], .[profiling], .[docs]).

Poetry (Alternative): für Lockfiles/Environments. Wenn genutzt: poetry install, poetry run pytest, Hooks per pre-commit identisch.

pip‑tools (Alternative): pip-compile/pip-sync zur Reproduzierbarkeit.



---

12) AI‑Arbeitsanweisungen (Do/Don’t)

Do

Neue Funktionen mit Type‑Hints, Tests und NumPy‑Dokstrings liefern.

Regeln respektieren; erst Failing Test, dann Fix.

Kleine, pure Funktionen; I/O an Rändern.


Don’t

Keine parallelen Formatter/Linter einführen (nur Ruff maßgeblich).

Keine Logik in scripts/ ablegen.

Keine großen Daten/Geheimnisse commiten.



---

13) Minimal‑Beispiele

# src/myproject/utils/arithmetic.py
from __future__ import annotations
from typing import Iterable

def mean(values: Iterable[float]) -> float:
    """Compute the arithmetic mean.

    Parameters
    ----------
    values
        Numeric sequence; must not be empty.

    Returns
    -------
    float
        Arithmetic average.

    Raises
    ------
    ValueError
        If *values* is empty.
    """
    vals = list(values)
    if not vals:
        raise ValueError("values must not be empty")
    return sum(vals) / len(vals)

# tests/test_arithmetic.py
import math
import pytest
from myproject.utils.arithmetic import mean

def test_mean_basic() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0

def test_mean_empty_raises() -> None:
    with pytest.raises(ValueError):
        mean([])

def test_mean_precision() -> None:
    assert math.isclose(mean([0.1, 0.2, 0.3]), 0.2, rel_tol=1e-9)


---

Abschluss

Wenn ein AI‑Agent neuen Code beiträgt, gilt: Tests + Typen + Dokstrings + Gates grün. Damit bleibt die Codebasis robust, nachvollziehbar und erweiterbar.



Zusatz Informationen!:

# Python Code Quality Tools: Vollständiger Leitfaden 2024/2025

## Executive Summary: Die 10 unverzichtbaren Tools

Für ein wissenschaftliches Python-Projekt mit ~800 Zeilen empfehle ich diese **optimale Tool-Kombination**:

**Top 5 Must-Have Tools:**
1. **Ruff** (0.8+) - Ersetzt Black, isort, Flake8, pyupgrade, pydocstyle in einem Tool (10-100x schneller) [Astral +2](https://docs.astral.sh/ruff/)
2. **mypy** (1.18+) - Type Checking für Typ-Sicherheit
3. **pytest** + **coverage.py** - Testing und Code Coverage
4. **bandit** - Security Scanning [GitHub](https://github.com/PyCQA/bandit) [PyPI](https://pypi.org/project/bandit/)
5. **pre-commit** - Automatisierung aller Checks

**Zusätzlich empfohlen:**
6. **interrogate** - Docstring Coverage Messung
7. **pip-audit** - Dependency Vulnerability Scanning [PyPI](https://pypi.org/project/pip-audit/)
8. **radon** oder **xenon** - Complexity Monitoring
9. **scalene** oder **py-spy** - Performance Profiling bei Bedarf
10. **poetry** oder **pip-tools** - Dependency Management

**Die moderne Revolution:** Ruff hat die Python-Tool-Landschaft 2024 transformiert und macht viele separate Tools überflüssig. [Plone Community +4](https://community.plone.org/t/drop-black-isort-flake8-and-use-ruff/16413)

---

## 1. CODE-FORMATTER

### 1.1 Ruff Format (EMPFOHLEN für neue Projekte)

**Status:** ✅ Aktiv maintained, Rust-basiert, Black-kompatibel

**Installation:**
```bash
pip install ruff
```

**Warum Ruff Format:**
- 10-100x schneller als Black [Astral +4](https://astral.sh/blog/the-ruff-formatter)
- 99.9%+ Black-kompatibel [Astral +2](https://astral.sh/blog/the-ruff-formatter)
- Integriert mit Linter
- Ein Tool für alles

**Vollständige Konfiguration (pyproject.toml):**
```toml
[tool.ruff]
line-length = 100  # Für wissenschaftlichen Code mit Formeln
target-version = "py39"
src = ["src", "scripts"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
docstring-code-format = true
docstring-code-line-length = 88
```

**CLI-Befehle:**
```bash
# Formatieren
ruff format .

# Nur checken (CI/CD)
ruff format --check .

# Mit Diff anzeigen
ruff format --diff .
```

### 1.2 Black (Etablierter Standard)

**Status:** ✅ Sehr aktiv, Version 24.10+

**Installation:**
```bash
pip install black
```

**Vollständige Konfiguration:**
```toml
[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
include = '\\.pyi?$'
extend-exclude = '''
/(
  \\.eggs
  | \\.git
  | \\.venv
  | build
  | dist
)/
'''
```

**CLI-Befehle:**
```bash
black .
black --check .  # CI/CD
black --diff .   # Änderungen anzeigen
```

**Pros/Cons:**
- ✅ Industriestandard, stabil, minimal konfigurierbar
- ✅ Weit verbreitet, IDE-Integration
- ❌ Langsamer als Ruff
- ❌ Keine Linting-Funktionen

---

## 2. LINTERS

### 2.1 Ruff Check (EMPFOHLEN - All-in-One)

**Was Ruff ersetzt:**
- ✅ Flake8 (pycodestyle, pyflakes, mccabe)
- ✅ isort (Import-Sortierung)
- ✅ pyupgrade (Syntax-Modernisierung)
- ✅ pydocstyle (Docstring-Checking)
- ✅ autoflake (Unused imports/variables) [Plone Community +5](https://community.plone.org/t/drop-black-isort-flake8-and-use-ruff/16413)
- ⚠️ Teile von pylint

**Vollständige Production-Konfiguration:**
```toml
[tool.ruff]
line-length = 100
target-version = "py39"
src = ["src"]

exclude = [
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
    "*.egg-info",
]

[tool.ruff.lint]
# Aktivierte Regelsets
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "NPY",    # NumPy-specific
    "PD",     # pandas-vet (falls pandas genutzt)
    "RUF",    # Ruff-specific
]

# Ignorierte Regeln
ignore = [
    "E501",    # Line too long (Formatter regelt das)
    "E731",    # Lambda assignments (in NumPy üblich)
    "NPY002",  # Legacy np.random (falls alte Patterns)
]

# Auto-fix erlaubt
fixable = ["ALL"]
unfixable = []

# Underscore-prefixed variables erlauben
dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Unused imports in __init__ OK
"tests/**" = ["S101", "ARG"]  # Asserts und unused args in Tests OK
"scripts/**" = ["T20"]  # Print statements in Scripts OK

[tool.ruff.lint.isort]
known-first-party = ["myproject"]
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]
combine-as-imports = true

[tool.ruff.lint.pydocstyle]
convention = "numpy"  # Für wissenschaftlichen Code
```

**CLI-Befehle:**
```bash
# Linting
ruff check .

# Mit Auto-Fix
ruff check --fix .

# Nur spezifische Regeln
ruff check --select I,UP .

# Regel-Dokumentation anzeigen
ruff rule E501

# Watch-Modus für Entwicklung
ruff check --watch .
```

**Pros/Cons:**
- ✅ 10-100x schneller als alternative Tools [Astral +3](https://astral.sh/blog/the-ruff-formatter)
- ✅ Ersetzt 8+ separate Tools
- ✅ 800+ Regeln verfügbar [GitHub +2](https://github.com/astral-sh/ruff)
- ✅ Excellent Auto-Fix
- ❌ Kein Plugin-System (by design)
- ❌ Etwas andere Regeln als Pylint

### 2.2 Pylint (Für tiefe Analyse)

**Status:** ✅ Aktiv, Version 4.0+

**Wann verwenden:** Zusätzlich zu Ruff für tiefere statische Analyse

**Vollständige Konfiguration:**
```toml
[tool.pylint.main]
jobs = 4
py-version = "3.9"
extension-pkg-whitelist = ["numpy", "pandas", "scipy", "deap"]

[tool.pylint.messages_control]
disable = [
    "C0330",  # Wrong hanging indentation
    "C0326",  # Bad whitespace
    "C0103",  # Invalid name (zu strikt für wissenschaftlichen Code)
    "R0913",  # Too many arguments
    "R0914",  # Too many local variables
    "W0212",  # Access to protected member
    "W0511",  # TODO/FIXME comments
]

# Gute Variablennamen für wissenschaftlichen Code
good-names = [
    "i", "j", "k",           # Loop counters
    "x", "y", "z",           # Koordinaten
    "df",                    # DataFrame
    "T", "X", "Y",           # Mathematische Notation
    "n", "m",                # Dimensionen
    "f", "g", "h",           # Funktionen
]

[tool.pylint.format]
max-line-length = 100
max-module-lines = 2000

[tool.pylint.design]
max-args = 10
max-attributes = 15
max-branches = 12
max-locals = 20
max-statements = 50
```

**CLI-Befehle:**
```bash
pylint src/
pylint -j 4 src/  # Parallel
pylint --errors-only src/  # Nur Errors
```

---

## 3. TYPE CHECKERS

### 3.1 mypy (EMPFOHLEN - Reifste Option)

**Status:** ✅ Sehr aktiv, Version 1.18+ [Mypy](https://mypy.readthedocs.io/en/stable/changelog.html) [Blogger](https://mypy-lang.blogspot.com/2025/07/mypy-117-released.html)

**Vollständige strenge aber praktische Konfiguration:**
```toml
[tool.mypy]
python_version = "3.9"
files = ["src", "tests"]
namespace_packages = false
explicit_package_bases = true

# Plugins für wissenschaftliche Pakete
plugins = ["numpy.typing.mypy_plugin"]

# Strict mode - granular konfiguriert
disallow_any_generics = true
disallow_subclassing_any = true
disallow_untyped_calls = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
strict_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
warn_unreachable = true
strict_equality = true
no_implicit_reexport = true

# Display
show_error_codes = true
show_column_numbers = true
pretty = true

# Incremental
cache_dir = ".mypy_cache"

# Per-Modul Overrides für Third-Party
[[tool.mypy.overrides]]
module = ["tests.*"]
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = ["scipy.*", "matplotlib.*", "deap.*"]
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["numpy.*"]
ignore_missing_imports = false
```

**CLI-Befehle:**
```bash
mypy src/
mypy --strict src/  # Maximale Strenge
dmypy run -- src/  # Daemon-Modus (schneller)
```

### 3.2 pyright (Für Geschwindigkeit)

**Status:** ✅ Aktiv, Microsoft

**pyrightconfig.json:**
```json
{
  "include": ["src"],
  "exclude": ["**/__pycache__", ".venv"],
  "typeCheckingMode": "standard",
  "reportMissingImports": "error",
  "reportMissingTypeStubs": false,
  "reportUnusedImport": "warning",
  "reportUnusedVariable": "warning",
  "reportUnknownMemberType": "none",
  "reportUnknownParameterType": "none",
  "reportUnknownArgumentType": "none",
  "pythonVersion": "3.9"
}
```

**Vergleich mypy vs pyright:**
- mypy: Reifer, Plugin-System, größeres Ökosystem [LinkedIn](https://www.linkedin.com/pulse/here-four-python-type-checkers-can-help-you-maintain-clean-code)
- pyright: 3-5x schneller, [Basedpyright](https://docs.basedpyright.com/dev/usage/mypy-comparison/) bessere VS Code Integration [GitHub](https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md)
- **Empfehlung:** mypy für wissenschaftliche Projekte (NumPy-Plugin) [LinkedIn](https://www.linkedin.com/pulse/here-four-python-type-checkers-can-help-you-maintain-clean-code)

---

## 4. SECURITY SCANNER

### 4.1 Bandit

**Installation:**
```bash
pip install bandit[toml]
```

**Konfiguration für wissenschaftlichen Code:**
```toml
[tool.bandit]
exclude_dirs = ["tests", ".venv", "build"]
skips = ["B101"]  # assert_used - in wissenschaftlichem Code legitim

# Für eval-Nutzung in kontrollierten Umgebungen (sparsam verwenden):
# Besser: Inline mit # nosec B307
```

**CLI-Befehle:**
```bash
bandit -r src/
bandit -r . -c pyproject.toml -f json -o bandit-report.json
```

### 4.2 pip-audit (EMPFOHLEN für Dependencies)

**Installation:**
```bash
pip install pip-audit
```

**CLI-Befehle:**
```bash
pip-audit
pip-audit -r requirements.txt
pip-audit --fix --dry-run  # Fix-Vorschläge anzeigen
```

**Pros:**
- ✅ Open Source, keine Lizenz-Einschränkungen
- ✅ Nutzt PyPA Advisory Database [PyPI](https://pypi.org/project/pip-audit/)
- ✅ Fix-Vorschläge [PortSwigger](https://portswigger.net/daily-swig/pip-audit-google-backed-tool-probes-python-environments-for-vulnerable-packages)

---

## 5. COMPLEXITY ANALYZER

### 5.1 Radon

**Installation:**
```bash
pip install radon xenon
```

**CLI-Befehle:**
```bash
# Cyclomatic Complexity
radon cc -s src/

# Nur C und schlechter
radon cc -nc src/

# Maintainability Index
radon mi src/

# Mit Xenon Thresholds durchsetzen
xenon --max-absolute B --max-modules B --max-average A src/
```

**Optimale Thresholds:**
- Cyclomatic Complexity: Max 10 (B rank) [Medium](https://medium.com/@sumansaurabh/measuring-function-complexity-in-python-tools-and-techniques-3410330425a1)
- Kritischer Code: Max 5 (A rank) [GitHub](https://github.com/rubik/radon/blob/master/radon/complexity.py)
- Maintainability Index: Min 20 (A rank) [Readthedocs](https://radon.readthedocs.io/en/latest/intro.html)

**Konfiguration:**
```toml
[tool.radon]
exclude = "tests/*,docs/*"
show_complexity = true
average = true
```

### 5.2 Flake8 + mccabe

**setup.cfg:**
```ini
[flake8]
max-complexity = 10
max-line-length = 100
exclude = .git,__pycache__,.venv
per-file-ignores =
    __init__.py:F401
    tests/*:C901
ignore = E203,W503,E501
```

---

## 6. DEAD CODE DETECTION

### 6.1 Vulture

**Installation:**
```bash
pip install vulture
```

**Konfiguration:**
```toml
[tool.vulture]
exclude = ["*test*.py", "*/migrations/*", ".venv/"]
ignore_decorators = ["@app.route", "@require_*"]
ignore_names = ["setUp*", "tearDown*", "test_*", "Meta"]
min_confidence = 80
sort_by_size = true
```

**CLI-Befehle:**
```bash
vulture src/
vulture --min-confidence 100 src/  # Nur 100% sicher
vulture --make-whitelist src/ > whitelist.py
vulture src/ whitelist.py
```

### 6.2 Autoflake

**Installation:**
```bash
pip install autoflake
```

**Konfiguration:**
```toml
[tool.autoflake]
in-place = true
recursive = true
remove-all-unused-imports = true
ignore-init-module-imports = true
remove-unused-variables = true
```

**CLI-Befehle:**
```bash
autoflake --in-place --remove-all-unused-imports src/
autoflake --check --recursive src/  # Nur prüfen
```

---

## 7. IMPORT SORTER

### 7.1 Ruff (integriert via --select I)

**Bereits in Ruff-Konfiguration enthalten** (siehe oben)

### 7.2 isort (Falls nicht Ruff)

**Konfiguration:**
```toml
[tool.isort]
profile = "black"
line_length = 100
skip_gitignore = true
known_first_party = ["myproject"]
import_heading_stdlib = "Standard library"
import_heading_thirdparty = "Third party"
import_heading_firstparty = "Project imports"
include_trailing_comma = true
use_parentheses = true
multi_line_output = 3
force_grid_wrap = 0
ensure_newline_before_comments = true
```

**CLI-Befehle:**
```bash
isort .
isort --check-only --diff src/
```

---

## 8. DOCSTRING TOOLS

### 8.1 Interrogate (Coverage-Messung)

**Installation:**
```bash
pip install interrogate
```

**Konfiguration:**
```toml
[tool.interrogate]
fail-under = 80
verbose = 2
color = true
style = "numpy"  # oder "google"
ignore-init-method = false
ignore-magic = true
ignore-private = true
exclude = ["tests", "docs", "build"]
generate-badge = "."
badge-format = "svg"
```

**CLI-Befehle:**
```bash
interrogate src/
interrogate -vv src/
interrogate --generate-badge docs/ src/
```

### 8.2 pydocstyle (oder Ruff mit --select D)

**Wenn Ruff:** Bereits in Konfiguration mit `convention = "numpy"`

**Standalone:**
```toml
[tool.pydocstyle]
convention = "numpy"
add-ignore = ["D100", "D104"]
match = "(?!test_).*\\.py"
property_decorators = ["property", "cached_property"]
```

---

## 9. PERFORMANCE PROFILER

### 9.1 py-spy (Production-safe)

**Installation:**
```bash
pip install py-spy
```

**Nutzung:**
```bash
# Flame graph erstellen
py-spy record -o profile.svg -- python script.py

# An laufenden Prozess anhängen
py-spy record -o profile.svg --pid 12345

# Live top-like view
py-spy top --pid 12345
```

### 9.2 scalene (CPU + Memory + GPU)

**Installation:**
```bash
pip install scalene
```

**Nutzung:**
```bash
scalene your_script.py
scalene --html --outfile profile.html your_script.py
scalene --cpu-only your_script.py  # Schneller
```

### 9.3 line_profiler (Line-by-Line)

**Installation:**
```bash
pip install line_profiler
```

**Nutzung:**
```python
import line_profiler

@line_profiler.profile
def slow_function():
    # Code
    pass
```

```bash
LINE_PROFILE=1 python script.py
# Oder klassisch:
kernprof -l -v script.py
```

### 9.4 memray (Memory Leak Detection)

**Installation:**
```bash
pip install memray
```

**Nutzung:**
```bash
python -m memray run -o output.bin script.py
python -m memray flamegraph output.bin
python -m memray table output.bin
```

---

## 10. CODE MODERNIZER

### 10.1 Ruff (integriert via --select UP)

**Bereits in Ruff-Konfiguration enthalten**

### 10.2 pyupgrade (Standalone)

**Installation:**
```bash
pip install pyupgrade
```

**CLI-Befehle:**
```bash
pyupgrade --py39-plus *.py
pyupgrade --py310-plus src/**/*.py
```

### 10.3 flynt (f-string Conversion)

**Installation:**
```bash
pip install flynt
```

**Konfiguration:**
```toml
[tool.flynt]
transform-concats = true
line-length = 100
aggressive = false
```

**CLI-Befehle:**
```bash
flynt src/
flynt --dry-run src/
```

---

## 11. REFACTORING TOOLS

### 11.1 rope (IDE-Integration) [GitHub](https://github.com/python-rope/rope)

**Installation:**
```bash
pip install rope
```

**Nutzung:** Hauptsächlich über IDE (VS Code, PyCharm, Vim, Emacs) [Stack Overflow](https://stackoverflow.com/questions/28796/what-refactoring-tools-do-you-use-for-python) [Real Python](https://realpython.com/python-refactoring/)

### 11.2 refurb (Modernisierungs-Hints) [GitHub](https://github.com/dosisod/refurb) [PyPI](https://pypi.org/project/refurb/)

**Installation:**
```bash
pip install refurb
```

**Konfiguration:**
```toml
[tool.refurb]
enable_all = false
disable = ["FURB105"]
python_version = "3.9"
```

**CLI-Befehle:**
```bash
refurb .
refurb --explain FURB123
```

---

## 12. PRE-COMMIT FRAMEWORK

### Installation

```bash
pip install pre-commit
```

### Vollständige .pre-commit-config.yaml [pre-commit](https://pre-commit.com/)

```yaml
# .pre-commit-config.yaml
default_language_version:
  python: python3.9

repos:
  # Ruff - Linting und Formatierung
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # Type Checking
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.18.0
    hooks:
      - id: mypy
        additional_dependencies:
          - numpy
          - types-requests
        args: [--config-file=pyproject.toml]

  # Security
  - repo: https://github.com/PyCQA/bandit
    rev: 1.8.6
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]

  # Dead Code
  - repo: https://github.com/jendrikseipp/vulture
    rev: v2.3
    hooks:
      - id: vulture
        args: [--min-confidence=100]

  # Standard Hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: debug-statements
      - id: mixed-line-ending

  # Docstring Coverage (optional)
  - repo: https://github.com/econchick/interrogate
    rev: 1.7.0
    hooks:
      - id: interrogate
        args: [--quiet, --fail-under=80]
        pass_filenames: false
```

**Setup:**
```bash
pre-commit install
pre-commit run --all-files
```

---

## 13. ALL-IN-ONE TOOLS

### 13.1 Ruff (EMPFOHLEN)

**Vollständige Konfiguration bereits oben gezeigt**

### 13.2 Prospector (Alternative)

**Installation:**
```bash
pip install prospector[with_everything]
```

**Konfiguration (.prospector.yaml):**
```yaml
strictness: medium
doc-warnings: true
test-warnings: false
max-line-length: 100

pylint:
  run: true
  disable:
    - C0103
    - R0913

pep8:
  run: true
  max-line-length: 100

pyflakes:
  run: true

mccabe:
  run: true
  max-complexity: 10
```

---

## 14. DEPENDENCY MANAGEMENT

### 14.1 poetry (EMPFOHLEN für neue Projekte)

**Installation:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**pyproject.toml:**
```toml
[tool.poetry]
name = "myproject"
version = "0.1.0"
description = "Scientific Python project with DEAP"

[tool.poetry.dependencies]
python = "^3.9"
numpy = "^1.24"
deap = "^1.4"
tqdm = "^4.66"

[tool.poetry.group.dev.dependencies]
ruff = "^0.8"
mypy = "^1.18"
pytest = "^7.4"
pytest-cov = "^4.1"
pre-commit = "^3.5"
interrogate = "^1.7"
bandit = {extras = ["toml"], version = "^1.8"}
```

**Befehle:**
```bash
poetry install
poetry add numpy
poetry add --group dev pytest
poetry run pytest
poetry shell
```

### 14.2 pip-tools

**Installation:**
```bash
pip install pip-tools
```

**requirements.in:**
```
numpy>=1.24
deap>=1.4
tqdm>=4.66
```

**requirements-dev.in:**
```
-r requirements.txt
ruff>=0.8
mypy>=1.18
pytest>=7.4
pytest-cov>=4.1
```

**Befehle:**
```bash
pip-compile requirements.in
pip-compile requirements-dev.in
pip-sync requirements-dev.txt
```

---

## 15. TESTING & COVERAGE

### 15.1 pytest

**Konfiguration:**
```toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing:skip-covered",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow",
    "integration: integration tests",
]
```

**Befehle:**
```bash
pytest
pytest -v
pytest --cov=src tests/
pytest -k "test_function_name"
pytest -m "not slow"
```

### 15.2 coverage.py

**Konfiguration:**
```toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/__init__.py",
]

[tool.coverage.report]
precision = 2
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]

[tool.coverage.html]
directory = "htmlcov"
```

### 15.3 hypothesis (Property-based Testing)

**Installation:**
```bash
pip install hypothesis
```

**Beispiel für numerischen Code:**
```python
from hypothesis import given, strategies as st
import numpy as np

@given(st.integers(min_value=1, max_value=1000))
def test_my_algorithm(n):
    result = my_algorithm(n)
    assert result >= 0
    assert isinstance(result, (int, float))
```

---

## 16. DOCUMENTATION

### 16.1 sphinx

**Installation:**
```bash
pip install sphinx sphinx-rtd-theme
```

**Konfiguration (conf.py):**
```python
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = 'sphinx_rtd_theme'
```

**Befehle:**
```bash
sphinx-quickstart docs
sphinx-apidoc -o docs/api src/
sphinx-build -b html docs docs/_build
```

### 16.2 mkdocs

**Installation:**
```bash
pip install mkdocs mkdocs-material
```

**mkdocs.yml:**
```yaml
site_name: My Project
theme:
  name: material
  palette:
    primary: indigo

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy

nav:
  - Home: index.md
  - API Reference: api.md
```

---

## ULTIMATE PYPROJECT.TOML

```toml
[build-system]
requires = ["setuptools>=65", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "myproject"
version = "0.1.0"
description = "Scientific Python project with DEAP"
readme = "README.md"
requires-python = ">=3.9"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "numpy>=1.24",
    "deap>=1.4",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.8",
    "mypy>=1.18",
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pre-commit>=3.5",
    "interrogate>=1.7",
    "bandit[toml]>=1.8",
    "pip-audit>=2.7",
    "radon>=6.0",
    "vulture>=2.3",
]
profiling = [
    "py-spy>=0.3",
    "scalene>=1.5",
    "memray>=1.10",
    "line-profiler>=4.1",
]
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
]

# ===== RUFF =====
[tool.ruff]
line-length = 100
target-version = "py39"
src = ["src", "tests"]

exclude = [
    ".git",
    ".ruff_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
    "*.egg-info",
]

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "NPY",    # NumPy-specific
    "RUF",    # Ruff-specific
]

ignore = [
    "E501",    # Line too long
    "E731",    # Lambda assignments
]

fixable = ["ALL"]
unfixable = []

dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]
"tests/**" = ["S101", "ARG", "PLR2004"]
"scripts/**" = ["T20"]

[tool.ruff.lint.isort]
known-first-party = ["myproject"]
section-order = ["future", "standard-library", "third-party", "first-party", "local-folder"]
combine-as-imports = true

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
docstring-code-format = true
docstring-code-line-length = 88

# ===== MYPY =====
[tool.mypy]
python_version = "3.9"
files = ["src", "tests"]
plugins = ["numpy.typing.mypy_plugin"]

disallow_any_generics = true
disallow_untyped_calls = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
strict_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
warn_unreachable = true
strict_equality = true
no_implicit_reexport = true

show_error_codes = true
show_column_numbers = true
pretty = true

cache_dir = ".mypy_cache"

[[tool.mypy.overrides]]
module = ["tests.*"]
disallow_untyped_defs = false

[[tool.mypy.overrides]]
module = ["scipy.*", "matplotlib.*", "deap.*"]
ignore_missing_imports = true

# ===== PYTEST =====
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
addopts = [
    "-ra",
    "--strict-markers",
    "--cov=src",
    "--cov-report=html",
    "--cov-report=term-missing:skip-covered",
    "--cov-fail-under=80",
]

# ===== COVERAGE =====
[tool.coverage.run]
source = ["src"]
branch = true
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
precision = 2
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]

[tool.coverage.html]
directory = "htmlcov"

# ===== INTERROGATE =====
[tool.interrogate]
fail-under = 80
verbose = 2
color = true
style = "numpy"
ignore-magic = true
ignore-private = true
exclude = ["tests", "docs", "build"]

# ===== BANDIT =====
[tool.bandit]
exclude_dirs = ["tests", ".venv", "build"]
skips = ["B101"]

# ===== VULTURE =====
[tool.vulture]
exclude = ["*test*.py", ".venv/"]
ignore_decorators = ["@app.route"]
min_confidence = 80

# ===== RADON =====
[tool.radon]
exclude = "tests/*,docs/*"
show_complexity = true
average = true
```

---

## MAKEFILE / JUSTFILE

### Makefile

```makefile
.PHONY: help format lint test coverage clean install dev

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Install dev dependencies"
	@echo "  make format      - Format code with ruff"
	@echo "  make lint        - Run all linters"
	@echo "  make test        - Run tests"
	@echo "  make coverage    - Generate coverage report"
	@echo "  make security    - Run security checks"
	@echo "  make complexity  - Check code complexity"
	@echo "  make docs        - Generate documentation"
	@echo "  make clean       - Remove artifacts"
	@echo "  make all         - Run format, lint, test"

install:
	pip install -e .

dev:
	pip install -e ".[dev,profiling,docs]"
	pre-commit install

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	mypy src/

test:
	pytest

coverage:
	pytest --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report: htmlcov/index.html"

security:
	bandit -r src/ -c pyproject.toml
	pip-audit

complexity:
	radon cc -nc src/
	radon mi src/

deadcode:
	vulture --min-confidence 80 src/

docstring-coverage:
	interrogate src/

docs:
	sphinx-build -b html docs docs/_build

profile:
	@echo "Use: scalene your_script.py"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete

all: format lint test

ci: lint test security
```

### justfile (Alternative)

```justfile
# justfile - Modern Alternative zu Make

# List available commands
default:
    @just --list

# Install dependencies
install:
    pip install -e .

# Install development dependencies
dev:
    pip install -e ".[dev,profiling,docs]"
    pre-commit install

# Format code
format:
    ruff format .
    ruff check --fix .

# Run linters
lint:
    ruff check .
    mypy src/

# Run tests
test:
    pytest

# Generate coverage report
coverage:
    pytest --cov=src --cov-report=html --cov-report=term
    @echo "Coverage report: htmlcov/index.html"

# Run security checks
security:
    bandit -r src/ -c pyproject.toml
    pip-audit

# Check code complexity
complexity:
    radon cc -nc src/
    radon mi src/

# Find dead code
deadcode:
    vulture --min-confidence 80 src/

# Check docstring coverage
docstring-coverage:
    interrogate src/

# Generate documentation
docs:
    sphinx-build -b html docs docs/_build

# Clean build artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

# Run all checks
all: format lint test

# CI/CD checks
ci: lint test security
```

---

## WORKFLOW-EMPFEHLUNG

### Täglicher Development Workflow

```bash
# 1. Morgens: Projekt-Setup
make dev  # oder: just dev

# 2. Während der Entwicklung (automatisch via pre-commit)
# - Bei jedem commit laufen automatisch:
#   - ruff format (Formatierung)
#   - ruff check --fix (Linting mit Auto-Fix)
#   - mypy (Type Checking)
#   - bandit (Security)

# 3. Vor größeren Commits: Manuelle Checks
make lint
make test
make docstring-coverage

# 4. Vor Push: Vollständige Prüfung
make all

# 5. Bei Performance-Problemen
scalene your_script.py
# oder
py-spy record -o profile.svg -- python your_script.py
```

### Wöchentlicher Workflow

```bash
# Security Audit
make security

# Complexity Review
make complexity

# Dead Code Check
make deadcode

# Documentation Update
make docs
```

### CI/CD Pipeline Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Format check
        run: ruff format --check .
      
      - name: Lint
        run: ruff check .
      
      - name: Type check
        run: mypy src/
      
      - name: Security
        run: |
          bandit -r src/ -c pyproject.toml
          pip-audit
      
      - name: Tests
        run: pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

## EMPFOHLENE TOOL-KOMBINATIONEN

### Minimal-Stack (Schnellstart)

```bash
pip install ruff mypy pytest pre-commit
```

**Tools:** Ruff (Linting + Formatting), mypy (Type Checking), pytest (Testing)

### Standard-Stack (Empfohlen für wissenschaftliche Projekte)

```bash
pip install ruff mypy pytest pytest-cov pre-commit \
    interrogate bandit pip-audit
```

**Tools:** Ruff, mypy, pytest, coverage, interrogate, bandit, pip-audit

### Enterprise-Stack (Maximum Quality)

```bash
pip install ruff mypy pytest pytest-cov hypothesis pre-commit \
    interrogate bandit pip-audit radon vulture \
    py-spy scalene sphinx
```

**Alle Tools:** Linting, Type Checking, Testing, Coverage, Documentation, Security, Complexity, Profiling

---

## SPEZIELLE EMPFEHLUNGEN FÜR IHR PROJEKT

### Für DEAP (Genetische Programmierung) + Numerische Berechnungen

**Optimale Konfiguration:**

```toml
[tool.ruff]
line-length = 100  # Längere Zeilen für mathematische Formeln

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "NPY", "RUF"]
ignore = [
    "E501",    # Line too long
    "E731",    # Lambda assignments (in DEAP üblich)
]

[tool.ruff.lint.per-file-ignores]
"**/genetic_*.py" = ["C901"]  # Komplexität in GA-Code OK

[tool.mypy]
plugins = ["numpy.typing.mypy_plugin"]

[[tool.mypy.overrides]]
module = ["deap.*"]
ignore_missing_imports = true

[tool.interrogate]
fail-under = 75  # Etwas niedrigerer Threshold für Forschungscode
```

**Empfohlene Tools:**
1. **Ruff** - Linting + Formatting
2. **mypy** - Type Checking (mit NumPy Plugin)
3. **pytest** + **hypothesis** - Testing (property-based für GA)
4. **scalene** - Profiling (CPU + Memory für numerische Berechnungen)
5. **interrogate** - Docstring Coverage
6. **bandit** + **pip-audit** - Security

**Workflow:**
```bash
# Setup
pip install -e ".[dev]"
pre-commit install

# Development
# (pre-commit läuft automatisch)

# Vor wichtigen Commits
make lint test

# Performance-Optimierung
scalene genetic_algorithm.py
```

---

## HÄUFIGE PROBLEME UND LÖSUNGEN

### Problem: Ruff vs. Black Konflikte

**Lösung:** Nutze entweder Ruff Format ODER Black, nicht beide.

```bash
# Option 1: Nur Ruff
ruff format .

# Option 2: Nur Black
black .
```

### Problem: mypy findet numpy nicht

**Lösung:** NumPy Plugin aktivieren

```toml
[tool.mypy]
plugins = ["numpy.typing.mypy_plugin"]
```

### Problem: Zu viele False Positives

**Lösung:** Per-File Ignores nutzen

```toml
[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "ARG", "PLR2004"]
"scripts/**" = ["T20"]
```

### Problem: Pre-commit zu langsam

**Lösung:** Nur wichtigste Hooks aktivieren

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
```

---

## ZUSAMMENFASSUNG UND QUICKSTART

### 5-Minuten Setup für Ihr Projekt

```bash
# 1. Dependencies installieren
pip install ruff mypy pytest pytest-cov pre-commit interrogate bandit pip-audit

# 2. pyproject.toml erstellen (siehe "Ultimate pyproject.toml" oben)

# 3. .pre-commit-config.yaml erstellen (siehe oben)

# 4. Pre-commit installieren
pre-commit install

# 5. Ersten Run
pre-commit run --all-files

# 6. Makefile kopieren (siehe oben)

# Fertig! Ab jetzt automatische Checks bei jedem commit
```

### Die wichtigsten Befehle

```bash
# Formatierung + Linting
ruff format . && ruff check --fix .

# Type Checking
mypy src/

# Tests mit Coverage
pytest --cov=src

# Security Checks
bandit -r src/
pip-audit

# Alles zusammen
make all  # oder: just all
```

### Key Takeaways

1. **Ruff ist game-changer** - Ersetzt 8+ Tools, 10-100x schneller
2. **mypy ist essential** - Type Safety für wissenschaftlichen Code
3. **pre-commit automatisiert alles** - Setup einmal, läuft immer
4. **pytest + coverage** - Minimum 80% Coverage anstreben
5. **interrogate** - Docstring Coverage messen (wichtig für wissenschaftlichen Code)
6. **Profiling nur bei Bedarf** - scalene für numerische Berechnungen
7. **poetry oder pip-tools** - Dependency Management vereinfachen

---

## TOOL-VERSIONEN (Stand Januar 2025)

- **ruff:** 0.8.4+
- **mypy:** 1.18.0+
- **pytest:** 7.4+
- **black:** 24.10+ (falls nicht Ruff)
- **bandit:** 1.8.6+
- **pip-audit:** 2.7+
- **interrogate:** 1.7+
- **pre-commit:** 3.5+
- **poetry:** 1.7+
- **scalene:** 1.5+

Alle Tools sind aktiv maintained und production-ready für Python 3.9-3.13.








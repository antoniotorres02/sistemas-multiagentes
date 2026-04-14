# sistemas multiagentes

Repositorio público y reducido que contiene solo la demo didáctica de LangGraph.

La pieza principal es la CLI de `DEMO/`, pensada para mostrar una organización multiagente observable con:

`load_workspace -> research -> curate -> verify -> write -> review -> (write | render)`

## Qué incluye
- Código de la demo en `DEMO/`
- Corpus local en `DEMO/corpus/news_corpus.json`
- Una prueba mínima en `tests/test_demo.py`
- Configuración de empaquetado y ejecución con `pyproject.toml`

## Qué no incluye
- Frontend
- Backend web
- Scraping en vivo
- El sistema completo de `news_system`

## Requisitos
- Python 3.12
- `OPENROUTER_KEY` en un archivo `.env` en la raíz del repo
- `DEMO_OPENROUTER_MODEL` opcional si quieres fijar el modelo de la demo

## Instalación
Con `uv`:

```bash
uv sync --dev
```

Con `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuración
```bash
cp .env.example .env
```

Edita `.env` y añade:

```bash
OPENROUTER_KEY=tu_clave
DEMO_OPENROUTER_MODEL=minimax/minimax-m2.7
```

## Ejecución
```bash
uv run sma-demo run --topic "ai regulation europe"
```

Si prefieres invocarlo como módulo:

```bash
uv run python -m DEMO run --topic "ai regulation europe"
```

## Inspección
La demo genera por cada ejecución:
- `DEMO/runs/<thread_id>/events.jsonl`
- `DEMO/runs/<thread_id>/graph.mmd`
- `DEMO/runs/<thread_id>/report.md`
- `DEMO/runs/<thread_id>/state_history.json`

Y guarda checkpoints en:
- `DEMO/data/checkpoints.sqlite3`

Comandos útiles:

```bash
uv run sma-demo show-history --thread-id <thread_id>
uv run sma-demo show-state --thread-id <thread_id>
uv run sma-demo show-trace --thread-id <thread_id>
uv run sma-demo replay --thread-id <thread_id> --checkpoint-index 0
```

## Prueba
```bash
uv run pytest
```

# sistemas multiagentes

Repositorio público y reducido que contiene la demo didáctica de LangGraph en `news_system_demo/`.

La pieza principal es la CLI de `news_system_demo/`, pensada para mostrar una organización multiagente observable con:

`load_workspace -> research -> curate -> verify -> write -> review -> (write | render)`

## Qué incluye
- Código de la demo en `news_system_demo/`
- Corpus local en `news_system_demo/corpus/news_corpus.json`
- Una prueba mínima en `tests/test_demo.py`
- Configuración de empaquetado y ejecución con `pyproject.toml`

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
uv run python -m news_system_demo run --topic "ai regulation europe"
```

## Inspección
La demo genera por cada ejecución:
- `news_system_demo/runs/<thread_id>/events.jsonl`
- `news_system_demo/runs/<thread_id>/graph.mmd`
- `news_system_demo/runs/<thread_id>/report.md`
- `news_system_demo/runs/<thread_id>/state_history.json`

Y guarda checkpoints en:
- `news_system_demo/data/checkpoints.sqlite3`

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

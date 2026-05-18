# sistemas-multiagentes

Demo pequeña de un sistema multiagente con LangGraph para producir una noticia a partir de un corpus local.

La demo prioriza que el flujo sea fácil de enseñar: cada nodo escribe una entrada breve en `run.log` y la salida principal queda en `report.md`.

## Instalación

```bash
uv sync
```

Crea un `.env` con tu clave de OpenRouter:

```bash
OPENROUTER_KEY=...
DEMO_OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```

## Uso

```bash
uv run sma-demo --topic "ai regulation europe" --thread-id demo-intro
```

Cada ejecución crea:

- `news_system_demo/runs/<thread_id>/report.md`
- `news_system_demo/runs/<thread_id>/run.log`

## Flujo

El grafo mantiene pasos separados para que la coordinación sea visible sin añadir una capa de tracing pesada:

1. `load_workspace`: inicializa la ejecución.
2. `research`: selecciona noticias del corpus local.
3. `curate`: reduce la evidencia a una selección manejable.
4. `verify`: resume qué permite afirmar la evidencia.
5. `write`: redacta la noticia en Markdown.
6. `review`: revisa la noticia y puede pedir una reescritura.
7. `render`: guarda `report.md`.

Si el LLM falla o devuelve JSON inválido, la demo aborta con un error explícito y deja visible el `run.log` generado hasta ese punto.

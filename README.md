# sistemas-multiagentes

Demo pequeña de un sistema multiagente con LangGraph para producir una noticia a partir de un corpus local.

La demo prioriza que el flujo sea fácil de enseñar: cada nodo escribe una entrada breve en `run.log` y la salida principal queda en `report.md`. El corpus es offline y sintético, pero simula una mesa editorial con varias fuentes, temas, ángulos y puntuación de relevancia.

## Instalación

```bash
uv sync
```

Crea un `.env` con tu clave de OpenRouter:

```bash
OPENROUTER_API_KEY=...
DEMO_OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```

`OPENROUTER_KEY` sigue aceptándose como alias legacy, pero la integración
LangChain usa `OPENROUTER_API_KEY` como variable preferida.

## Uso

```bash
uv run sma-demo --topic "ai regulation europe" --thread-id demo-intro
```

Otros temas útiles para enseñar la selección editorial:

```bash
uv run sma-demo --topic "ransomware en ayuntamientos"
uv run sma-demo --topic "red electrica y almacenamiento en España"
uv run sma-demo --topic "privacidad e identidad digital europea"
uv run sma-demo --topic "plataformas, recomendaciones y desinformacion"
uv run sma-demo --topic "IA en hospitales publicos"
```

Cada ejecución crea:

- `news_system_demo/runs/<thread_id>/report.md`
- `news_system_demo/runs/<thread_id>/run.log`

## Flujo

El grafo mantiene pasos separados para que la coordinación sea visible sin añadir una capa de tracing pesada:

1. `load_workspace`: inicializa la ejecución.
2. `research`: puntúa noticias del corpus local y explica por qué encajan con el tema.
3. `curate`: reduce la evidencia a una selección manejable con pieza principal, contexto y contraste.
4. `verify`: resume qué permite afirmar la evidencia.
5. `write`: redacta la noticia en Markdown.
6. `review`: revisa la noticia y puede pedir una reescritura.
7. `render`: guarda `report.md`.

Si el LLM falla o devuelve JSON inválido, la demo aborta con un error explícito y deja visible el `run.log` generado hasta ese punto.

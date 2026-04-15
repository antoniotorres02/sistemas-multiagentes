# news_system_demo

Microapp didáctica por CLI para explicar LangGraph como organización multiagente observable, no como producto de noticias completo.

Grafo principal:

`load_workspace -> research -> curate -> verify -> write -> review -> (write | render)`

## Objetivo
- Mostrar roles diferenciados con un estado compartido pequeño.
- Hacer visibles nodos, aristas, handoffs, checkpoints y artefactos.
- Mantener el montaje ligero con corpus local y ejecución por CLI.

## Requisitos
- Estar en la raíz del repositorio.
- `.venv` activa.
- `OPENROUTER_KEY` disponible en `.env`.
- `DEMO_OPENROUTER_MODEL` opcional si quieres desacoplar la demo del modelo por defecto.

## Ejecución
```bash
source .venv/bin/activate
python -m news_system_demo run --topic "ai regulation europe"
DEMO_OPENROUTER_MODEL=minimax/minimax-m2.7 python -m news_system_demo run --topic "ai regulation europe"
```

## Comandos
```bash
python -m news_system_demo run --topic "ai regulation europe"
python -m news_system_demo show-history --thread-id <thread_id>
python -m news_system_demo show-state --thread-id <thread_id>
python -m news_system_demo show-trace --thread-id <thread_id>
python -m news_system_demo replay --thread-id <thread_id> --checkpoint-index 0
```

## Qué enseña cada comando
- `run`: ejecuta el grafo completo y muestra nodos, aristas y handoffs en tiempo real.
- `show-history`: enseña la secuencia de checkpoints de LangGraph.
- `show-state`: imprime el estado compartido en el checkpoint más reciente o en uno concreto.
- `show-trace`: lee `events.jsonl` y deja ver la interacción entre agentes sin reejecutar la run.
- `replay`: relee la historia persistida desde un checkpoint para estudiar la evolución del estado.

## Artefactos
Cada run crea `news_system_demo/runs/<thread_id>/` con:
- `events.jsonl`
- `graph.mmd`
- `report.md`
- `state_history.json`

Los checkpoints de LangGraph se guardan en:
- `news_system_demo/data/checkpoints.sqlite3`

## Perspectiva SMA
- `research` y `curate` hacen trabajo determinista sobre el entorno local.
- `verify`, `write` y `review` concentran el juicio semántico con LLM.
- `review` fuerza una única vuelta didáctica de realimentación para que la arista condicional quede visible.
- `handoffs` materializa qué entrega cada agente al siguiente.

La demo prioriza coordinación, memoria de trabajo y trazabilidad.

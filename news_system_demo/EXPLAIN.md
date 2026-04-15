# Cómo leer `news_system_demo`

## 1. `load_workspace`
- Introduce en el estado la configuración mínima del flujo.
- Enseña que la organización prepara contexto y reglas antes de actuar.

## 2. `research`
- Filtra el corpus local y normaliza evidencia útil.
- Enseña percepción del entorno sin ruido de red ni scraping.

## 3. `curate`
- Agrupa artículos en historias observables.
- Enseña que un agente puede reorganizar el estado, no solo añadir texto.

## 4. `verify`
- Usa LLM real para producir conclusión, claims y cautelas por historia.
- Enseña dónde aparece el juicio cognitivo dentro de un flujo acotado.

## 5. `write`
- Usa LLM real para redactar un borrador corto y trazable.
- Enseña cómo un agente consume verificación y feedback previo.

## 6. `review`
- Evalúa el borrador y decide si la arista condicional vuelve a `write` o sigue a `render`.
- Enseña coordinación explícita y una única vuelta de realimentación observable.

## 7. `render`
- Saca la información fuera del grafo y genera artefactos.
- Enseña la diferencia entre estado interno y efectos externos.

## Handoffs
- Cada nodo deja un mensaje explícito hacia el siguiente agente.
- El handoff indica propósito, inputs usados, outputs escritos y resumen.
- Esto permite leer la coordinación como comunicación entre agentes, no solo como funciones encadenadas.

## Checkpoints
- LangGraph guarda snapshots del estado por `thread_id`.
- `show-history` permite ver la secuencia.
- `show-state` permite inspeccionar un snapshot concreto.
- `replay` vuelve a recorrer la historia guardada sin reejecutar el flujo.

## Lectura para el trabajo
La demo está pensada para respaldar cuatro ideas:
- LangGraph no es solo un encadenador de funciones: hace explícita la coordinación.
- El estado compartido actúa como memoria de trabajo de la organización.
- Las aristas son decisiones observables, no transiciones implícitas.
- La trazabilidad mejora cuando cada nodo, cada arista y cada handoff quedan persistidos.

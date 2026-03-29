# TODO — Play: Entrenamiento de Tablero y Atril

> **Visión:** Herramienta de entrenamiento para jugadores de Scrabble en español que presenta situaciones diversas de tablero y atril. El usuario analiza la posición y evalúa jugadas considerando tres criterios: máxima puntuación, mejor residuo de atril (rack leave), y el mejor compromiso entre ambos.
>
> Este módulo consolida `analyze_board.py` y `autoplay_scrabble.py` en una estructura organizada bajo `play/`.

---

## Estado Actual

Los componentes ya existen pero están dispersos en la raíz del paquete:

| Archivo actual | Funcionalidad |
|---------------|---------------|
| `analyze_board.py` (~1100 líneas) | OCR de tablero, generación de jugadas (Appel-Jacobson), puntuación, notación |
| `autoplay_scrabble.py` | Simulación solitaria, manejo de bolsa/atril, renderizado de tablero |
| `lexicon.py` | Trie compartido (ya extraído, se queda en su lugar) |

**Dependencias actuales:**
- `duplicate/engine.py` importa de ambos archivos (find_best_moves, Move, create_bag, apply_move, etc.)
- `duplicate/server.py` importa `to_display`, `VOWELS`
- `duplicate/ui.py` importa `to_display`, `print_board_text`

---

## Milestone A — Reorganización en `play/`

Mover el código existente a una estructura modular sin romper funcionalidad.

```
scrabble/play/
├── __init__.py           # Re-exporta API pública para compatibilidad
├── board.py              # OCR + análisis de imagen del tablero
├── moves.py              # Generación de jugadas (Appel-Jacobson) + cross-checks
├── scoring.py            # Cálculo de puntuación (premium squares, bingo, cross-words)
├── rack.py               # Manejo de atril y bolsa (crear, robar, contar, residuo)
├── game.py               # Bucle de juego solitario (autoplay) + apply_move
├── notation.py           # Helpers de notación (row_letter, col_number, pos_notation, to_internal, to_display)
├── render.py             # Renderizado de tablero (texto + imagen PNG)
├── cli.py                # Puntos de entrada CLI (analizar tablero, autoplay)
├── README.md
├── INSTRUCCIONES.md
└── TODO.md               # (este archivo)
```

| # | Tarea | Esfuerzo | Estado |
|---|-------|----------|--------|
| 1 | Crear estructura de directorios y `__init__.py` | Pequeño | |
| 2 | Extraer `notation.py`: to_internal, to_display, tile_points, row_letter, col_number, pos_notation | Pequeño | |
| 3 | Extraer `scoring.py`: score_move, premium squares logic | Pequeño | |
| 4 | Extraer `moves.py`: Move dataclass, find_best_moves, compute_cross_checks, transpose_board | Medio | |
| 5 | Extraer `board.py`: OCR functions (find_board_bounds, extract_cells, cell_has_tile, recognize_letter, read_board_image, read_rack_image) | Medio | |
| 6 | Extraer `rack.py`: parse_rack_text, compute_remaining_tiles, create_bag, draw_tiles, count_rack_composition, VOWELS, RACK_SIZE | Pequeño | |
| 7 | Extraer `game.py`: play_game, apply_move, remove_from_rack, should_stop, create_empty_board | Medio | |
| 8 | Extraer `render.py`: print_board_text, print_board_summary, render_board_image, print_rack, print_remaining, print_moves | Pequeño | |
| 9 | Crear `cli.py`: entry points para `python -m play.cli analyze` y `python -m play.cli autoplay` | Pequeño | |
| 10 | Crear `__init__.py` con re-exports para compatibilidad | Pequeño | |
| 11 | Actualizar imports en `duplicate/engine.py`, `duplicate/server.py`, `duplicate/ui.py` | Medio | |
| 12 | Dejar `analyze_board.py` y `autoplay_scrabble.py` como wrappers de compatibilidad (importan y re-exportan de play/) | Pequeño | |
| 13 | Tests: verificar que analyze_board CLI, autoplay CLI y duplicate (CLI + web) funcionan sin cambios | Medio | |

---

## Milestone B — Entrenamiento Interactivo (CLI)

Herramienta de entrenamiento donde el usuario practica encontrar las mejores jugadas.

### B.1 Generador de Posiciones

| # | Tarea | Esfuerzo | Descripción |
|---|-------|----------|-------------|
| 14 | Generar posiciones aleatorias de entrenamiento | Medio | Simular N rondas de autoplay, guardar tablero + atril en cada punto como "posición de entrenamiento". Filtrar posiciones interesantes (múltiples jugadas de alto puntaje, opciones de rack leave) |
| 15 | Banco de posiciones predefinidas | Medio | Curar posiciones manualmente o desde partidas reales. Formato JSON: tablero (15x15), atril, jugadas válidas con puntajes y rack leave |
| 16 | Posiciones desde imágenes | Pequeño | Reutilizar OCR de board.py para crear posiciones desde fotos de tableros reales |

### B.2 Análisis de Jugadas

| # | Tarea | Esfuerzo | Descripción |
|---|-------|----------|-------------|
| 17 | Calcular rack leave para cada jugada | Medio | Después de cada jugada posible, calcular las fichas restantes en el atril. Usar `synergy.csv` para evaluar la calidad del residuo |
| 18 | Puntuación compuesta (score + rack leave) | Medio | Combinar puntaje de jugada con calidad del residuo. Pesos configurables. Identificar la jugada "óptima" vs la de mayor puntaje |
| 19 | Top-N jugadas con análisis | Pequeño | Para cada posición, mostrar las N mejores jugadas con: puntaje, residuo, sinergia del residuo, puntuación compuesta |

### B.3 Modo de Entrenamiento CLI

| # | Tarea | Esfuerzo | Descripción |
|---|-------|----------|-------------|
| 20 | Mostrar posición al usuario | Pequeño | Tablero + atril en terminal. El usuario ingresa su jugada (PALABRA POSICIÓN) |
| 21 | Evaluar jugada del usuario | Medio | Comparar con la lista de jugadas válidas. Mostrar: rank de la jugada, puntaje, diferencia vs la mejor, rack leave |
| 22 | Revelar mejores jugadas | Pequeño | Después de la respuesta, mostrar las top-5 jugadas con análisis completo |
| 23 | Modo aleatorio vs secuencial | Pequeño | Elegir posiciones al azar del banco o seguir una secuencia didáctica (fácil → difícil) |
| 24 | Seguimiento de progreso | Pequeño | Registrar aciertos, tendencias (¿favorece puntaje sobre rack leave?), mejora en el tiempo |

---

## Milestone C — Entrenamiento Web

Llevar el modo de entrenamiento a la interfaz web.

| # | Tarea | Esfuerzo | Descripción |
|---|-------|----------|-------------|
| 25 | Backend API: posición aleatoria, evaluar jugada, revelar | Medio | Endpoints REST para el flujo de entrenamiento |
| 26 | Frontend: tablero interactivo 15x15 | Grande | Renderizar tablero con casillas premium, fichas colocadas, atril del usuario. CSS grid o canvas |
| 27 | Frontend: input de jugada | Medio | El usuario selecciona posición en el tablero (click) y escribe la palabra. Validación en tiempo real |
| 28 | Frontend: análisis visual | Medio | Después de evaluar, mostrar las mejores jugadas con colores: verde (top-1), amarillo (top-5), gris (otras). Destacar rack leave |
| 29 | Integrar con módulo de estudio | Medio | Palabras que el usuario no conoce durante el entrenamiento se agregan automáticamente al sistema SRS de estudio |
| 30 | Montar en `/jugar/` junto a `/estudio/` y `/duplicada/` | Pequeño | Servidor unificado con las tres herramientas |

---

## Milestone D — Funcionalidades Avanzadas

| # | Tarea | Esfuerzo | Descripción |
|---|-------|----------|-------------|
| 31 | Análisis de posiciones desde fotos en la web | Grande | Subir foto del tablero → OCR → posición de entrenamiento. Reutilizar board.py |
| 32 | Modo "replay" de partidas | Medio | Cargar partida completa (CSV de autoplay o duplicada), navegar ronda por ronda, analizar cada jugada |
| 33 | Simulación de oponente | Grande | Dado un tablero y atril, simular qué haría un oponente de nivel X. Útil para practicar defensa |
| 34 | Endgame trainer | Grande | Posiciones de final de partida donde se conocen las fichas del oponente. Encontrar la secuencia óptima |
| 35 | Análisis probabilístico de rack leave | Grande | Dado un residuo, calcular la probabilidad de robar fichas útiles de la bolsa restante. Integrar con scoring compuesto |

---

## Dependencias entre Módulos

```
lexicon.py (trie compartido)
    │
    ├── play/ (generación de jugadas, scoring, tablero)
    │     │
    │     └── duplicate/ (usa play/ para el motor de juego)
    │
    └── study/ (validación, transformaciones, quiz)
```

La reorganización no debe romper ningún import existente. Los archivos `analyze_board.py` y `autoplay_scrabble.py` en la raíz se mantienen temporalmente como wrappers de compatibilidad que re-exportan desde `play/`.

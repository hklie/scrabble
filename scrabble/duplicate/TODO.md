# TODO — Scrabble Duplicada

> Mejoras y funcionalidades pendientes para el módulo de Scrabble Duplicada.
> Toda la interfaz de usuario debe estar en **español**.

---

## Estado Actual ✅

- Modo CLI funcional con registro de jugadores, temporizador, validación de jugadas
- Modo web funcional con FastAPI + WebSocket (host + jugadores)
- Anti-trampa: detección de cambio de pestaña, pantalla completa obligatoria, Wake Lock
- Exportación multi-formato: CSV, Excel, HTML, gráfico PNG
- Restricciones configurables de atril (mínimo de vocales/consonantes)
- Generación de jugadas con Appel-Jacobson + trie FISE2

---

## Mejoras de Interfaz Web

| # | Tarea | Esfuerzo | Estado |
|---|-------|----------|--------|
| 1 | Mejorar diseño responsive para tabletas/pantallas grandes | Medio | |
| 2 | Añadir código QR en pantalla del anfitrión para que jugadores se unan fácilmente | Pequeño | |
| 3 | Mostrar composición del atril en tiempo real (vocales/consonantes/comodines) | Pequeño | |
| 4 | Historial de jugadas de rondas anteriores visible para el anfitrión | Medio | |
| 5 | Sonido/vibración al finalizar temporizador | Pequeño | |
| 6 | Animaciones de transición entre rondas | Pequeño | |

## Mejoras de Juego

| # | Tarea | Esfuerzo | Estado |
|---|-------|----------|--------|
| 7 | Modo práctica individual (jugar contra el Maestro sin otros jugadores) | Medio | |
| 8 | Opción de mostrar las N mejores jugadas después de cada ronda (no solo la maestra) | Pequeño | |
| 9 | Estadísticas post-partida detalladas: promedio por ronda, mejor/peor ronda, racha | Medio | |
| 10 | Replay de partida: revisar tablero ronda por ronda después de terminar | Medio | |
| 11 | Modo torneo: múltiples partidas con clasificación acumulativa | Grande | |
| 12 | Guardar/cargar estado de partida para continuar después | Medio | |

## Exportación y Datos

| # | Tarea | Esfuerzo | Estado |
|---|-------|----------|--------|
| 13 | Exportar jugadas de cada jugador por ronda (no solo puntajes) | Pequeño | |
| 14 | Exportar tablero final como imagen | Medio | |
| 15 | Integrar estadísticas de partidas con el sistema de estudio (palabras falladas → quiz) | Grande | |

## Infraestructura

| # | Tarea | Esfuerzo | Estado |
|---|-------|----------|--------|
| 16 | Unificar servidor: montar `/duplicada/` y `/estudio/` en una sola app FastAPI | Medio | |
| 17 | Persistencia de partidas en base de datos (para historial y torneos) | Grande | |
| 18 | Autenticación de jugadores (para rankings persistentes) | Grande | |

---

## Prioridades Sugeridas

**Corto plazo** (mejoras rápidas):
- #2 (QR code), #3 (composición atril), #5 (sonido), #8 (top N jugadas), #13 (exportar jugadas)

**Mediano plazo** (valor alto):
- #7 (práctica individual), #9 (estadísticas), #15 (integración con estudio), #16 (servidor unificado)

**Largo plazo** (ambicioso):
- #11 (torneo), #17 (BD), #18 (autenticación)

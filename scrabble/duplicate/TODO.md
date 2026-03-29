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
| 1 | **Tablero + atril + preview de jugada en el celular del jugador** | Grande | |
| 2 | Mejorar diseño responsive para tabletas/pantallas grandes | Medio | |
| 3 | Añadir código QR en pantalla del anfitrión para que jugadores se unan fácilmente | Pequeño | |
| 4 | Mostrar composición del atril en tiempo real (vocales/consonantes/comodines) | Pequeño | |
| 5 | Historial de jugadas de rondas anteriores visible para el anfitrión | Medio | |
| 6 | Sonido/vibración al finalizar temporizador | Pequeño | |
| 7 | Animaciones de transición entre rondas | Pequeño | |

### Detalle de #1: Tablero + atril + preview en el celular del jugador

**Objetivo:** Que los jugadores puedan jugar Duplicada sin necesidad de un tablero físico ni fichas. El celular muestra todo lo necesario: el tablero actual, su atril, y una previsualización de la jugada antes de enviarla.

**Componentes:**
- **Tablero mini en el celular**: Grilla 15x15 responsive con casillas premium (TP, DP, TL, DL) y fichas ya colocadas. Debe ser legible en pantalla de celular (scroll o zoom).
- **Atril interactivo**: Las 7 fichas del atril mostradas como fichas arrastrables o seleccionables.
- **Preview de jugada**: Al escribir `PALABRA POSICION`, el tablero resalta dónde se colocarían las fichas (color distinto) y muestra el puntaje estimado antes de enviar.
- **Confirmación visual**: El jugador ve exactamente cómo quedaría su jugada en el tablero antes de confirmar el envío.

**Implementación:**
1. El servidor ya envía el estado del tablero en `round_start`. Agregar las posiciones de fichas existentes al mensaje WebSocket.
2. Renderizar el tablero en el cliente con CSS grid (similar a host.html pero compacto).
3. Parser de jugada en JS: tokenizar `PALABRA POSICION` → calcular celdas afectadas → resaltar en el tablero.
4. Botón "Enviar" solo se activa cuando la preview es válida (posición dentro del tablero).

---

## Mejoras de Juego

| # | Tarea | Esfuerzo | Estado |
|---|-------|----------|--------|
| 8 | Modo práctica individual (jugar contra el Maestro sin otros jugadores) | Medio | |
| 9 | Opción de mostrar las N mejores jugadas después de cada ronda (no solo la maestra) | Pequeño | |
| 10 | Estadísticas post-partida detalladas: promedio por ronda, mejor/peor ronda, racha | Medio | |
| 11 | Replay de partida: revisar tablero ronda por ronda después de terminar | Medio | |
| 12 | Modo torneo: múltiples partidas con clasificación acumulativa | Grande | |
| 13 | Guardar/cargar estado de partida para continuar después | Medio | |

## Exportación y Datos

| # | Tarea | Esfuerzo | Estado |
|---|-------|----------|--------|
| 14 | Exportar jugadas de cada jugador por ronda (no solo puntajes) | Pequeño | |
| 15 | Exportar tablero final como imagen | Medio | |
| 16 | Integrar estadísticas de partidas con el sistema de estudio (palabras falladas → quiz) | Grande | |

## Infraestructura

| # | Tarea | Esfuerzo | Estado |
|---|-------|----------|--------|
| 17 | Unificar servidor: montar `/duplicada/`, `/estudio/` y `/jugar/` en una sola app FastAPI | Medio | |
| 18 | Persistencia de partidas en base de datos (para historial y torneos) | Grande | |
| 19 | Autenticación de jugadores (para rankings persistentes) | Grande | |

---

## Prioridades Sugeridas

**Corto plazo** (mejoras rápidas):
- #3 (QR code), #4 (composición atril), #6 (sonido), #9 (top N jugadas), #14 (exportar jugadas)

**Mediano plazo** (valor alto):
- **#1 (tablero en celular — elimina necesidad de tablero físico)**, #8 (práctica individual), #10 (estadísticas), #16 (integración con estudio), #17 (servidor unificado)

**Largo plazo** (ambicioso):
- #12 (torneo), #18 (BD), #19 (autenticación)

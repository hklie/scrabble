# Scrabble Duplicada

Juego de Scrabble Duplicada multijugador donde todos los jugadores ven el mismo tablero y atril en cada ronda. Los jugadores compiten por encontrar la mejor jugada. Soporta modo terminal (CLI) y modo web (FastAPI + WebSocket).

## Como Funciona el Scrabble Duplicada

A diferencia del Scrabble estandar, en el modo Duplicada:
- Todos los jugadores reciben el **mismo atril** y ven el **mismo tablero** cada ronda.
- Los jugadores envian su mejor jugada de forma independiente dentro de un limite de tiempo.
- Solo la **mejor jugada posible** (la jugada "Maestra") se coloca en el tablero.
- Los jugadores se clasifican segun que tan cerca esta su puntuacion de la del Maestro.

## Inicio Rapido

### Modo Servidor Web (recomendado)

Los jugadores se unen desde su celular/tablet escaneando un codigo QR o ingresando la URL del servidor.

```bash
# Uso basico
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt

# Con semilla para reproducibilidad y titulo personalizado
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt \
    --seed 42 --title "Campeonato del Club - Partida 1"

# Puerto personalizado
python scrabble/duplicate/server.py scrabble/duplicate/dupli_config.txt --port 9000
```

1. El anfitrion abre `http://localhost:8000/host` en una laptop/proyector.
2. Los jugadores abren `http://<ip-del-host>:8000/play` en sus celulares.
3. Los jugadores ingresan el codigo de sala de 4 digitos y su nombre.
4. El anfitrion hace clic en "Iniciar Ronda" para comenzar cada ronda.

### Modo CLI

Juego basado en terminal donde un moderador ingresa la jugada de cada jugador.

```bash
python scrabble/duplicate/main.py scrabble/duplicate/dupli_config.txt [--seed 42]
```

## Configuracion

Edita `dupli_config.txt` para personalizar el juego:

```
# Titulo del juego (opcional)
title = CAMPEONATO MUNDIAL DE DUPLICADA - Partida 1

# Numero de rondas (0 = ilimitado, jugar hasta agotar la bolsa)
rounds = 15

# Restricciones del atril: (min_vocales_y_consonantes, hasta_ronda)
# Rondas 1-15: al menos 2 vocales Y 2 consonantes en el atril
# Rondas 16-30: al menos 1 de cada una
# Despues de la ronda 30: sin restriccion
constraints = (2,15),(1,30)

# Tiempo por ronda (M:SS o segundos)
time = 3:00

# Formato de exportacion: csv, excel, html, graphical
output = csv
```

| Opcion | Descripcion |
|--------|-------------|
| `rounds` | Numero de rondas. `0` = jugar hasta agotar la bolsa de fichas |
| `constraints` | Pares `(min, hasta_ronda)`. Asegura que los atriles tengan suficientes vocales y consonantes |
| `time` | Cuenta regresiva por ronda. Formato: `M:SS` o segundos |
| `output` | Formato de exportacion de resultados: `csv`, `excel` (.xlsx), `html`, `graphical` (grafico PNG) |
| `title` | Titulo del juego mostrado en la pantalla del anfitrion |

## Formato de Jugada

Los jugadores envian jugadas como `PALABRA POSICION`:
- `CORTES H8` — jugar CORTES comenzando en H8
- Notacion de posicion: `H8` (fila-columna) = horizontal, `8H` (columna-fila) = vertical
- **Fichas en blanco**: usar minuscula para la letra del comodin. `CORTEs H8` significa que la S es una ficha en blanco.

## Interfaz Web

### Vista del Anfitrion (`/host`)
- Tablero 15x15 con casillas premium (TP, DP, TL, DL), estilo tapete verde
- Atril con valor en puntos de cada ficha
- Temporizador (3x tamano), alerta sonora de 3 pitidos a los 30 segundos
- Conteo de jugadores y seguimiento de envios en tiempo real
- Resultados de ronda: jugada Maestra + clasificacion acumulativa (top 20)
- Las jugadas de los jugadores no se muestran para evitar filtrar estrategias
- Resumen al final del juego: jugadas maestras por ronda + top 10 clasificacion final
- Codigo QR para que los jugadores se unan facilmente

### Vista del Jugador (`/play`)
- Pantalla de ingreso: codigo de sala + nombre
- Modo pantalla completa obligatorio (anti-trampa)
- Atril con valor en puntos y boton para mezclar fichas
- Temporizador, entrada de jugada con confirmacion de comodines
- Resultados de ronda: tu puntaje vs Maestro, posicion en clasificacion
- Reconexion automatica en caso de desconexion

### Funciones Anti-Trampa
- Deteccion de cambio de pestana: solo registra salidas de 5+ segundos (ignora notificaciones, barra de estado)
- Periodo de gracia en transiciones de pantalla completa para evitar falsos positivos
- Modo pantalla completa obligatorio en moviles
- API Wake Lock mantiene la pantalla activa
- Conteo agregado de infracciones mostrado en pantalla del anfitrion

## Estructura del Modulo

| Archivo | Descripcion |
|---------|-------------|
| `server.py` | Servidor web FastAPI con endpoints WebSocket para anfitrion y jugadores |
| `engine.py` | Motor del juego: estado del tablero, reparto de fichas, generacion de jugadas, validacion |
| `main.py` | Punto de entrada CLI para juego basado en terminal |
| `players.py` | Registro de jugadores y seguimiento de puntajes |
| `ui.py` | Utilidades de visualizacion en terminal (tablero, atril, temporizador, resultados) |
| `export.py` | Exportacion de resultados en multiples formatos (CSV, Excel, HTML, grafico PNG) |
| `dupli_config.py` | Analizador de archivo de configuracion |
| `dupli_config.txt` | Configuracion de ejemplo |
| `static/host.html` | Interfaz web del anfitrion/moderador |
| `static/player.html` | Interfaz web del jugador (compatible con moviles, capacidad PWA) |

## Detalles del Motor de Juego

### Reparto de Fichas
`draw_tiles_constrained()` asegura un minimo de vocales/consonantes por ronda. Las fichas en blanco (`?`) cuentan para cualquier requisito. Si las restricciones no se pueden cumplir (bolsa muy pequena), se imprime una advertencia y se reparte el mejor atril posible.

### Generacion de Jugadas
Usa el algoritmo de Appel-Jacobson con un trie construido a partir del lexico FISE2 (~639K palabras). Todas las jugadas legales se generan y clasifican por puntaje. La jugada de mayor puntaje se convierte en la jugada Maestra.

### Validacion de Jugadas
`validate_play(palabra, posicion, moves_df)` verifica el envio del jugador contra la lista completa de jugadas. La verificacion de posicion de comodines distingue mayusculas de minusculas: mayuscula = ficha normal, minuscula = comodin.

### Puntuacion
Puntuacion estandar de Scrabble con casillas premium (Doble/Triple Palabra, Doble/Triple Letra) y bono de 50 puntos por usar las 7 fichas del atril (bingo).

## Exportacion

Los resultados se guardan en `duplicate/resultados/` con marca de tiempo:
- **CSV**: UTF-8 con BOM, filas de jugadores ordenadas por puntaje total
- **Excel**: .xlsx formateado con encabezados en negrita, fila del Maestro coloreada, ancho automatico
- **HTML**: Tabla con tema oscuro estilizado
- **Grafico**: Grafico de lineas PNG mostrando la progresion de puntaje acumulado por jugador

## Dependencias

| Paquete | Proposito |
|---------|-----------|
| `fastapi` | Servidor HTTP + WebSocket |
| `uvicorn` | Servidor ASGI |
| `websockets` | Protocolo WebSocket |
| `pandas` | DataFrames de jugadas, exportacion de resultados |
| `openpyxl` | Exportacion Excel |
| `matplotlib` | Exportacion grafica (graficos PNG) |

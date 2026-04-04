# Estudio — Sistema de Estudio y Quiz de Palabras

Conjunto completo de herramientas para el estudio de palabras del Scrabble en espanol: scripts de analisis del lexico que generan listas de palabras categorizadas, mas un sistema interactivo de quiz con repeticion espaciada, 8 modos de estudio, consulta de palabras, 20 mazos preconfigurados y programacion SRS.

## Inicio Rapido

### Quiz Interactivo

```bash
cd scrabble

# Lanzar menu interactivo
python -m study.quiz

# Modo directo con mazo preconfigurado
python -m study.quiz --mode anagram --deck words-5
python -m study.quiz --mode hooks --deck 5L-end-l
python -m study.quiz --mode pattern --deck verbs-7
python -m study.quiz --mode morphology --length 6
python -m study.quiz --mode transformation --deck words-4
python -m study.quiz --mode extension --deck words-3

# Herramientas de transformacion de palabras (CLI independiente)
python -m study.transforms CASA              # Cambios de una letra
python -m study.transforms --all CASA        # Todas las transformaciones

# Ver progreso
python -m study.quiz --stats
python -m study.quiz --list-decks
```

### Regenerar Listas de Palabras

```bash
# Paso 1: Limpiar lexico (requerido primero)
python scrabble/study/clean_no_verbs.py

# Paso 2: Generar CSVs de metadatos
python scrabble/study/nouns_csv.py       # word_analysis.csv (con ganchos de 28 fichas)
python scrabble/study/verbs_csv.py       # verbs.csv

# Paso 3: Generar listas de palabras (todos independientes)
python scrabble/study/probability.py
python scrabble/study/study_list.py
python scrabble/study/vowel_patterns.py
python scrabble/study/endings.py
python scrabble/study/prefixes.py
python scrabble/study/suffixes.py
python scrabble/study/ocurrences.py
python scrabble/study/filter_by_tiers.py
python scrabble/study/unique_anagrams.py
python scrabble/study/endings_with_useful_plurals.py
python scrabble/study/chains.py
python scrabble/study/synergy.py
```

## Sistema de Quiz

### Modos

| Modo | Descripcion |
|------|-------------|
| **Repaso** | Tarjetas de autoevaluacion. Muestra la palabra, revela ganchos/morfologia/anagramas, calificar 0–5. |
| **Anagrama** | Se muestran letras desordenadas, escribir la palabra. 2 intentos antes de revelar. |
| **Ganchos** | Dada una palabra, nombrar las letras que pueden engancharse antes y despues (los 28 tipos de fichas). Puntuacion por completitud. |
| **Patron** | Palabra con ~40% de letras ocultas (se ocultan primero las de mayor valor). Completar la palabra. |
| **Morfologia** | Dada una palabra, identificar su prefijo y sufijo. |
| **Transformacion** | Dada una palabra con una posicion oculta, nombrar letras de reemplazo validas que formen nuevas palabras. Puntuacion por completitud. |
| **Extension** | Dada una palabra con un espacio de insercion en una posicion aleatoria, nombrar letras validas para insertar. Puntuacion por completitud. |
| **Reduccion** | Dada una palabra, identificar que letra(s) se pueden eliminar dejando una palabra valida. Puntuacion por completitud. |

Todos los modos muestran informacion completa despues de cada tarjeta: ganchos, prefijo, sufijo, terminacion, tipo de verbo, anagramas y valor en puntos.

### Mazos Preconfigurados

**Palabras por longitud:**
| Mazo | Palabras | Descripcion |
|------|------:|-------------|
| `words-2` | 81 | Palabras de 2 letras |
| `words-3` | 387 | Palabras de 3 letras |
| `words-4` | 1.966 | Palabras de 4 letras |
| `words-5` | 4.963 | Palabras de 5 letras |

**Patrones vocalicos de 7 letras:**
| Mazo | Palabras | Descripcion |
|------|------:|-------------|
| `7L-2vowels` | 294 | Palabras de 7 letras con solo 2 vocales |
| `7L-2cons` | 135 | Palabras de 7 letras con solo 2 consonantes |

**Probabilidad y puntuacion:**
| Mazo | Palabras | Descripcion |
|------|------:|-------------|
| `high-prob` | 8.828 | Palabras de alta probabilidad (Top10, 4–8 letras) |
| `scoring-5` | 788 | Palabras de 5 letras con letras de alta puntuacion |
| `scoring-6` | 1.260 | Palabras de 6 letras con letras de alta puntuacion |

**Palabras de 5 letras por terminacion:**
| Mazo | Palabras | Descripcion |
|------|------:|-------------|
| `5L-end-d` | 16 | Terminadas en D |
| `5L-end-l` | 322 | Terminadas en L |
| `5L-end-n` | 388 | Terminadas en N |
| `5L-end-r` | 214 | Terminadas en R |
| `5L-end-z` | 71 | Terminadas en Z |

**Verbos por longitud:**
| Mazo | Palabras | Descripcion |
|------|------:|-------------|
| `verbs-3` | 5 | Verbos de 3 letras |
| `verbs-4` | 44 | Verbos de 4 letras |
| `verbs-5` | 484 | Verbos de 5 letras |
| `verbs-6` | 1.203 | Verbos de 6 letras |
| `verbs-7` | 1.945 | Verbos de 7 letras |
| `verbs-8` | 2.600 | Verbos de 8 letras |

### Organizacion del Estudio

- **Consultar palabra** (`c` en el menu): Verificar la validez de cualquier palabra contra el lexico FISE2. Muestra valor en puntos, ganchos, morfologia y conteo de transformaciones (cambios, inserciones, eliminaciones).
- **Estudio por grupo** (`g` en el menu): Navegar y estudiar palabras agrupadas por prefijo, sufijo o terminacion comun. Navegacion paginada para seleccionar grupos.
- **Estudio de verbos** (`v` en el menu): Filtrar verbos por longitud, comienzo, tipo (transitivo/intransitivo/pronominal/antiguo), o navegar comienzos agrupados por los primeros N caracteres.
- **Filtro personalizado**: Especificar longitud, nivel consonantico (1–4), letra final y percentil minimo.

### Repeticion Espaciada (SRS)

El quiz usa el **algoritmo SM-2** para programar repasos:

- **Escala de calidad**: 0 (no recuerdo nada) a 5 (recuerdo facil).
- Palabras calificadas < 3 se reinician a intervalos cortos; palabras calificadas 3+ se espacian en intervalos crecientes.
- Cada sesion: primero se repasan las tarjetas pendientes, luego se introducen nuevas (maximo 10 nuevas por sesion).
- El progreso se guarda en `Data/progress.json`.

Ejecuta `python -m study.quiz --stats` para ver: total de palabras estudiadas, pendientes hoy, dominadas/en aprendizaje/en dificultad, factor de facilidad promedio.

### Opciones de Linea de Comandos

| Opcion | Descripcion |
|--------|-------------|
| `--mode` | Modo de quiz: `review`, `anagram`, `hooks`, `pattern`, `morphology`, `transformation`, `extension`, `reduction` |
| `--deck` | Nombre del mazo preconfigurado (ver `--list-decks`) |
| `--length` | Filtrar por longitud de palabra |
| `--tier` | Filtrar por nivel consonantico (1–4) |
| `--size` | Tamano de sesion (predeterminado: 20) |
| `--min-percentile` | Percentil minimo de probabilidad (predeterminado: P25) |
| `--stats` | Mostrar estadisticas de progreso |
| `--list-decks` | Listar todos los mazos disponibles |

### Formato de Entrada de Ganchos

El quiz de ganchos acepta multiples formatos:
- Separados por espacios: `s z t`
- Separados por comas: `s, z, t`
- Concatenados: `szt`
- Dígrafos y Ñ: `ch ll rr ñ`

## Scripts de Generacion de Listas

### Flujo de Datos

```
No_verbos.txt (crudo, latin-1)
    │
    └─ clean_no_verbs.py ──► No_verbos_filtrados.txt (limpio, UTF-8)
                                │
    ┌───────────────────────────┤
    │                           │
    ▼                           ▼
nouns_csv.py              Todos los scripts de filtro
    │                     (endings, prefixes, suffixes, etc.)
    ▼                           │
word_analysis.csv               ▼
(92K palabras, 56 cols)     archivos *.txt en Data/
```

### Scripts

| Script | Entrada | Salida | Descripcion |
|--------|---------|--------|-------------|
| `clean_no_verbs.py` | `No_verbos.txt` | `No_verbos_filtrados.txt` | Decodificar digrafos, deduplicar, ordenar |
| `nouns_csv.py` | Lexico limpio | `word_analysis.csv` | Metadatos completos: probabilidad, ganchos (28 fichas), prefijo/sufijo, anagramas, percentil |
| `verbs_csv.py` | `Verbos.txt` + clasificaciones | `verbs.csv` | Tipo de verbo (transitivo/intransitivo/pronominal/antiguo) |
| `probability.py` | Lexico limpio | `Ranked_Scrabble_Suggestions.txt` | Ranking combinado de probabilidad NVF + fichas Scrabble |
| `study_list.py` | Lexico limpio | `Optimized_Study_List.txt` | Lista de estudio en 3 niveles con metadatos de prefijo/sufijo/raiz |
| `generator.py` | Entrada interactiva | `<palabra>.txt` | Generar palabras a partir de restricciones de letras |
| `vowel_patterns.py` | Lexico limpio | `vowel_pattern_*.txt` (8 archivos) | Palabras de 7 letras por patron de vocales/consonantes |
| `endings.py` | Lexico limpio | `ends_with_*.txt` (21 archivos) | Palabras agrupadas por letra final |
| `prefixes.py` | Lexico limpio | `prefix_*.txt` (15+ archivos) | Palabras que coinciden con patrones de prefijo |
| `suffixes.py` | Lexico limpio | `suffix_*.txt` (40+ archivos) | Palabras que coinciden con patrones de sufijo, agrupacion por variante vocalica |
| `ocurrences.py` | Lexico limpio | `pattern_*.txt` (6 archivos) | Palabras que contienen patrones internos (VVV, Vh, tl, etc.) |
| `filter_by_tiers.py` | Lexico limpio | `words_only_TIER_*.txt` (4 archivos) | Palabras por nivel de dificultad consonantica |
| `unique_anagrams.py` | Lexico limpio | `singleton_anagrams_*.txt` | Palabras sin anagramas en el lexico |
| `endings_with_useful_plurals.py` | Lexico limpio | `five/six_token_ending_*.txt` (7 archivos) | Palabras de 5 letras terminadas en L/R/S/Z/N, de 6 en U/I |
| `chains.py` | Lexico limpio | `chains_study_list.txt` | Cadenas de transformacion de palabras (distancia de Hamming 1) |
| `synergy.py` | Lexico limpio | `synergy.csv` | Puntuaciones de sinergia de fichas restantes (percentil 0–100) |

### Niveles Consonanticos

| Nivel | Consonantes | Dificultad |
|-------|------------|------------|
| 1 | l, s, r, n, t | Mas comunes |
| 2 | c, g, m, p, b, d | Medias |
| 3 | v, ch, y, q, f, h | Menos comunes |
| 4 | rr, ll, j, x, z, ñ | Raras |

## Interfaz Web

La herramienta de estudio tiene una interfaz web accesible desde cualquier navegador.

```bash
cd scrabble
python -m study.web.server --port 8080
# Abrir http://localhost:8080
```

**Explorador**: Escribe cualquier palabra para verificar validez, ver puntos, ganchos, prefijo/sufijo, y expandir transformaciones/extensiones/reducciones.

**Quiz**: Selecciona un modo (8 disponibles), elige un mazo por categoria, configura el tamano de sesion y estudia con tarjetas estilo ficha de Scrabble. El progreso SRS se guarda entre sesiones.

Todo el texto esta en espanol. Tema oscuro, diseno responsive.

## Estructura del Modulo

| Archivo | Tipo | Descripcion |
|---------|------|-------------|
| `quiz.py` | Aplicacion | Quiz CLI interactivo con 8 modos, consulta de palabras, menus de verbos/grupos |
| `web/server.py` | Web | Backend FastAPI: API REST para explorador, sesiones de quiz, SRS, listado de mazos |
| `web/static/index.html` | Web | Frontend de pagina unica: explorador + quiz + dashboard (JS vanilla) |
| `transforms.py` | Biblioteca + CLI | Transformaciones de palabras: cambio, insercion, eliminacion de una letra (`python -m study.transforms`) |
| `srs.py` | Motor | Algoritmo de repeticion espaciada SM-2 + persistencia JSON |
| `decks.py` | Biblioteca | Carga de tarjetas, filtrado, 20 mazos, agrupacion |
| `nouns_csv.py` | Generador | CSV completo de metadatos (probabilidad, ganchos, morfologia) |
| `verbs_csv.py` | Generador | CSV de clasificacion de verbos |
| `clean_no_verbs.py` | Generador | Limpieza y deduplicacion del lexico |
| `probability.py` | Generador | Ranking probabilistico de palabras |
| `study_list.py` | Generador | Lista de estudio por niveles |
| `generator.py` | Interactivo | Generador de palabras por restriccion de letras |
| `vowel_patterns.py` | Generador | Filtros de patron vocalico para 7 letras |
| `endings.py` | Generador | Filtro por letra final |
| `prefixes.py` | Generador | Filtro por patron de prefijo |
| `suffixes.py` | Generador | Filtro por patron de sufijo (con agrupacion vocalica) |
| `ocurrences.py` | Generador | Filtro por patrones internos de letras |
| `filter_by_tiers.py` | Generador | Filtro por dificultad consonantica |
| `unique_anagrams.py` | Generador | Deteccion de anagramas unicos |
| `endings_with_useful_plurals.py` | Generador | Filtros de longitud + terminacion especificos |
| `chains.py` | Generador | Cadenas de transformacion de palabras |
| `synergy.py` | Generador | Calculo de sinergia de fichas restantes |

# TODO — Estudio: Zyzzyva / Aerolith para Scrabble en Español

> **Vision:** La herramienta definitiva de estudio de palabras para jugadores hispanohablantes de Scrabble — el equivalente de Zyzzyva y Aerolith, diseñada para el léxico FISE2, dígrafos españoles (CH, LL, RR, Ñ) y el alfabeto de 28 fichas. Plataforma web accesible desde cualquier dispositivo.
>
> Toda la interfaz de usuario debe estar en **español**.

---

## Estado Actual — Milestone A ✅ Completado

Todo funciona via CLI (`python -m study.quiz`):

- **8 modos de quiz**: Repaso, Anagrama, Ganchos, Patrón, Morfología, Transformación, Extensión, Reducción
- **Consulta de palabras**: validación contra el léxico FISE2 con ganchos, morfología y transformaciones
- **20 mazos preconfigurados**: por longitud, patrones vocálicos, alta probabilidad, terminaciones, verbos
- **Estudio por grupo**: prefijos, sufijos, terminaciones
- **Estudio de verbos**: por longitud, comienzo, tipo
- **SRS SM-2**: repetición espaciada con persistencia en `progress.json`
- **Módulo de transformaciones**: `lexicon.py` (trie compartido) + `transforms.py` (cambiar/insertar/eliminar)

---

## Milestone B — Web MVP (en progreso)

### Comparación con Zyzzyva y Aerolith

| Funcionalidad | Zyzzyva | Aerolith | Nuestra herramienta |
|---------------|---------|----------|---------------------|
| Validación de palabras | Sí | No | Sí |
| Quiz de anagramas | Sí | Sí | Sí |
| Quiz de ganchos | Sí | No | Sí (28 fichas) |
| Patrón/comodines | Sí | No | Sí |
| Listas por longitud | Sí | Sí | Sí (20 mazos) |
| Repetición espaciada | No | No | **Sí (SM-2 — ventaja única)** |
| Morfología | No | No | **Sí (prefijo/sufijo — único)** |
| Transformaciones | No | No | **Sí (cambiar/insertar/eliminar — único)** |
| Estudio por grupo | No | No | **Sí (único)** |
| Estudio de verbos por tipo | No | No | **Sí (único)** |
| Soporte de dígrafos españoles | No | No | **Sí (CH/LL/RR/Ñ nativos)** |
| Acceso web | No (escritorio) | Sí | Sí |
| Responsive móvil | No | Parcial | Sí (mobile-first) |
| Idioma | Inglés | Inglés | **Español** |

### Prioridades

| # | Tarea | Esfuerzo | Dependencias | Estado |
|---|-------|----------|-------------|--------|
| 5 | Refactorizar lógica de quiz en `quiz_engine.py` (desacoplar de terminal) | Medio | Ninguna | |
| 6 | Backend web: FastAPI REST API + WebSocket para quiz | Medio | Tarea 5 | |
| 7 | Frontend: Explorador de palabras (validar/transformar/extender/reducir) | Medio | Tarea 6 | |
| 8 | Frontend: Página de quiz con 8 modos + selector de mazos + SRS | Grande | Tarea 6 | |
| 9 | Frontend: Dashboard (progreso, pendientes, estadísticas) | Pequeño | Tarea 8 | |

### Fase 1: Backend API

#### 1a. Refactorizar quiz en `study/quiz_engine.py`

Extraer la lógica de cada modo de quiz en funciones puras (sin `input()`/`print()`):

```python
def generate_anagram_prompt(card) -> dict
def check_anagram_answer(card, answer) -> dict
def generate_hook_prompt(card) -> dict
def check_hook_answer(card, given_hooks) -> dict
def generate_pattern_prompt(card) -> dict
def check_pattern_answer(card, answer) -> dict
def generate_morphology_prompt(card) -> dict
def check_morphology_answer(card, prefix, suffix) -> dict
def generate_transformation_prompt(card, trie) -> dict
def check_transformation_answer(card, given, actual) -> dict
def generate_extension_prompt(card, trie) -> dict
def check_extension_answer(card, given, actual) -> dict
def generate_reduction_prompt(card, trie) -> dict
def check_reduction_answer(card, given, actual) -> dict
```

Tanto `quiz.py` (CLI) como `quiz_ws.py` (web) llamarán estas funciones.

#### 1b. REST endpoints (`web/api.py`)

```
GET  /api/validar/{word}      → validez + metadatos (puntos, ganchos, prefijo/sufijo)
GET  /api/transformar/{word}   → cambios de una letra
GET  /api/extender/{word}      → inserciones de una letra
GET  /api/reducir/{word}       → eliminaciones de una letra
GET  /api/mazos                → lista de mazos con conteo de palabras
GET  /api/progreso             → estadísticas SRS
GET  /api/mazo/{name}          → tarjetas de un mazo específico
```

#### 1c. WebSocket endpoint (`web/quiz_ws.py`)

```
WS /ws/quiz

Cliente → Servidor:
  {"type": "start", "mode": "anagram", "deck": "words-5", "size": 20}
  {"type": "answer", "card_index": 3, "answer": "palabra"}
  {"type": "rate", "card_index": 3, "quality": 4}

Servidor → Cliente:
  {"type": "card", "index": 3, "total": 20, "prompt": {...}}
  {"type": "result", "correct": true, "quality": 5, "reveal": {...}}
  {"type": "summary", "reviewed": 20, "avg_quality": 3.8, "struggling": [...]}
```

### Fase 2: Frontend UI

#### Arquitectura

```
scrabble/web/
├── app.py                    # FastAPI: REST + WebSocket + archivos estáticos
├── api.py                    # Rutas REST
├── quiz_ws.py                # WebSocket para sesiones de quiz
└── static/
    ├── index.html            # Dashboard / página principal
    ├── quiz.html             # Sesión de quiz
    ├── explorer.html         # Explorador de palabras
    ├── css/
    │   └── style.css         # Estilos responsive (mobile-first)
    └── js/
        ├── quiz.js           # Lógica de sesión (WebSocket)
        ├── explorer.js       # Lógica del explorador
        └── common.js         # Utilidades compartidas
```

#### Página principal (`index.html`)

- Dashboard: palabras estudiadas, pendientes hoy, dominadas, racha
- Acceso rápido: "Iniciar repaso", "Explorar palabras", navegador de mazos
- Todo el texto en español

#### Página de quiz (`quiz.html`)

- **Selector de mazos** organizado por categoría:
  - Por longitud: 2, 3, 4, 5 letras
  - Patrones vocálicos: 7L con 2 vocales / 2 consonantes
  - Alta probabilidad y puntuación
  - Por terminación: 5L en D, L, N, R, Z
  - Verbos: por longitud (3-8), por comienzo, por tipo
  - Estudio por grupo: prefijos, sufijos, terminaciones
- **Selector de modo** — los 8 modos
- **Tamaño de sesión** (10–50)
- **Área de tarjetas**:
  - Palabra grande estilo ficha de Scrabble (fondo beige, subíndice de puntos)
  - Input: campo de texto + teclado en pantalla para móvil (CH, LL, RR, Ñ)
  - Botones de calificación (0–5): Nulo / Error / Difícil / Correcto / Bien / Fácil
  - Panel de revelación: ganchos, prefijo, sufijo, terminación, tipo, anagramas, puntos
- **Barra de progreso** + resumen de sesión

#### Explorador de palabras (`explorer.html`)

- Barra de búsqueda: "Escribe una palabra..."
- Validación en tiempo real
- Para palabras válidas, panel con pestañas:
  - **Info**: longitud, puntos, percentil, prefijo, sufijo, ganchos, anagramas
  - **Transformaciones**: cambios por posición
  - **Extensiones**: inserciones por posición
  - **Reducciones**: sub-palabras válidas

---

## Milestone C — Móvil y Pulido

| # | Tarea | Esfuerzo |
|---|-------|----------|
| 10 | Diseño responsive mobile-first + renderizado de fichas Scrabble | Medio |
| 11 | Teclado en pantalla (con CH, LL, RR, Ñ) | Pequeño |

---

## Milestone D — Contenido Enriquecido (Definiciones, Imágenes, Mnemónicos)

Actualmente el sistema solo sabe si una palabra es válida, su valor en puntos, ganchos y morfología. No contiene significados, imágenes ni ayudas de memoria.

| # | Tarea | Esfuerzo | Descripción |
|---|-------|----------|-------------|
| 12 | Modelo de datos para definiciones | Pequeño | Archivo `definitions.json` o tabla en BD mapeando palabra → definición corta, mnemónico, categoría semántica, URL de imagen |
| 13 | Fuente de definiciones | Grande | Evaluar opciones: diccionario RAE (restricciones legales), Wikcionario (licencia libre), diccionario colaborativo de la comunidad Scrabble, o IA generativa para definiciones cortas |
| 14 | Interfaz de edición de definiciones | Medio | Formulario web para agregar/editar definiciones manualmente. Cualquier usuario puede contribuir definiciones al explorador |
| 15 | Mostrar definiciones en el explorador | Pequeño | Pestaña "Significado" en el explorador de palabras con definición, categoría semántica y mnemónico |
| 16 | Mostrar definiciones en el quiz (reveal) | Pequeño | Panel de revelación incluye definición después de cada tarjeta en todos los modos |
| 17 | Imágenes/thumbnails para sustantivos | Grande | Generar o buscar imágenes para sustantivos comunes. Vincular con definiciones. Mostrar en explorador y reveal |
| 18 | Mnemónicos para palabras difíciles | Medio | Sistema de ayudas de memoria: frases cortas, asociaciones, contexto ("YANGÜES = pueblo de Don Quijote"). Pueden ser generados por IA o contribuidos por la comunidad |
| 19 | Categorías semánticas | Medio | Clasificar palabras por campo semántico (animales, plantas, geografía, etc.) para crear mazos temáticos de estudio |
| 20 | Importar/exportar definiciones | Pequeño | Formato JSON/CSV para compartir definiciones entre usuarios o clubs |

### Fuentes de Datos Posibles

| Fuente | Ventajas | Desventajas |
|--------|----------|-------------|
| **RAE (dle.rae.es)** | Autoridad oficial, completa | Restricciones de uso, no permite scraping |
| **Wikcionario** | Licencia libre (CC BY-SA) | Cobertura incompleta para palabras raras del Scrabble |
| **Comunidad Scrabble** | Relevante, enfocada en memorización | Requiere esfuerzo de contribución, control de calidad |
| **IA generativa** | Rápida, puede generar mnemónicos | Puede tener errores, requiere verificación |
| **Combinación** | Lo mejor de cada fuente | Mayor complejidad de integración |

### Estructura de Datos Propuesta

```json
{
  "casa": {
    "definicion": "Edificio para habitar",
    "mnemonico": "",
    "categoria": "construcción",
    "imagen_url": "",
    "fuente": "wikcionario",
    "contribuido_por": ""
  }
}
```

---

## Milestone E — Multijugador y Cloud

| # | Tarea | Esfuerzo |
|---|-------|----------|
| 21 | Carreras de quiz multijugador (modo club/LAN) | Grande |
| 22 | Despliegue en cloud: Docker + autenticación + BD | Grande |

---

## Traducciones Clave (UI)

| Inglés | Español |
|--------|---------|
| Review | Repaso |
| Anagram | Anagrama |
| Hooks | Ganchos |
| Pattern | Patrón |
| Morphology | Morfología |
| Transformation | Transformación |
| Extension | Extensión |
| Reduction | Reducción |
| Valid / Invalid | Válida / No válida |
| Front hooks / Back hooks | Ganchos delanteros / Ganchos traseros |
| Prefix / Suffix / Ending | Prefijo / Sufijo / Terminación |
| Points / Letters | Puntos / Letras |
| Progress / Due today | Progreso / Pendientes hoy |
| Mastered / Struggling | Dominadas / En dificultad |
| Start session / Quit | Iniciar sesión / Salir |
| Correct / Wrong / Reveal / Next | Correcto / Incorrecto / Revelar / Siguiente |
| Session summary | Resumen de sesión |
| Word explorer / Check word | Explorador de palabras / Consultar palabra |

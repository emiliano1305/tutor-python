# Tutor de Python con fuente acotada

Aplicación educativa en Python que usa recuperación de información (RAG) para responder a estudiantes **solo a partir de un PDF elegido por quien ejecuta la aplicación**. Busca fragmentos localmente y solicita una explicación en español con citas a páginas del PDF. Incluye un modo para pedir código o ejercicios; el tutor debe abstenerse si el material no respalda los conceptos necesarios.

## Importante sobre el PDF compartido

El archivo proporcionado es *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron, 2019), un libro de aprendizaje automático, no un manual general de Python para principiantes. Puede respaldar algunos ejemplos y flujos en Python relacionados con aprendizaje automático, pero no necesariamente `print`, variables, condicionales o fundamentos similares. La aplicación no inventa una progresión que no está en el libro y está configurada para abstenerse si no recupera evidencia.

El repositorio **no incluye ni redistribuye el PDF**. Guardalo localmente como `data/book.pdf` si tenés derecho a usarlo, o cargalo desde la interfaz. No subas a GitHub material protegido sin autorización.

## Funciones

- Interfaz web local en español (Streamlit).
- Extracción por página y segmentación del PDF con `pypdf`.
- Búsqueda TF-IDF local; el libro completo no se manda al modelo.
- Traducción/normalización de la consulta para buscar en el libro en inglés.
- Umbral configurable de evidencia y respuesta de abstención para consultas sin respaldo.
- Respuestas con citas del tipo `[S1]` y página física del PDF; se neutralizan IDs de cita desconocidos.
- Modo de explicación y modo de construcción de código/ejercicio respaldado por la fuente.
- Código creado por el tutor no se ejecuta automáticamente.

## Requisitos

- Python 3.10 o superior.
- Una clave de API para un modelo compatible con la API de chat de OpenAI. El nombre del modelo se puede cambiar con `OPENAI_MODEL`.
- PDF de texto seleccionable. Los PDF escaneados podrían necesitar OCR antes de cargarlos.

## Instalación y ejecución

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

Editá `.env` con tu clave (no la publiques ni la agregues a Git):

```dotenv
OPENAI_API_KEY=tu_clave
OPENAI_MODEL=gpt-4o-mini
# Opcional, para una API compatible:
# OPENAI_BASE_URL=https://tu-endpoint/v1
# Opcional, si guardás el PDF en otra ubicación:
# BOOK_PDF=data/book.pdf
```

Luego:

```bash
streamlit run app.py
```

Abrí la dirección local que Streamlit muestre en la terminal. También se puede subir el PDF desde el panel lateral.

## Estructura

```text
python-tutor/
├── app.py             # interfaz y flujo RAG
├── core.py            # extracción, fragmentación e índice local
├── requirements.txt
├── .env.example
├── .gitignore
├── data/.gitkeep
└── tests/test_core.py
```

## Pruebas

```bash
pytest -q
```

Las pruebas del núcleo no necesitan clave API. Para probar respuestas completas se requiere configurar el modelo y cargar un PDF autorizado.

## Límites y uso responsable

- La restricción a la fuente es una mitigación, no una garantía matemática contra errores del modelo. Revisá siempre las páginas citadas.
- La similitud de búsqueda no demuestra que una respuesta esté respaldada; el umbral de la interfaz permite ajustar cuán conservador es el tutor.
- El modo de código solo genera texto; no ejecuta, instala ni publica el código.
- El archivo se procesa en memoria en el equipo donde se ejecuta Streamlit. Solo se envían al modelo la consulta y los fragmentos recuperados necesarios para responder.
- No incluyas datos personales del alumnado en las preguntas.
- La numeración de citas corresponde a la página física del PDF, que puede diferir de la numeración impresa del libro.

## Publicarlo en GitHub

1. Revisá que `data/book.pdf`, `.env` y cualquier índice/archivo privado no estén en el commit.
2. Creá un repositorio vacío en GitHub.
3. Desde esta carpeta ejecutá:

```bash
git init
git add app.py core.py requirements.txt README.md .env.example .gitignore data/.gitkeep tests/test_core.py
git commit -m "Crear tutor Python con fuente acotada"
git branch -M main
git remote add origin https://github.com/USUARIO/REPOSITORIO.git
git push -u origin main
```

No pegues la clave API en el repositorio; si luego lo desplegás, configúrala como secreto/variable del entorno de despliegue.

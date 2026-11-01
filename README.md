# Student handbook assistant

A Python API that answers questions using the ZAIO student handbook. It extracts
PDF text, creates embeddings, saves them in FAISS, and uses an LLM to select an
answer with a page reference. If the information is missing, it returns a refusal.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | FastAPI app and `/ask` endpoint |
| `read_pdf.py` | PDF extraction and text chunks |
| `database.py` | Embeddings, FAISS storage and retrieval |
| `ai.py` | Model calls and answer/source validation |
| `load_data.py` | Build the handbook index |
| `setup_models.py` | Download models explicitly |
| `check_answers.py` | Run the question list and save results |
| `tests/` | Automated tests |
| `questions.json` | Supported and unsupported test questions |
| `results/` | Recorded question/source/answer results |

## Before starting

Use **Python 3.12**. Extract this ZIP and open the `rag_assignment` folder in VS
Code. Select **Terminal > New Terminal**. These commands are for Windows
PowerShell, and must run inside the folder containing `main.py`.

A virtual environment keeps this project's packages separate. Commands below use
its Python directly, so activating it or changing PowerShell execution policy is
not necessary.

If you see `ModuleNotFoundError: No module named 'fastapi'`, install the application
dependencies with `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
Then start the API with `.\.venv\Scripts\python.exe -m uvicorn main:app --reload`.
In VS Code, use **Python: Select Interpreter** and choose
`.venv\Scripts\python.exe` so editor runs use the same installed packages.

## 1. Quick tests — start here

These check answer validation and provider request handling. They use controlled
responses and do not download or call a chat model.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pytest==9.2.0 httpx==0.28.1
.\.venv\Scripts\python.exe -m pytest tests/test_ai.py -q
```

Expected: **19 passed**. A green dot means a check passed. `FAILED` means a check
failed; read the error underneath it. The `-q` option keeps the output short.
Use `-v` instead to see every test name.

## 2. Full automated tests

Install the application dependencies and download the smaller embedding model.
This uses more data, particularly if PyTorch is not already installed. Use Wi-Fi.
The large chat model is not required for these tests.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_models.py --embeddings-only
.\.venv\Scripts\python.exe -m pytest -q
```

The full suite contains **55 checks**. It tests PDF reading, chunks, embedding
retrieval, index loading, API input validation, provider failures and answer
validation. The handbook is already at `data/handbook.pdf`. Fixtures build test
indexes in temporary folders; you do not need to run `load_data.py` first.

These tests include model test doubles. Passing them checks the code; it does not
mean every possible generated answer will be correct.

## 3. Run the actual assistant

The default local model is Qwen2.5-3B-Instruct Q4_K_M. Its first-time download is
about 2.1 GB, plus the embedding model. Model weights are not included in this ZIP.
Use Wi-Fi and preferably a computer with at least 8 GB RAM.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe setup_models.py
.\.venv\Scripts\python.exe load_data.py
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

The local model package can require C++ build tools if a compatible wheel is not
available for your computer. Installation errors happen before the application
runs; keep the exact error text when troubleshooting.

On Windows x64 with Python 3.12, `requirements-local.txt` selects a prebuilt
runtime with automatic CPU backend selection, including CPUs without AVX.
The wheel is about 177 MB and runs locally on the CPU; the model is a separate
download. To repair the runtime without changing `.env` or rebuilding the index:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

If `/ask` returns 503, the server terminal logs the underlying exception.
`LocalEntryNotFoundError` means the configured model is absent from the local
cache: run `setup_models.py` to finish the explicit download (about 2.1 GB for
the default Qwen model). `No module named 'llama_cpp'` means the local runtime
needs the installation command above. Windows error `0xc000001d` during
`llama_backend_init()` indicates an illegal CPU instruction; the portable Windows
wheel pinned above fixes the failure observed on this machine. Stop and restart
the API after installing it so the previous native DLLs are unloaded.
Keep your existing `.env` and index when repairing these setup problems.

The local runtime uses a 4,096-token context and 128-token batches to reduce
memory pressure. On 1 November 2026, a real HTTP request using the existing index
and Qwen model returned `200`, `Total Fees: R 29,950`, and `Page 16`.
The cold request took 1,139 seconds on this machine with no AVX support and very
little free RAM; local inference can therefore be slow even after setup succeeds.
The recorded response is in `results/runtime_verification.json`.

Leave this terminal running. Open **http://127.0.0.1:8000/docs** in your browser.
Expand **POST /ask**, select **Try it out**, paste this body, then select **Execute**:

```json
{"question":"What are the total bootcamp fees?"}
```

A successful response has status **200**, an answer containing **R 29,950**, and
`"source": "Page 16"`. The wording may vary. Page numbers are physical PDF pages,
which sometimes differ from the handbook's printed labels.

Try an unsupported question such as `What is the refund policy?`. It should return:

```json
{"answer":"I could not find that information in the available knowledge base.","source":null}
```

A blank/missing/invalid question returns **422**. A missing index, unavailable model
or invalid model response returns **503**. Stop the server with **Ctrl+C**.

## 4. Test the full question list

Use a second terminal while the server is running, or stop the server first.
The script below calls the same Python assistant directly; the API server is not
required. Stopping the server avoids loading a second copy of the local model.

```powershell
.\.venv\Scripts\python.exe check_answers.py --live --output results/my_results.json
```

This runs 17 questions and saves actual answers and source pages. Review them
against the reference answers. To inspect retrieval without calling the LLM:

```powershell
.\.venv\Scripts\python.exe check_answers.py
```

A retrieval-only run leaves ungenerated answers marked `NOT_RUN_LLM`.
`results/test_results.md` and `results/test_results.json` contain the recorded
local-model results from 31 October 2026, for this project.
Run the commands above to produce your own results.

## Other configurations

The default uses MiniLM embeddings, FAISS retrieval with keyword reranking, and a
local Qwen model. Only exact answer extracts present in retrieved chunks are
accepted. Sentence conditions are retained when an extract stops mid-sentence.
The model can still choose a wrong fact, so manual answer review matters.

For an existing Ollama installation, set `LLM_PROVIDER=ollama`,
`LLM_BASE_URL=http://127.0.0.1:11434` and `LLM_MODEL` to your installed model in
`.env`. For a hosted compatible provider, set `LLM_PROVIDER=compatible`, its `/v1`
base URL, model ID and private API key. Restart the server after configuration
changes. Never commit `.env`.

The optional `--backend tfidf` argument to `load_data.py` selects lexical embeddings
instead of MiniLM. Leave `MIN_SCORE` unset for backend-specific thresholds.
For tests, `TEST_EMBEDDINGS=tfidf` selects this alternative; the neural-specific
schedule test is skipped. The normal full test command uses neural embeddings.

## Submission

Include source code, requirements, README, tests and question/source/answer results
as requested. The PDF contains private course meeting details, so `.gitignore`
excludes PDFs and generated indexes from public Git commits. After cloning a
repository, add your authorized handbook at `data/handbook.pdf` and build the index.

The API accepts JSON for the next n8n practical. In n8n, send a POST request to
`http://127.0.0.1:8000/ask` with a JSON `question`, then read `answer` and `source`.
If n8n runs in Docker, use the host address reachable from the container instead
of the container's own `127.0.0.1`.

## Latest file-layout check

After reorganising the files: the 19 quick tests passed. The TF-IDF test run
passed 54 checks and skipped the one neural-only schedule check. The full command
in section 2 uses neural embeddings and includes all 55 checks. The recorded
17-question model run in `results/` is from 31 October 2026.

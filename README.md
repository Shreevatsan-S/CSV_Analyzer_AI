# CSV AI Analyzer

Simple app to analyze CSV files with Google Gemini API.

## Features

- Upload CSV + prompt
- Gemini provides insights and matplotlib code
- Code is executed and graph is displayed

## Usage

1. Clone the repo
2. Create `.env` file with your `GOOGLE_API_KEY`
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run app:

```bash
python app.py
```

5. Open browser at `http://127.0.0.1:5000`

## Warning

- `exec` is used to run Gemini's code. This is NOT safe for production.
- Only for demo purposes.

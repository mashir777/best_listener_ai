python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy config\keys.env.example config\keys.env
python main.py

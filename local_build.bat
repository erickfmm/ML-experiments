python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt

python -m spacy download en_core_web_sm
python -m spacy download es_core_news_lg
python -m spacy download es_core_news_sm
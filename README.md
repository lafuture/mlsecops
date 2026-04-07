В репозитории: заполненный appendix.xlsx, скрипты генерации и обучения, PDF-отчёт по ТЗ.

## Окружение

```bash
cd "/path/to/ML Ozon"
python3 -m venv scripts/.venv
source scripts/.venv/bin/activate   
pip install -r requirements.txt
```

## Модели

- `task3.py` / `task4.py`: `Qwen/Qwen2.5-7B-Instruct` (Hugging Face). Первый запуск скачает веса (нужен интернет).

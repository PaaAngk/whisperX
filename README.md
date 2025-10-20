# Инструкция по запуску WhisperX в Docker

Этот `Dockerfile` и `docker-compose.yml` предназначены для запуска приложения WhisperX в изолированном окружении с поддержкой GPU (CUDA 12.6+) и предзагруженными моделями для русского языка.

## Требования

1.  **Docker** и **Docker Compose**
2.  **NVIDIA GPU** с установленным драйвером версии 550.xx или новее.
3.  **NVIDIA Container Toolkit** для поддержки GPU в Docker.

## Настройка

1.  **Токен Hugging Face**
    Для скачивания модели диаризации (разделения спикеров) требуется токен Hugging Face.
    - Создайте файл `.env` в этой же директории (`whisperX/.env`).
    - Добавьте в него ваш токен в следующем формате:
      ```
      HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      ```
    `docker-compose` автоматически подхватит этот файл.

## Сборка Docker-образа

Образ будет собран с предзагруженными моделями, что может занять некоторое время (15-30 минут в зависимости от скорости сети и диска).

Откройте терминал в директории `whisperX` и выполните команду:

```bash
docker-compose build
```

Эта команда создаст образ с именем `whisperx-ru` и тегом `latest`.

## Запуск транскрибации

Для запуска обработки аудиофайла используйте `docker-compose run`.

- Убедитесь, что ваши аудиофайлы находятся в директории `data` (на уровень выше `whisperX`).
- Команда монтирует директорию `../data` в `/app/data` внутри контейнера.

### Пример команды

```bash
docker-compose run --rm whisperx data/cut5m.wav --language ru --model large-v3 --align_model jonatasgrosman/wav2vec2-large-xlsr-53-russian --diarize --print_progress True --compute_type float16
```

### Разбор аргументов:

- `docker-compose run --rm whisperx`: Запускает сервис `whisperx` из `docker-compose.yml` и удаляет контейнер после выполнения (`--rm`).
- `data/cut5m.wav`: Путь к вашему файлу *внутри контейнера*.
- `--language ru`: Язык.
- `--model large-v3`: Модель Whisper.
- `--align_model ...`: Модель для выравнивания.
- `--diarize`: Включить диаризацию (разделение спикеров).
- `--print_progress True`: Показывать прогресс.
- `--compute_type float16`: Использовать `float16` для ускорения на совместимых GPU.

Результаты (файлы `.srt`, `.tsv` и т.д.) будут сохранены в директории `data`, рядом с исходным аудиофайлом.

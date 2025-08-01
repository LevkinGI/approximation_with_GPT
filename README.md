# Spectral Pipeline

Этот проект выполняет аппроксимацию сигналов из файлов `.dat` и строит графики.

## Установка
```bash
pip install -r requirements.txt
```

## Запуск
```bash
python find_freqs_and_visualize.py data
```
После обработки скрипт открывает интерактивный график со спектрами в браузере.

### Логирование
Все сообщения сохраняются в `logs/pipeline.log`. Уровень подробности можно
настроить опцией `--log-level`, например:

```bash
python find_freqs_and_visualize.py data --log-level=DEBUG
# или
python -m spectral_pipeline.cli data --log-level=DEBUG
```
При значении `DEBUG` выводятся дополнительные сведения о работе ESPRIT и
fallback-алгоритма.

## Тесты
```bash
pytest
```


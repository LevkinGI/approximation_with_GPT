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
Все сообщения сохраняются в `logs/pipeline.log`. По умолчанию используется
подробный уровень `DEBUG`. Его можно изменить опцией `--log-level`, например:

```bash
python find_freqs_and_visualize.py data --log-level=INFO
# или
python -m spectral_pipeline.cli data --log-level=INFO
```
При уровне `DEBUG` выводятся дополнительные сведения о работе ESPRIT и
fallback-алгоритма. Получившийся файл `logs/pipeline.log` можно отправлять для
диагностики и анализа результатов.

## Тесты
```bash
pytest
```


### Настройки аппроксимации
Параметры модели и оптимизатора вынесены в `spectral_pipeline/approximation_config.py`.

Ключевые флаги:
- `use_theory_guess` — использовать теоретические значения в качестве первого приближения.
- `force_lf_only` — применять LF-only режим ко всей программе целиком; если `False`, везде используется совместная аппроксимация LF+HF (когда HF доступен).
- `equal_amplitudes` — равенство амплитуд мод.
- `equal_phases` — равенство фаз мод.
- `zero_phases_if_equal` — фиксация фаз в ноль, применяется только если `equal_phases=True`.
- `lf_band_hz` / `hf_band_hz` — полосы частот поиска и фильтрации кандидатов.
- `outlier_intervals` — интервалы по оси `x`, где вырезаются выбросы перед аппроксимацией.
- `cutoff_lf_s` / `cutoff_hf_s` — длительность окна сигнала для LF/HF после пика.
- `init_freq_lo_mul` / `init_freq_hi_mul` — множители для начальных частотных границ fit.
- `tau1_bounds_s` / `tau2_bounds_s` — границы времени затухания для обеих мод.

Также там находятся прочие настройки аппроксимации: `max_cost` и параметры `least_squares` (`ftol`, `xtol`, `gtol`, `max_nfev`, `loss`, `f_scale`, `x_scale`).

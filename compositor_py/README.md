# Compositor (Python/Qt port)

Кроссплатформенная версия графического редактора Compositor, перенесённая
с Swift/SwiftUI (macOS) на Python + PyQt6 (Linux / Windows / macOS).

## Установка

```bash
python3 -m venv .venv
# Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

Требования: Python ≥ 3.9, PyQt6, numpy, Pillow.
Опционально: `pillow-heif` (импорт HEIC), `scipy` (ускорение фильтров —
без него используется встроенный numpy-фолбэк).

На Linux для запуска Qt-приложения нужны системные библиотеки X/Wayland
(обычно уже есть в десктопных дистрибутивах):
`sudo apt install libegl1 libgl1 libxkbcommon0 libdbus-1-3 libfontconfig1`

## Запуск

Из корня репозитория:

```bash
python -m compositor_py                 # обычный запуск
python -m compositor_py my.compositor   # открыть проект
```

Или установите пакет один раз (создаст команду `compositor`):

```bash
pip install -e .
compositor
```

### Windows без консоли (окно без чёрного окна cmd)

```bat
pythonw -m compositor_py
```

### Проверка без дисплея (CI / сервер)

```bash
QT_QPA_PLATFORM=offscreen python -m compositor_py
```

## Структура

| Python (`compositor_py/`)        | Оригинал (Swift)                     |
|----------------------------------|--------------------------------------|
| `main.py`, `__main__.py`         | `CompositorApp.swift`, `ContentView.swift` |
| `core/model.py`                  | `Document/*` (модели слоёв, документов) |
| `core/session.py`                | `Document/EditorSession*.swift`      |
| `core/editing.py`                | `Document/*Editing, *Tool, *Selection.swift` |
| `core/compositor.py`             | `Rendering/*` (композиция, blend-режимы, эффекты) |
| `io_files/project_io.py`         | `IO/*` + `Document/EditorSession+Projects.swift` (формат `.compositor`, PSD-ридер, экспорт PNG/JPEG) |
| `ui/canvas_view.py`              | `Rendering/CanvasSurface.swift`, `UI/CanvasRulers.swift`, `TransformInspector.swift` |
| `ui/layers_panel.py`             | `UI/LayersPanel.swift`, `NativeLayerList.swift` |
| `ui/tool_options.py`             | `UI/ToolOptionsBar/BrushControls/ShapeControls/...` |
| `ui/dialogs.py`, `ui/curve_widget.py` | `UI/LevelsSheet, CurvesControls, HueSaturationSheet, EffectsSheet, NewCanvasSheet` |

## Горячие клавиши

`V M L W C B J S R G U T I H Z` — инструменты; `[`/`]` — размер кисти;
`,`/`.` — жёсткость; `E` — ластик; `Ctrl+Z / Ctrl+Shift+Z` — undo/redo;
`Ctrl+A/D` — выделить/снять; `Delete` — удалить выделенное; стрелки — сдвиг
(Shift ×10); `Ctrl+0/1` — вписать/100%; колесо мыши — зум.

# Official Ultralytics Refactor Log

## 2026-06-16

- Started in-place execution of `docs/superpowers/plans/2026-06-16-yolo26-official-ultralytics.md`.
- User explicitly declined creating an isolated worktree.
- Important path constraint: do not pass `project="runs"` by default. In the installed Ultralytics version, relative `project` is nested under `runs/<task>/<project>/<name>`, so `project=None` preserves the normal `runs/<task>/<name>` layout.
- Existing dirty files before this refactor: `train.py`, `utils/utils_bbox.py`, and `tests/test_utils_bbox.py` from the earlier debugging attempt. This refactor is expected to overwrite/delete those paths as part of removing local training/inference code.
- Task 1 complete: replaced `train.py` with a thin official `ultralytics.YOLO.train()` wrapper, added focused dispatch tests in `tests/test_train_official.py`, verified red failure before implementation, then verified `python -m pytest -q tests\test_train_official.py` passes.
- Task 2 complete: added `tests/test_predict_official.py`, verified the tests failed against the old local prediction entry point, then replaced `predict.py` with a thin official `ultralytics.YOLO.predict()` / `YOLO.export()` wrapper. `project=None` is omitted from prediction kwargs so Ultralytics keeps its default run layout. Verified `python -m pytest -q tests\test_predict_official.py` passes.
- Task 3 complete: replaced obsolete helper tests that imported removed custom train helpers and `utils.utils.measure_text` with focused `get_map.default_model_path` coverage. Verified red failure against the old `runs/<task>/logs/*_unfreeze/weights/best.pt` lookup, then changed discovery to the official `runs/<task>/*/weights/best.pt` layout while keeping validation on `ultralytics.YOLO.val()`. Verified `python -m pytest -q tests\test_helpers.py` passes.
- Task 4 deletion step applied: removed `yolo.py`, `nets/yolo_training.py`, `utils/utils_fit.py`, `utils/callbacks.py`, `utils/utils_bbox.py`, `utils/utils_map.py`, `utils/utils.py`, and `tests/test_utils_bbox.py`. Verified focused wrapper/helper tests still pass after deletion.
- Task 5 complete: updated `README.md` and `CLAUDE.md` to describe the official Ultralytics wrapper workflow, removed stale references to deleted local training/inference helpers, documented the `project=None` path constraint, and updated verification commands to compile `train.py get_map.py predict.py scripts tests`.
- Pre-commit verification complete: `python -m pytest -q` passed with 25 tests, `python -m compileall -q train.py get_map.py predict.py scripts tests` passed, and `python -c "import train, get_map, predict; print('official wrappers import OK')"` passed.

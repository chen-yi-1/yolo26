import csv
from pathlib import Path


def _read_numeric_results(csv_path):
    with Path(csv_path).open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = []
        for raw_row in reader:
            row = {}
            for raw_key, raw_value in raw_row.items():
                if raw_key is None:
                    continue
                key = raw_key.strip()
                value = (raw_value or "").strip()
                if not key or value == "":
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            if row:
                rows.append(row)
    return rows


def _plot_columns(axis, rows, x_values, columns, title):
    for column in columns:
        y_values = [row[column] for row in rows if column in row]
        x_for_column = [x for x, row in zip(x_values, rows) if column in row]
        if y_values:
            axis.plot(x_for_column, y_values, linewidth=1.8, label=column)
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.grid(True, alpha=0.25)
    if axis.lines:
        axis.legend(fontsize=8)


def plot_training_results(csv_path, output_path=None):
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_name("training_summary.png")
    output_path = Path(output_path)

    rows = _read_numeric_results(csv_path)
    if not rows:
        return None

    x_values = [
        int(row["epoch"]) if "epoch" in row else index
        for index, row in enumerate(rows, 1)
    ]
    columns = [column for column in rows[0] if column != "epoch"]
    train_loss_columns = [
        column for column in columns
        if column.startswith("train/") and "loss" in column.lower()
    ]
    val_loss_columns = [
        column for column in columns
        if column.startswith("val/") and "loss" in column.lower()
    ]
    metric_columns = [column for column in columns if column.startswith("metrics/")]
    lr_columns = [column for column in columns if column.startswith("lr/")]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()
    _plot_columns(axes[0], rows, x_values, train_loss_columns, "Train Loss")
    _plot_columns(axes[1], rows, x_values, val_loss_columns, "Validation Loss")
    _plot_columns(axes[2], rows, x_values, metric_columns, "Validation Metrics")
    _plot_columns(axes[3], rows, x_values, lr_columns, "Learning Rate")
    fig.suptitle(f"Training Summary: {csv_path.parent}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_training_results_from_run(train_result, model=None):
    save_dir = getattr(train_result, "save_dir", None)
    if save_dir is None and model is not None:
        trainer = getattr(model, "trainer", None)
        save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        return None
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.is_file():
        return None
    return plot_training_results(csv_path)


def add_training_plot_callback(model):
    def plot_results_on_fit_epoch_end(trainer):
        csv_path = Path(trainer.save_dir) / "results.csv"
        if csv_path.is_file():
            plot_training_results(csv_path)

    model.add_callback("on_fit_epoch_end", plot_results_on_fit_epoch_end)

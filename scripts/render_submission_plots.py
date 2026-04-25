from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "hack" / "assets"


MAIN_TASKS = ["single_fault", "n_minus_1", "cascade_prevent", "multi_stage_cascade"]

BASE_MAIN = {
    "single_fault": 0.856,
    "n_minus_1": 0.952,
    "cascade_prevent": 0.000,
    "multi_stage_cascade": 0.000,
}

SFT_MAIN = {
    "single_fault": 0.856,
    "n_minus_1": 0.990,
    "cascade_prevent": 0.990,
    "multi_stage_cascade": 0.9156444,
}

GRPO_MAIN = {
    "single_fault": 0.856,
    "n_minus_1": 0.990,
    "cascade_prevent": 0.990,
    "multi_stage_cascade": 0.9156444,
}

SFT_UNSEEN = {
    "single_fault": 0.830,
    "n_minus_1": 0.9222223,
    "cascade_prevent": 0.990,
    "multi_stage_cascade": 0.9069863,
}

GRPO_UNSEEN = {
    "single_fault": 0.7833333,
    "n_minus_1": 0.9222223,
    "cascade_prevent": 0.990,
    "multi_stage_cascade": 0.9069863,
}

MULTISTAGE_DAPO = {
    "SFT": 0.9156444,
    "HF GRPO (DAPO loss)": 0.9156444,
}

FAILURES = {
    "Base": 10,
    "SFT": 0,
    "GRPO": 0,
}

FRONTIER = [
    ("Base", 8, sum(BASE_MAIN.values()) / len(BASE_MAIN), "#222222"),
    ("SFT", 30, sum(SFT_MAIN.values()) / len(SFT_MAIN), "#e68613"),
    ("GRPO", 54, sum(GRPO_MAIN.values()) / len(GRPO_MAIN), "#d1495b"),
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE = font(34, bold=True)
SUB = font(20)
LABEL = font(18)
SMALL = font(16)


def canvas(title: str, subtitle: str, size: tuple[int, int] = (1600, 1000)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, "#f7f3ee")
    draw = ImageDraw.Draw(img)
    draw.text((70, 40), title, fill="#1f1f1f", font=TITLE)
    draw.text((70, 95), subtitle, fill="#5f5f5f", font=SUB)
    return img, draw


def draw_axes(draw: ImageDraw.ImageDraw, left: int, top: int, right: int, bottom: int, y_ticks: Iterable[float], y_label: str) -> None:
    draw.line((left, top, left, bottom), fill="#1f1f1f", width=3)
    draw.line((left, bottom, right, bottom), fill="#1f1f1f", width=3)
    for tick in y_ticks:
        y = bottom - int((bottom - top) * tick)
        draw.line((left - 8, y, right, y), fill="#e0d7cf", width=1)
        draw.text((left - 52, y - 10), f"{tick:.1f}", fill="#444444", font=SMALL)
    draw.text((left - 60, top - 30), y_label, fill="#444444", font=LABEL)


def save(img: Image.Image, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / name)


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    bottom: int,
    value: float,
    max_height: int,
    color: str,
    radius: int = 10,
) -> int:
    height = max(2, int(max_height * max(0.0, value)))
    y0 = bottom - height
    draw.rounded_rectangle((x0, y0, x1, bottom - 1), radius=radius, fill=color)
    return y0


def plot_grouped_bars() -> None:
    img, draw = canvas(
        "Benchmark Scores By Task",
        "Seed block 0..4, 5 episodes per task. Base vs final SFT vs completed GRPO.",
    )
    left, top, right, bottom = 120, 180, 1500, 850
    draw_axes(draw, left, top, right, bottom, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "Score")
    colors = {"Base": "#2f2f2f", "SFT": "#e68613", "GRPO": "#d1495b"}
    series = [("Base", BASE_MAIN), ("SFT", SFT_MAIN), ("GRPO", GRPO_MAIN)]
    group_w = (right - left) / len(MAIN_TASKS)
    bar_w = 70
    for i, task in enumerate(MAIN_TASKS):
        cx = left + int(group_w * i + group_w / 2)
        for j, (name, values) in enumerate(series):
            x0 = cx - 110 + j * 85
            x1 = x0 + bar_w
            y0 = draw_bar(draw, x0, x1, bottom, values[task], bottom - top, colors[name], radius=12)
            draw.text((x0 + 5, y0 - 26), f"{values[task]:.3f}", fill="#333333", font=SMALL)
        label = task.replace("_", "\n")
        draw.multiline_text((cx - 70, bottom + 20), label, fill="#222222", font=LABEL, align="center", spacing=2)
    legend_x = 1120
    for idx, (name, _) in enumerate(series):
        y = 145 + idx * 30
        draw.rounded_rectangle((legend_x, y, legend_x + 22, y + 22), radius=5, fill=colors[name])
        draw.text((legend_x + 34, y - 2), name, fill="#222222", font=LABEL)
    save(img, "benchmark_task_scores.png")


def plot_seen_unseen() -> None:
    img, draw = canvas(
        "Generalization And Stability",
        "Seen seeds 0..4 vs unseen seeds 100..102 for final SFT and completed GRPO.",
    )
    left, top, right, bottom = 120, 180, 1500, 850
    draw_axes(draw, left, top, right, bottom, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "Score")
    colors = {"SFT seen": "#e68613", "SFT unseen": "#f2b970", "GRPO seen": "#d1495b", "GRPO unseen": "#ef9aa9"}
    data = [("SFT seen", SFT_MAIN), ("SFT unseen", SFT_UNSEEN), ("GRPO seen", GRPO_MAIN), ("GRPO unseen", GRPO_UNSEEN)]
    group_w = (right - left) / len(MAIN_TASKS)
    bar_w = 42
    for i, task in enumerate(MAIN_TASKS):
        cx = left + int(group_w * i + group_w / 2)
        for j, (name, values) in enumerate(data):
            x0 = cx - 110 + j * 52
            x1 = x0 + bar_w
            draw_bar(draw, x0, x1, bottom, values[task], bottom - top, colors[name], radius=8)
        draw.multiline_text((cx - 70, bottom + 20), task.replace("_", "\n"), fill="#222222", font=LABEL, align="center", spacing=2)
    legend_x = 1040
    for idx, name in enumerate(colors):
        y = 130 + idx * 28
        draw.rounded_rectangle((legend_x, y, legend_x + 20, y + 20), radius=5, fill=colors[name])
        draw.text((legend_x + 30, y - 2), name, fill="#222222", font=SMALL)
    save(img, "generalization_seen_vs_unseen.png")


def plot_failures() -> None:
    img, draw = canvas(
        "Safety And Failure Count",
        "Main seed block 0..4. Lower is better.",
        size=(1300, 900),
    )
    left, top, right, bottom = 140, 180, 1180, 760
    max_val = max(FAILURES.values()) or 1
    draw_axes(draw, left, top, right, bottom, [0.0, 0.25, 0.5, 0.75, 1.0], "Failure ratio")
    colors = {"Base": "#2f2f2f", "SFT": "#e68613", "GRPO": "#d1495b"}
    gap = (right - left) / len(FAILURES)
    for idx, (name, value) in enumerate(FAILURES.items()):
        x0 = int(left + gap * idx + 70)
        x1 = x0 + 120
        ratio = value / max_val
        y0 = draw_bar(draw, x0, x1, bottom, ratio, bottom - top, colors[name], radius=12)
        draw.text((x0 + 42, y0 - 28), str(value), fill="#333333", font=LABEL)
        draw.text((x0 + 10, bottom + 20), name, fill="#222222", font=LABEL)
    save(img, "safety_failures.png")


def plot_multistage_dapo() -> None:
    img, draw = canvas(
        "Focused Multistage Comparison",
        "SFT vs HF GRPO run trained with DAPO loss. Same final score on multi_stage_cascade.",
        size=(1300, 900),
    )
    left, top, right, bottom = 140, 180, 1180, 760
    draw_axes(draw, left, top, right, bottom, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "Score")
    colors = ["#e68613", "#d1495b"]
    for idx, (name, value) in enumerate(MULTISTAGE_DAPO.items()):
        x0 = left + 180 + idx * 320
        x1 = x0 + 160
        y0 = draw_bar(draw, x0, x1, bottom, value, bottom - top, colors[idx], radius=12)
        draw.text((x0 + 32, y0 - 28), f"{value:.6f}", fill="#333333", font=LABEL)
        draw.multiline_text((x0 - 10, bottom + 20), name.replace(" ", "\n"), fill="#222222", font=LABEL, align="center", spacing=2)
    save(img, "multistage_dapo_focus.png")


def plot_frontier() -> None:
    img, draw = canvas(
        "Performance vs Training Effort",
        "Main benchmark mean score. This is the high-level tradeoff view for judges.",
        size=(1500, 900),
    )
    left, top, right, bottom = 120, 160, 1380, 780
    draw.line((left, top, left, bottom), fill="#1f1f1f", width=3)
    draw.line((left, bottom, right, bottom), fill="#1f1f1f", width=3)
    poly = [(left, bottom - 260), (left, top + 40), (left + 520, top + 40)]
    draw.polygon(poly, fill="#fde5cf", outline="#efb57c")
    draw.text((left + 20, top + 10), "Best performance / effort region", fill="#d4761c", font=SMALL)
    for tick, label in [(0.4, "40%"), (0.5, "50%"), (0.6, "60%"), (0.7, "70%"), (0.8, "80%"), (0.9, "90%"), (1.0, "100%")]:
        y = bottom - int((bottom - top) * tick)
        draw.line((left - 8, y, right, y), fill="#e0d7cf", width=1)
        draw.text((left - 62, y - 10), label, fill="#444444", font=SMALL)
    draw.text((left + 10, top - 30), "Performance", fill="#333333", font=LABEL)
    draw.text((right - 280, bottom + 25), "Training / engineering effort", fill="#333333", font=LABEL)
    for name, xval, yval, color in FRONTIER:
        x = left + int((right - left) * (xval / 60.0))
        y = bottom - int((bottom - top) * yval)
        draw.ellipse((x - 14, y - 14, x + 14, y + 14), fill=color, outline="white", width=3)
        draw.text((x + 18, y - 10), name, fill="#222222", font=LABEL)
    save(img, "performance_vs_effort.png")


def main() -> None:
    plot_grouped_bars()
    plot_seen_unseen()
    plot_failures()
    plot_multistage_dapo()
    plot_frontier()
    print(f"wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()

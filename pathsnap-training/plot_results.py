"""Create metric comparison SVGs from the recorded classifier results."""
import argparse
import csv
from html import escape
from pathlib import Path


METRICS = {
    "cv_macro_f1": "Cross-validation Macro F1",
    "student_top1_accuracy": "Student-test Top-1 accuracy",
    "student_macro_f1": "Student-test Macro F1",
    "student_covered_log_loss": "Student-test log loss on covered observations",
}


def plot_results(comparison):
    with comparison.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for key, title in METRICS.items():
        axis_x, axis_width = 230, 530
        svg = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="310" viewBox="0 0 860 310">',
            '<rect width="860" height="310" fill="#ffffff"/>',
            f'<text x="20" y="32" font-family="sans-serif" font-size="21" font-weight="bold">{title}</text>',
        ]
        for index in range(6):
            x = axis_x + axis_width * index / 5
            svg.append(f'<path d="M{x},52 V269" stroke="#dbe4ef"/>')
            svg.append(f'<text x="{x}" y="290" text-anchor="middle" font-family="sans-serif" font-size="12">{index / 5:.1f}</text>')
        for index, row in enumerate(rows):
            y, value = 62 + index * 42, float(row[key])
            name = escape(row["model"].replace("hist_", "histogram_").replace("_", " ").title())
            svg.append(f'<text x="20" y="{y + 17}" font-family="sans-serif" font-size="15">{name}</text>')
            svg.append(f'<rect x="{axis_x}" y="{y}" width="{axis_width * value:.2f}" height="23" rx="3" fill="#235da8"/>')
            svg.append(f'<text x="{axis_x + axis_width * value + 9:.2f}" y="{y + 17}" font-family="sans-serif" font-size="14">{value:.4f}</text>')
        svg.append("</svg>")
        comparison.with_name(key + ".svg").write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    plot_results(parser.parse_args().comparison)

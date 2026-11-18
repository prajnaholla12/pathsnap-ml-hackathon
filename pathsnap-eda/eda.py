"""Dataset exploration on the train and student_test partitions (feature cache from data.py).

Writes results/eda_inventory.csv, results/eda_summary.csv, results/projection_check.csv
and the eda_*.png figures.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pathsnap-training"))
from data import project  # noqa: E402
from train import read_trips  # noqa: E402

BLUE, ORANGE = "#3b6ea8", "#d9822b"


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


def describe(partition, trip_id, trip):
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    dt = np.diff(track[:, 2])
    m = len(nodes)
    graph = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(m, m))
    _, strong = connected_components(graph, directed=True, connection="strong")
    n_strong = strong.max() + 1
    degree = np.bincount(arcs.ravel(), minlength=m)
    key, reverse = arcs[:, 0] * m + arcs[:, 1], arcs[:, 1] * m + arcs[:, 0]
    origin = track[:, :2].mean(axis=0)
    xy = project(track[:, :2], origin)
    step = haversine(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])
    covered = (trip["candidates"] == trip["labels"][:, None]).any(axis=1)
    return {
        "partition": partition, "trip_id": trip_id, "observations": len(track),
        "nodes": m, "arcs": len(arcs), "reference_arcs": len(route),
        "invalid_rows": int(trip["invalid_rows"]), "reordered": bool(trip["reordered"]),
        "duplicate_timestamps": int(np.sum(dt == 0)), "share_1s_steps": float(np.mean(dt == 1)),
        "gaps_over_10s": int(np.sum(dt > 10)), "maximum_gap_s": float(dt.max()),
        "duration_min": float((track[-1, 2] - track[0, 2]) / 60), "path_km": float(step.sum() / 1000),
        "lon_center": float(origin[0]), "lat_center": float(origin[1]),
        "extent_km": float(np.ptp(xy, axis=0).max() / 1000),
        "strong_components": int(n_strong),
        "largest_strong_share": float(np.bincount(strong).max() / m),
        "isolated_nodes": int(np.sum(degree == 0)),
        "two_way_share": float(np.mean(np.isin(reverse, key))),
        "route_connected_share": float(np.mean(arcs[route[:-1], 1] == arcs[route[1:], 0])) if len(route) > 1 else 1.0,
        "route_repeated_arcs": int(len(route) - len(np.unique(route))),
        "median_alignment_m": float(np.median(trip["alignment_distance"])),
        "share_alignment_over_50m": float(np.mean(trip["alignment_distance"] > 50)),
        "candidate_miss_rate": float(np.mean(~covered)),
    }


def run(cache, out):
    out.mkdir(parents=True, exist_ok=True)
    parts = {"train": read_trips(cache, "train"), "student_test": read_trips(cache, "student_test")}
    inv = pd.DataFrame([describe(p, t, trip) for p, trips in parts.items() for t, trip in trips.items()])
    inv.to_csv(out / "eda_inventory.csv", index=False)

    summary = inv.groupby("partition").agg(
        trips=("trip_id", "count"), observations=("observations", "sum"),
        observations_median=("observations", "median"), arcs_min=("arcs", "min"), arcs_max=("arcs", "max"),
        invalid_rows=("invalid_rows", "sum"), reordered_trips=("reordered", "sum"),
        duplicate_timestamps=("duplicate_timestamps", "sum"), gaps_over_10s=("gaps_over_10s", "sum"),
        maximum_gap_s=("maximum_gap_s", "max"),
        trips_alignment_over_50m_20pct=("share_alignment_over_50m", lambda x: int((x > 0.2).sum())),
    ).T
    summary.to_csv(out / "eda_summary.csv")
    print(summary)

    # local projection vs haversine
    rows = []
    for p, trips in parts.items():
        for t, trip in trips.items():
            track = trip["track"]
            xy = project(track[:, :2], track[:, :2].mean(axis=0))
            i = np.linspace(0, len(track) - 1, 60).astype(int)
            j = np.roll(i, -7)
            h = haversine(track[i, 0], track[i, 1], track[j, 0], track[j, 1])
            e = np.linalg.norm(xy[i] - xy[j], axis=1)
            keep = h > 100
            rows.append({"trip_id": t, "max_relative_error_pct": float(np.max(abs(e[keep] - h[keep]) / h[keep]) * 100),
                         "max_absolute_error_m": float(np.max(abs(e[keep] - h[keep])))})
    proj = pd.DataFrame(rows)
    proj.to_csv(out / "projection_check.csv", index=False)
    print("projection: worst %.3f%% / %.2f m" % (proj.max_relative_error_pct.max(), proj.max_absolute_error_m.max()))

    figures(inv, parts, out)


def figures(inv, parts, out):
    colour = inv.partition.map({"train": BLUE, "student_test": ORANGE})
    tr, te = inv[inv.partition == "train"], inv[inv.partition == "student_test"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist([tr.observations, te.observations], bins=np.logspace(2, 4.4, 20), color=[BLUE, ORANGE],
               label=["train", "student_test"])
    ax[0].set_xscale("log"); ax[0].set_xlabel("GPS readings per trip"); ax[0].set_ylabel("trips")
    ax[0].set_title("Trip length"); ax[0].legend()
    ax[1].scatter(inv.arcs, inv.observations, c=colour, s=18)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("arcs in the trip's network"); ax[1].set_ylabel("GPS readings")
    ax[1].set_title("Network size vs trip length")
    ax[2].scatter(inv.lon_center, inv.lat_center, c=colour, s=18)
    ax[2].set_xlabel("longitude"); ax[2].set_ylabel("latitude"); ax[2].set_title("Trip centres")
    plt.tight_layout(); plt.savefig(out / "eda_overview.png", dpi=130); plt.close()

    dts = np.concatenate([np.diff(t["track"][:, 2]) for trips in parts.values() for t in trips.values()])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    vals, counts = np.unique(np.clip(dts, 0, 11), return_counts=True)
    ax[0].bar([("%g" % v if v < 11 else ">10") for v in vals], counts, color=BLUE)
    ax[0].set_yscale("log"); ax[0].set_xlabel("seconds between readings"); ax[0].set_ylabel("count (log)")
    ax[0].set_title("Sampling interval (%.1f%% are 1 s)" % (100 * np.mean(dts == 1)))
    ax[1].bar(range(len(inv)), inv.gaps_over_10s, color=colour)
    ax[1].set_xlabel("trip (train, then student_test)"); ax[1].set_ylabel("gaps > 10 s")
    ax[1].set_title("Large time gaps per trip")
    plt.tight_layout(); plt.savefig(out / "eda_sampling.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(8, 3.4))
    cols = ["invalid_rows", "reordered", "duplicate_timestamps", "gaps_over_10s"]
    q = inv.groupby("partition")[cols].sum().astype(int)
    q.T.plot.bar(ax=ax, color=[ORANGE, BLUE])
    for patch in ax.patches:
        ax.annotate(f"{int(patch.get_height())}", (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_title("Missing/invalid rows and timestamp issues (summed over trips)"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(out / "eda_missing.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for p, c in (("train", BLUE), ("student_test", ORANGE)):
        d = np.concatenate([t["alignment_distance"] for t in parts[p].values()])
        ax[0].hist(np.clip(d, 0, 200), bins=80, histtype="step", density=True, color=c, lw=1.5, label=p)
    ax[0].axvline(50, ls="--", c="k")
    ax[0].set_xlabel("GPS reading to aligned reference arc (m, clipped at 200)")
    ax[0].set_title("GPS noise against the reference route"); ax[0].legend()
    worst = inv.sort_values("share_alignment_over_50m", ascending=False).head(12)
    ax[1].barh(worst.trip_id, worst.share_alignment_over_50m * 100,
               color=np.where(worst.share_alignment_over_50m > 0.2, "#c0392b", BLUE))
    ax[1].invert_yaxis(); ax[1].set_xlabel("% of readings > 50 m from the aligned arc")
    ax[1].set_title("Trips where reference and GPS disagree")
    plt.tight_layout(); plt.savefig(out / "eda_alignment.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].scatter(inv.nodes, inv.strong_components, c=colour, s=16)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("nodes"); ax[0].set_ylabel("strongly connected components")
    ax[0].set_title("Directed connectivity per trip network")
    ax[1].hist(inv.largest_strong_share, bins=20, color=BLUE)
    ax[1].set_xlabel("share of nodes in the largest SCC"); ax[1].set_ylabel("trips")
    ax[1].set_title("Largest strongly connected component")
    plt.tight_layout(); plt.savefig(out / "eda_network.png", dpi=130); plt.close()

    chosen = [("train", "00000048"), ("train", "00000021"), ("train", "00000033"),
              ("student_test", "00000040"), ("student_test", "00000056"), ("student_test", "00000091")]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, (p, t) in zip(axs.ravel(), chosen):
        trip = parts[p][t]
        origin = trip["track"][:, :2].mean(axis=0)
        pts, xy = project(trip["track"][:, :2], origin), project(trip["nodes"], origin)
        a, b = xy[trip["arcs"][:, 0]], xy[trip["arcs"][:, 1]]
        lo, hi = pts.min(0) - 300, pts.max(0) + 300
        near = ((a >= lo) & (a <= hi)).all(1)
        ax.plot(np.c_[a[near, 0], b[near, 0]].T, np.c_[a[near, 1], b[near, 1]].T, c="#cfcfcf", lw=0.4)
        r = trip["route"]
        ax.plot(np.c_[a[r, 0], b[r, 0]].T, np.c_[a[r, 1], b[r, 1]].T, c="#2e8b57", lw=2.2)
        ax.scatter(pts[:, 0], pts[:, 1], s=1.2, c="#1f4fd1")
        ax.set_aspect("equal"); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{p} {t} ({len(pts)} readings)", fontsize=10)
    axs[0, 0].plot([], [], c="#cfcfcf", label="road arcs")
    axs[0, 0].plot([], [], c="#2e8b57", lw=2, label="reference route")
    axs[0, 0].scatter([], [], c="#1f4fd1", s=8, label="GPS")
    axs[0, 0].legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(out / "eda_trajectories.png", dpi=120); plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    run(**vars(parser.parse_args()))

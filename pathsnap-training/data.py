"""Read the two development partitions and produce road candidate features."""
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree


K = 8
FEATURES = (
    "log_distance_m", "heading_cosine", "log_speed_mps", "log_segment_length_m",
    "projection_fraction", "distance_rank", "log_previous_distance_m",
    "log_next_distance_m",
)


def load_arrays(archive, member):
    path = member[:-6]
    arrays = []
    for suffix, dtype in ((".track", float), (".nodes", float),
                          (".arcs", int), (".route", int)):
        with archive.open(path + suffix) as file:
            arrays.append(np.loadtxt(file, dtype=dtype, ndmin=2))
    track, nodes, arcs, route = arrays
    if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2 or route.shape[1] != 1:
        raise ValueError(f"Unexpected schema: {member}")
    if np.any((arcs < 0) | (arcs >= len(nodes))) or np.any((route < 0) | (route >= len(arcs))):
        raise ValueError(f"Out-of-range index: {member}")
    return track, nodes, arcs, route.ravel()


def project(coordinates, origin):
    """Local equirectangular projection with distances in metres."""
    scale = np.array([np.cos(np.deg2rad(origin[1])), 1.0])
    return np.deg2rad(coordinates - origin) * 6371008.8 * scale


def segment_distance(points, starts, vectors):
    delta = points - starts
    norm2 = np.sum(vectors * vectors, axis=-1)
    fraction = np.clip(np.sum(delta * vectors, axis=-1) / np.maximum(norm2, 1e-12), 0, 1)
    return np.linalg.norm(delta - fraction[..., None] * vectors, axis=-1), fraction


def reference_labels(points, starts, vectors, route):
    """Find the minimum-distance chronological alignment along the supplied route."""
    route_starts, route_vectors = starts[route], vectors[route]
    distance, _ = segment_distance(points[:, None, :], route_starts, route_vectors)
    positions = np.arange(len(route), dtype=np.int32)
    previous = np.empty(distance.shape, dtype=np.int32)
    cost = distance[0]
    for row in range(1, len(points)):
        prefix_min = np.minimum.accumulate(cost)
        # Keep the first state attaining each prefix minimum.
        changed = np.r_[True, prefix_min[1:] < prefix_min[:-1]]
        previous[row] = np.maximum.accumulate(np.where(changed, positions, 0))
        cost = distance[row] + prefix_min
    aligned = np.empty(len(points), dtype=np.int32)
    aligned[-1] = np.argmin(cost)
    for row in range(len(points) - 1, 0, -1):
        aligned[row - 1] = previous[row, aligned[row]]
    return route[aligned], distance[np.arange(len(points)), aligned]


def nearest_candidates(points, starts, vectors, k=K):
    """Find exact nearest segments using midpoint bounds within length buckets."""
    lengths = np.linalg.norm(vectors, axis=1)
    centers = starts + vectors / 2
    seed = cKDTree(centers).query(points, k=k)[1].reshape(len(points), k)
    radius = segment_distance(points[:, None, :], starts[seed], vectors[seed])[0].max(axis=1)
    buckets = []
    length_bins = np.floor(np.log2(np.maximum(lengths, 1.0))).astype(int)
    for bucket in np.unique(length_bins):
        ids = np.flatnonzero(length_bins == bucket)
        tree = cKDTree(centers[ids])
        # A segment within radius has its midpoint within radius + half its length.
        matches = tree.query_ball_point(points, radius + lengths[ids].max() / 2 + 1e-7)
        buckets.append((ids, matches))
    result = np.empty((len(points), k), dtype=np.int32)
    for row, point in enumerate(points):
        ids = np.concatenate([group[indices[row]] for group, indices in buckets])
        distance, _ = segment_distance(point, starts[ids], vectors[ids])
        result[row] = ids[np.lexsort((ids, distance))[:k]]
    return result


def prepare_candidates(track, nodes, arcs):
    """Prediction inputs depend only on GPS and the road graph."""
    origin = np.mean(track[:, :2], axis=0)
    points = project(track[:, :2], origin)
    xy = project(nodes, origin)
    starts = xy[arcs[:, 0]]
    vectors = xy[arcs[:, 1]] - starts
    candidates = nearest_candidates(points, starts, vectors, min(K, len(arcs)))
    indices = np.arange(len(points))
    before, after = np.maximum(indices - 1, 0), np.minimum(indices + 1, len(points) - 1)
    movement = points[after] - points[before]
    elapsed = track[after, 2] - track[before, 2]
    movement_norm = np.linalg.norm(movement, axis=1)
    speed = np.divide(movement_norm, elapsed, out=np.zeros_like(elapsed), where=elapsed > 0)
    candidate_starts, candidate_vectors = starts[candidates], vectors[candidates]
    length = np.linalg.norm(candidate_vectors, axis=-1)
    heading = np.sum(movement[:, None, :] * candidate_vectors, axis=-1)
    heading /= np.maximum(length * movement_norm[:, None], 1e-12)
    distance, fraction = segment_distance(points[:, None, :], candidate_starts, candidate_vectors)
    previous, _ = segment_distance(points[before, None, :], candidate_starts, candidate_vectors)
    following, _ = segment_distance(points[after, None, :], candidate_starts, candidate_vectors)
    features = np.stack((
        np.log1p(distance), heading, np.broadcast_to(np.log1p(speed[:, None]), distance.shape),
        np.log1p(length), fraction, np.broadcast_to(np.arange(candidates.shape[1]), distance.shape),
        np.log1p(previous), np.log1p(following),
    ), axis=-1).astype(np.float32)
    return points, starts, vectors, candidates, features


def prepare_dataset(zip_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        memberships = {}
        for partition in ("train", "student_test"):
            members = sorted(n for n in archive.namelist() if re.fullmatch(
                rf"[^/]+/{partition}/([^/]+)/\1\.track", n))
            memberships[partition] = {Path(n).stem for n in members}
            expected = 80 if partition == "train" else 10
            if len(members) != expected or len(memberships[partition]) != expected:
                raise ValueError(f"{partition}: expected {expected} unique complete trips")
            if partition == "student_test" and memberships["train"] & memberships[partition]:
                raise ValueError("Trip overlap between partitions")
            (destination / partition).mkdir(exist_ok=True)
            for member in members:
                trip_id = Path(member).stem
                track, nodes, arcs, route = load_arrays(archive, member)
                valid = np.isfinite(track).all(axis=1) & (abs(track[:, 0]) <= 180) & (abs(track[:, 1]) <= 90)
                track = track[valid]
                if not len(track) or not np.isfinite(nodes).all() or not len(route) or not len(arcs):
                    raise ValueError(f"Cannot process {member}")
                reordered = bool(np.any(np.diff(track[:, 2]) < 0))
                track = track[np.argsort(track[:, 2], kind="stable")]
                points, starts, vectors, candidates, features = prepare_candidates(track, nodes, arcs)
                labels, alignment_distance = reference_labels(points, starts, vectors, route)
                np.savez_compressed(
                    destination / partition / f"{trip_id}.npz", x=features, candidates=candidates,
                    labels=labels, arcs=arcs, nodes=nodes, track=track, route=route,
                    alignment_distance=alignment_distance, invalid_rows=(~valid).sum(),
                    reordered=reordered,
                )
                print(partition, trip_id, len(track), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    prepare_dataset(args.zip, args.out)

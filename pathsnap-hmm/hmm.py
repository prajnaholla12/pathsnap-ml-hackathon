"""HMM + Viterbi decoding over the candidate arcs of pathsnap-training/data.py.

Hidden state at GPS reading i: one of its 8 candidate arcs.
Emission:   w_emit * log p(arc), p = classifier score normalised over the 8 candidates.
Transition: -|route distance - straight distance| / beta   (Newson & Krumm style)
  route distance    = rest of arc a + shortest path (end of a -> start of b) + part of b
  straight distance = distance between the two GPS readings
beta grows with the time gap, so a long gap tolerates a longer detour.
An unreachable pair gets a fixed penalty instead of zero probability, so one
bad pair (one-way street, cut-off network edge, GPS jump) can't kill the path.
"""
import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class DecoderParams:
    w_emit: float = 1.0
    beta: float = 5.0
    beta_gap: float = 1.0
    backward_factor: float = 3.0
    disconnect_logp: float = -40.0
    min_search_m: float = 400.0


class RoadGraph:
    def __init__(self, arcs, starts, vectors):
        self.arcs = arcs
        self.starts = starts
        self.vectors = vectors
        self.length = np.linalg.norm(vectors, axis=1)
        order = np.argsort(arcs[:, 0], kind="stable")
        node_count = int(arcs.max()) + 1
        self.first = np.searchsorted(arcs[order, 0], np.arange(node_count + 1))
        self.out_arcs = order
        self.cache = {}

    def distances_from(self, node, limit):
        hit = self.cache.get(node)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        best = {node: 0.0}
        queue = [(0.0, node)]
        while queue:
            d, u = heapq.heappop(queue)
            if d > best.get(u, np.inf) or d > limit:
                continue
            for k in self.out_arcs[self.first[u]:self.first[u + 1]]:
                v, nd = self.arcs[k, 1], d + self.length[k]
                if nd <= limit and nd < best.get(v, np.inf):
                    best[v] = nd
                    heapq.heappush(queue, (nd, v))
        self.cache[node] = (limit, best)
        return best

    def path(self, source, target, limit=3000.0):
        previous, best, queue = {source: -1}, {source: 0.0}, [(0.0, source)]
        while queue:
            d, u = heapq.heappop(queue)
            if u == target:
                break
            if d > best[u] or d > limit:
                continue
            for k in self.out_arcs[self.first[u]:self.first[u + 1]]:
                v, nd = self.arcs[k, 1], d + self.length[k]
                if nd < best.get(v, np.inf):
                    best[v], previous[v] = nd, k
                    heapq.heappush(queue, (nd, v))
        if target not in previous:
            return None
        arcs, u = [], target
        while u != source:
            k = previous[u]
            arcs.append(int(k))
            u = self.arcs[k, 0]
        return arcs[::-1]


def projection_fraction(points, starts, vectors):
    delta = points - starts
    norm2 = np.maximum(np.sum(vectors * vectors, axis=-1), 1e-12)
    return np.clip(np.sum(delta * vectors, axis=-1) / norm2, 0, 1)


def viterbi(points, times, candidates, probabilities, graph, params=DecoderParams()):
    """Returns the decoded arc per GPS reading and the number of forced breaks."""
    p = params
    n, k = candidates.shape
    fraction = projection_fraction(points[:, None, :], graph.starts[candidates], graph.vectors[candidates])
    emission = p.w_emit * np.log(np.clip(probabilities, 1e-6, 1.0))
    score = emission[0].copy()
    back = np.zeros((n, k), dtype=np.int16)
    breaks = 0
    for i in range(1, n):
        dt = max(times[i] - times[i - 1], 1.0)
        straight = float(np.linalg.norm(points[i] - points[i - 1]))
        beta = p.beta + p.beta_gap * (dt - 1.0)
        limit = max(p.min_search_m, 3.0 * straight + 200.0)
        trans = np.empty((k, k))
        for a in range(k):
            arc_a = candidates[i - 1, a]
            length_a = graph.length[arc_a]
            reach = None
            for b in range(k):
                arc_b = candidates[i, b]
                if arc_b == arc_a:
                    move = (fraction[i, b] - fraction[i - 1, a]) * length_a
                    route = move if move >= 0 else -move * p.backward_factor
                else:
                    if reach is None:
                        reach = graph.distances_from(graph.arcs[arc_a, 1], limit)
                    between = reach.get(graph.arcs[arc_b, 0])
                    if between is None:
                        trans[a, b] = p.disconnect_logp
                        continue
                    route = (1 - fraction[i - 1, a]) * length_a + between + fraction[i, b] * graph.length[arc_b]
                trans[a, b] = -abs(route - straight) / beta
        total = score[:, None] + trans
        best_prev = np.argmax(total, axis=0)
        new = total[best_prev, np.arange(k)] + emission[i]
        top = int(np.argmax(new))
        if trans[best_prev[top], top] <= p.disconnect_logp:
            breaks += 1
        back[i] = best_prev
        score = new - new.max()
    state = np.empty(n, dtype=np.int64)
    state[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        state[i - 1] = back[i, state[i]]
    return candidates[np.arange(n), state], breaks


def stitch(graph, point_arcs, limit=3000.0):
    """Collapse per-reading arcs into a route; join non-adjacent arcs by a shortest path."""
    runs = point_arcs[np.r_[True, point_arcs[1:] != point_arcs[:-1]]].tolist()
    route, unfilled = [runs[0]], 0
    for arc in runs[1:]:
        if graph.arcs[route[-1], 1] != graph.arcs[arc, 0]:
            fill = graph.path(graph.arcs[route[-1], 1], graph.arcs[arc, 0], limit)
            if fill is None:
                unfilled += 1
            else:
                route.extend(fill)
        route.append(arc)
    return route, unfilled


def route_overlap(predicted, reference, length):
    """Length-weighted precision, recall and F1 of the predicted vs reference arc sets."""
    p, r = set(predicted), set(int(a) for a in reference)
    lp, lr, both = (sum(length[a] for a in s) for s in (p, r, p & r))
    precision = both / lp if lp else 0.0
    recall = both / lr if lr else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

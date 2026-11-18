"""Folium map helpers."""
import folium
import numpy as np
from scipy.spatial import cKDTree


def _network_layer(nodes, arcs, ids, name="Road network", color="#9a9a9a"):
    coords = [[[float(nodes[f, 0]), float(nodes[f, 1])], [float(nodes[t, 0]), float(nodes[t, 1])]]
              for f, t in arcs[ids]]
    geo = {"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": coords}}
    return folium.GeoJson(geo, name=name, style_function=lambda _f: {"color": color, "weight": 1, "opacity": 0.6})


def _route_layer(nodes, arcs, sequence, name, color, weight, dash=None):
    """Draw an arc sequence; a new polyline starts wherever consecutive arcs don't touch."""
    group, run, last = folium.FeatureGroup(name=name), [], None
    for a in sequence:
        f, t = arcs[a]
        if last is not None and f != last and run:
            folium.PolyLine(run, color=color, weight=weight, opacity=0.9, dash_array=dash).add_to(group)
            run = []
        if not run:
            run.append((nodes[f, 1], nodes[f, 0]))
        run.append((nodes[t, 1], nodes[t, 0]))
        last = t
    if run:
        folium.PolyLine(run, color=color, weight=weight, opacity=0.9, dash_array=dash).add_to(group)
    return group


def _gps_layer(track):
    group = folium.FeatureGroup(name="GPS readings")
    step = max(1, len(track) // 3000)
    for i in range(0, len(track), step):
        lon, lat, t = track[i]
        folium.CircleMarker((lat, lon), radius=2, color="#1f4fd1", weight=0, fill=True, fill_opacity=0.8,
                            tooltip=f"#{i}  t={t:.0f}s").add_to(group)
    return group


def _nearby(track, nodes, arcs, pad_deg=0.003, limit=25000):
    lo, hi = track[:, :2].min(0) - pad_deg, track[:, :2].max(0) + pad_deg
    a, b = nodes[arcs[:, 0]], nodes[arcs[:, 1]]
    ids = np.flatnonzero(((a >= lo) & (a <= hi)).all(1) | ((b >= lo) & (b <= hi)).all(1))
    if len(ids) > limit:
        d, _ = cKDTree(track[:, :2]).query((a[ids] + b[ids]) / 2)
        ids = ids[np.argsort(d)[:limit]]
    return ids


def _legend(m, title, items):
    rows = "".join(
        f'<div><span style="display:inline-block;margin-right:6px;vertical-align:middle;background:{c};'
        + ("width:8px;height:8px;border-radius:50%;" if dot else "width:18px;height:4px;") + f'"></span>{t}</div>'
        for t, c, dot in items)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
        f'border:1px solid #999;border-radius:4px;font:12px sans-serif;"><b>{title}</b>{rows}</div>'))


def _base(track):
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[track[:, 1].min(), track[:, 0].min()], [track[:, 1].max(), track[:, 0].max()]])
    return m


def _dedupe(arcs):
    arcs = np.asarray(arcs)
    return arcs[np.r_[True, arcs[1:] != arcs[:-1]]] if len(arcs) else arcs


def trip_map(path, title, track, nodes, arcs, reference=None, classifier=None, predicted=None):
    m = _base(track)
    _network_layer(nodes, arcs, _nearby(track, nodes, arcs)).add_to(m)
    items = [("Road network", "#9a9a9a", False)]
    if reference is not None:
        _route_layer(nodes, arcs, reference, "Reference route", "#2e8b57", 7 if predicted is not None else 5).add_to(m)
        items.append(("Reference route", "#2e8b57", False))
    if classifier is not None:
        _route_layer(nodes, arcs, _dedupe(classifier), "Classifier only (no HMM)", "#e67e22", 3, "4 6").add_to(m)
        items.append(("Classifier only", "#e67e22", False))
    if predicted is not None:
        _route_layer(nodes, arcs, predicted, "Predicted route (HMM)", "#d62728", 3).add_to(m)
        items.append(("Predicted route (HMM)", "#d62728", False))
    _gps_layer(track).add_to(m)
    items.append(("GPS readings", "#1f4fd1", True))
    folium.Marker([track[0, 1], track[0, 0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([track[-1, 1], track[-1, 0]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _legend(m, title, items)
    m.save(path)


def locations_map(path, inventory):
    m = folium.Map(location=[inventory.lat_center.mean(), inventory.lon_center.mean()], zoom_start=2,
                   tiles="OpenStreetMap")
    colours = {"train": "#3b6ea8", "student_test": "#d9822b"}
    for partition, group in inventory.groupby("partition"):
        layer = folium.FeatureGroup(name=partition)
        for _, r in group.iterrows():
            folium.CircleMarker((r.lat_center, r.lon_center), radius=5, color=colours[partition], fill=True,
                                tooltip=f"{r.trip_id} ({partition}) {r.observations} readings").add_to(layer)
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _legend(m, "Trip centres", [(k, v, True) for k, v in colours.items()])
    m.save(path)

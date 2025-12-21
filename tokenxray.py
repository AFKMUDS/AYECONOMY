import json
import argparse
from pathlib import Path

def explore_brain(file_path, search_token):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    
    found = False
    print(f"\n--- Results for '{search_token}' ---")
    
    for module_name, module_data in data.get("modules", {}).items():
        for token_info in module_data.get("top_tokens", []):
            if token_info["value"] == search_token:
                found = True
                print(f"Module: {module_name} | Importance: {token_info['importance']:.4f}")
                
                print("  [Direct Connections]:")
                for conn in token_info.get("connections", []):
                    print(f"    -> {conn['token']} (Strength: {conn['strength']:.4f})")
                
                print("  [Wormholes (The Jumps)]:")
                for worm in token_info.get("wormholes", []):
                    print(f"    ~> {worm['token']} (Dist: {worm['distance']}, Strength: {worm['strength']:.4f})")
                print("-" * 30)
    
    if not found:
        print(f"Token '{search_token}' not found in the currently loaded brain slice.")

def list_tokens_by_importance(file_path, threshold, mode='above', module=None, limit=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Expected 'modules' to be an object/dict")

    wanted_module = (module or "").strip() or None
    rows = []

    for module_name, module_data in modules.items():
        if wanted_module and module_name != wanted_module:
            continue
        top_tokens = (module_data or {}).get("top_tokens", [])
        if not isinstance(top_tokens, list):
            continue
        for token_info in top_tokens:
            if not isinstance(token_info, dict):
                continue
            token_value = token_info.get("value")
            if token_value is None:
                continue
            try:
                importance = float(token_info.get("importance", 0.0))
            except Exception:
                continue

            if mode == 'above' and importance > threshold:
                rows.append((module_name, str(token_value), importance))
            elif mode == 'below' and importance < threshold:
                rows.append((module_name, str(token_value), importance))

    rows.sort(key=lambda r: r[2], reverse=(mode == 'above'))
    if limit is not None and limit > 0:
        rows = rows[:limit]

    print(f"\n--- Tokens with importance {mode} {threshold} ---")
    if wanted_module:
        print(f"Module: {wanted_module}")
    for i, (m, v, imp) in enumerate(rows, start=1):
        print(f"{i}. [{m}] {v} (importance={imp:.6f})")
    if not rows:
        print("No tokens matched.")

def _clamp(x, lo, hi):
    if lo is not None and x < lo:
        return lo
    if hi is not None and x > hi:
        return hi
    return x

def scale_importance_by_ratio(
    file_path,
    threshold,
    ratio,
    mode='above',
    module=None,
    token=None,
    clamp_min=None,
    clamp_max=None,
    out_path=None,
    in_place=False,
):
    in_path = Path(file_path)
    with in_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Expected 'modules' to be an object/dict")

    wanted_module = (module or "").strip() or None
    wanted_token = (token or "").strip() or None

    visited = 0
    modified = 0
    for module_name, module_data in modules.items():
        if wanted_module and module_name != wanted_module:
            continue
        top_tokens = (module_data or {}).get("top_tokens", [])
        if not isinstance(top_tokens, list):
            continue
        for token_info in top_tokens:
            if not isinstance(token_info, dict):
                continue
            token_value = token_info.get("value")
            if token_value is None:
                continue
            if wanted_token and str(token_value) != str(wanted_token):
                continue
            try:
                current = float(token_info.get("importance", 0.0))
            except Exception:
                continue
            visited += 1

            should_adjust = (mode == 'above' and current > threshold) or (mode == 'below' and current < threshold)
            if not should_adjust:
                continue

            new_val = _clamp(current * ratio, clamp_min, clamp_max)
            if new_val != current:
                token_info["importance"] = new_val
                modified += 1

    if in_place:
        write_path = in_path
    else:
        if not out_path:
            raise ValueError("Provide --out or use --in-place")
        write_path = Path(out_path)

    with write_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Importance scaling complete. {modified} tokens modified out of {visited} visited.")
    print(f"Wrote: {write_path}")

def add_or_replace_wormhole(
    file_path,
    module,
    src,
    dst,
    strength,
    distance=2,
    out_path=None,
    in_place=False,
):
    in_path = Path(file_path)
    with in_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Expected 'modules' to be an object/dict")

    if module not in modules:
        raise ValueError(f"Module '{module}' not found")

    module_data = modules.get(module) or {}
    top_tokens = module_data.get("top_tokens", [])
    if not isinstance(top_tokens, list):
        raise ValueError(f"Expected modules['{module}'].top_tokens to be a list")

    src_obj = None
    for token_info in top_tokens:
        if isinstance(token_info, dict) and token_info.get("value") == src:
            src_obj = token_info
            break
    if src_obj is None:
        raise ValueError(f"Source token '{src}' not found in module '{module}' top_tokens")

    wormholes = src_obj.get("wormholes")
    if wormholes is None:
        wormholes = []
        src_obj["wormholes"] = wormholes
    if not isinstance(wormholes, list):
        raise ValueError("Expected 'wormholes' to be a list")

    replaced = False
    for wh in wormholes:
        if isinstance(wh, dict) and wh.get("token") == dst:
            wh["distance"] = int(distance)
            wh["strength"] = float(strength)
            replaced = True
            break
    if not replaced:
        wormholes.append({"token": dst, "distance": int(distance), "strength": float(strength)})

    if in_place:
        write_path = in_path
    else:
        if not out_path:
            raise ValueError("Provide --out or use --in-place")
        write_path = Path(out_path)

    with write_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    action = "Replaced" if replaced else "Added"
    print(f"{action} wormhole: [{module}] {src} ~> {dst} (dist={int(distance)}, strength={float(strength)})")
    print(f"Wrote: {write_path}")

def normalize_brain_by_ratio(
    file_path,
    threshold,
    ratio,
    mode='above',
    module=None,
    src_token=None,
    dst_token=None,
    kinds=("connections", "wormholes"),
    clamp_min=None,
    clamp_max=None,
    out_path=None,
    in_place=False,
):
    in_path = Path(file_path)
    with in_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Expected 'modules' to be an object/dict")

    wanted_module = (module or "").strip() or None
    kinds = tuple([k for k in kinds if k in ("connections", "wormholes")])
    if not kinds:
        raise ValueError("kinds must include 'connections' and/or 'wormholes'")

    visited = 0
    modified = 0
    for module_name, module_data in modules.items():
        if wanted_module and module_name != wanted_module:
            continue
        top_tokens = (module_data or {}).get("top_tokens", [])
        if not isinstance(top_tokens, list):
            continue
        for token_info in top_tokens:
            if not isinstance(token_info, dict):
                continue
            token_value = token_info.get("value")
            if src_token is not None and str(token_value) != str(src_token):
                continue
            for kind in kinds:
                items = token_info.get(kind, [])
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict) or "strength" not in item:
                        continue
                    if dst_token is not None:
                        item_dst = item.get("token")
                        if item_dst is None or str(item_dst) != str(dst_token):
                            continue
                    try:
                        current = float(item["strength"])
                    except Exception:
                        continue
                    visited += 1

                    should_adjust = (mode == 'above' and current > threshold) or (mode == 'below' and current < threshold)
                    if not should_adjust:
                        continue

                    new_val = _clamp(current * ratio, clamp_min, clamp_max)
                    if new_val != current:
                        item["strength"] = new_val
                        modified += 1

    if in_place:
        write_path = in_path
    else:
        if not out_path:
            raise ValueError("Provide --out or use --in-place")
        write_path = Path(out_path)

    with write_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Ratio adjustment complete. {modified} strengths modified out of {visited} visited.")
    print(f"Wrote: {write_path}")

def delete_edge(
    file_path,
    module,
    src,
    dst,
    kind,
    out_path=None,
    in_place=False,
):
    if kind not in ("connections", "wormholes"):
        raise ValueError("kind must be 'connections' or 'wormholes'")

    in_path = Path(file_path)
    with in_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Expected 'modules' to be an object/dict")

    if module not in modules:
        raise ValueError(f"Module '{module}' not found")

    top_tokens = (modules[module] or {}).get("top_tokens", [])
    if not isinstance(top_tokens, list):
        raise ValueError(f"Expected modules['{module}'].top_tokens to be a list")

    src_obj = None
    for token_info in top_tokens:
        if isinstance(token_info, dict) and token_info.get("value") == src:
            src_obj = token_info
            break
    if src_obj is None:
        raise ValueError(f"Source token '{src}' not found in module '{module}' top_tokens")

    items = src_obj.get(kind, [])
    if items is None:
        items = []
    if not isinstance(items, list):
        raise ValueError(f"Expected '{kind}' to be a list")

    before = len(items)
    items = [x for x in items if not (isinstance(x, dict) and x.get("token") == dst)]
    after = len(items)
    src_obj[kind] = items

    removed = before - after

    if in_place:
        write_path = in_path
    else:
        if not out_path:
            raise ValueError("Provide --out or use --in-place")
        write_path = Path(out_path)

    with write_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Removed {removed} {kind} edge(s): [{module}] {src} -> {dst}")
    print(f"Wrote: {write_path}")

def copy_token_between_modules(
    file_path,
    token,
    from_module,
    to_module,
    out_path=None,
    in_place=False,
    overwrite=False,
):
    in_path = Path(file_path)
    with in_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError("Expected 'modules' to be an object/dict")

    if from_module not in modules:
        raise ValueError(f"Module '{from_module}' not found")

    if to_module not in modules:
        raise ValueError(f"Module '{to_module}' not found")

    top_tokens = (modules[from_module] or {}).get("top_tokens", [])
    if not isinstance(top_tokens, list):
        raise ValueError(f"Expected modules['{from_module}'].top_tokens to be a list")

    src_obj = None
    for token_info in top_tokens:
        if isinstance(token_info, dict) and token_info.get("value") == token:
            src_obj = token_info
            break
    if src_obj is None:
        raise ValueError(f"Token '{token}' not found in module '{from_module}' top_tokens")

    to_module_data = modules.get(to_module) or {}
    top_tokens = (to_module_data or {}).get("top_tokens", [])
    if not isinstance(top_tokens, list):
        raise ValueError(f"Expected modules['{to_module}'].top_tokens to be a list")

    dst_obj = None
    for token_info in top_tokens:
        if isinstance(token_info, dict) and token_info.get("value") == token:
            dst_obj = token_info
            break

    if dst_obj is not None and not overwrite:
        raise ValueError(f"Token '{token}' already exists in module '{to_module}'")

    if dst_obj is None:
        dst_obj = {"value": token}
        top_tokens.append(dst_obj)

    for key, value in src_obj.items():
        if key not in dst_obj:
            dst_obj[key] = value

    if in_place:
        write_path = in_path
    else:
        if not out_path:
            raise ValueError("Provide --out or use --in-place")
        write_path = Path(out_path)

    with write_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Copied token '{token}' from [{from_module}] to [{to_module}]")
    print(f"Wrote: {write_path}")

def main():
    parser = argparse.ArgumentParser(description="Token X-Ray: inspect and ratio-normalize trained_model.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="Print detailed info for a specific token (exact match)")
    p_search.add_argument("model", help="Path to trained_model.json")
    p_search.add_argument("token", help="Token value to search for (exact)")

    p_list = sub.add_parser("list", help="List tokens above/below an importance threshold")
    p_list.add_argument("model", help="Path to trained_model.json")
    p_list.add_argument("--threshold", type=float, required=True, help="Importance threshold")
    p_list.add_argument("--mode", choices=["above", "below"], default="above")
    p_list.add_argument("--module", default="", help="Optional module name filter")
    p_list.add_argument("--limit", type=int, default=50, help="Max tokens to print")

    p_norm = sub.add_parser("normalize", help="Multiply strengths above/below a threshold by a ratio")
    p_norm.add_argument("model", help="Path to trained_model.json")
    p_norm.add_argument("--threshold", type=float, required=True, help="Strength threshold")
    p_norm.add_argument("--ratio", type=float, required=True, help="Multiplier ratio")
    p_norm.add_argument("--mode", choices=["above", "below"], default="above")
    p_norm.add_argument("--module", default="", help="Optional module name filter")
    p_norm.add_argument("--src", default="", help="Only adjust edges originating from this token value")
    p_norm.add_argument("--dst", default="", help="Only adjust edges pointing to this token value")
    p_norm.add_argument(
        "--kinds",
        default="connections,wormholes",
        help="Comma-separated list: connections,wormholes (default: both)",
    )
    p_norm.add_argument("--clamp-min", type=float, default=None, help="Optional minimum strength after scaling")
    p_norm.add_argument("--clamp-max", type=float, default=None, help="Optional maximum strength after scaling")
    p_norm.add_argument("--out", default="", help="Write to a new JSON file")
    p_norm.add_argument("--in-place", action="store_true", help="Modify the input file in place")

    p_mass = sub.add_parser("mass", help="Scale token importance (mass) above/below a threshold by a ratio")
    p_mass.add_argument("model", help="Path to trained_model.json")
    p_mass.add_argument("--threshold", type=float, required=True, help="Importance threshold")
    p_mass.add_argument("--ratio", type=float, required=True, help="Multiplier ratio")
    p_mass.add_argument("--mode", choices=["above", "below"], default="above")
    p_mass.add_argument("--module", default="", help="Optional module name filter")
    p_mass.add_argument("--token", default="", help="Optional exact token filter")
    p_mass.add_argument("--clamp-min", type=float, default=None, help="Optional minimum importance after scaling")
    p_mass.add_argument("--clamp-max", type=float, default=None, help="Optional maximum importance after scaling")
    p_mass.add_argument("--out", default="", help="Write to a new JSON file")
    p_mass.add_argument("--in-place", action="store_true", help="Modify the input file in place")

    p_wh = sub.add_parser("wormhole", help="Add or replace a wormhole door: src ~> dst")
    p_wh.add_argument("model", help="Path to trained_model.json")
    p_wh.add_argument("--module", required=True, help="Module name containing the source token")
    p_wh.add_argument("--src", required=True, help="Source token value")
    p_wh.add_argument("--dst", required=True, help="Destination token value")
    p_wh.add_argument("--strength", type=float, required=True, help="Wormhole strength")
    p_wh.add_argument("--distance", type=int, default=2, help="Wormhole distance (default: 2)")
    p_wh.add_argument("--out", default="", help="Write to a new JSON file")
    p_wh.add_argument("--in-place", action="store_true", help="Modify the input file in place")

    p_copy = sub.add_parser("copy-token", help="Copy a token from one module to another")
    p_copy.add_argument("model", help="Path to trained_model.json")
    p_copy.add_argument("--token", required=True, help="Token value to copy")
    p_copy.add_argument("--from-module", required=True, help="Module to copy from")
    p_copy.add_argument("--to-module", required=True, help="Module to copy to")
    p_copy.add_argument("--out", default="", help="Write to a new JSON file")
    p_copy.add_argument("--in-place", action="store_true", help="Modify the input file in place")
    p_copy.add_argument("--overwrite", action="store_true", help="Overwrite if token already exists in target module")

    p_del = sub.add_parser("delete-edge", help="Delete a specific edge (connection or wormhole) from a source token")
    p_del.add_argument("model", help="Path to trained_model.json")
    p_del.add_argument("--module", required=True, help="Module containing the source token")
    p_del.add_argument("--src", required=True, help="Source token value")
    p_del.add_argument("--dst", required=True, help="Destination token value")
    p_del.add_argument("--kind", choices=["connections", "wormholes"], required=True)
    p_del.add_argument("--out", default="", help="Write to a new JSON file")
    p_del.add_argument("--in-place", action="store_true", help="Modify the input file in place")

    args = parser.parse_args()

    if args.cmd == "search":
        explore_brain(args.model, args.token)
        return

    if args.cmd == "list":
        list_tokens_by_importance(
            file_path=args.model,
            threshold=args.threshold,
            mode=args.mode,
            module=args.module,
            limit=args.limit,
        )
        return

    if args.cmd == "normalize":
        kinds = tuple([k.strip() for k in (args.kinds or "").split(",") if k.strip()])
        normalize_brain_by_ratio(
            file_path=args.model,
            threshold=args.threshold,
            ratio=args.ratio,
            mode=args.mode,
            module=args.module,
            src_token=(args.src.strip() or None),
            dst_token=(args.dst.strip() or None),
            kinds=kinds,
            clamp_min=args.clamp_min,
            clamp_max=args.clamp_max,
            out_path=(args.out or None),
            in_place=args.in_place,
        )
        return

    if args.cmd == "mass":
        scale_importance_by_ratio(
            file_path=args.model,
            threshold=args.threshold,
            ratio=args.ratio,
            mode=args.mode,
            module=args.module,
            token=args.token,
            clamp_min=args.clamp_min,
            clamp_max=args.clamp_max,
            out_path=(args.out or None),
            in_place=args.in_place,
        )
        return

    if args.cmd == "wormhole":
        add_or_replace_wormhole(
            file_path=args.model,
            module=args.module,
            src=args.src,
            dst=args.dst,
            strength=args.strength,
            distance=args.distance,
            out_path=(args.out or None),
            in_place=args.in_place,
        )
        return

    if args.cmd == "copy-token":
        copy_token_between_modules(
            file_path=args.model,
            token=args.token,
            from_module=args.from_module,
            to_module=args.to_module,
            out_path=(args.out or None),
            in_place=args.in_place,
            overwrite=args.overwrite,
        )
        return

    if args.cmd == "delete-edge":
        delete_edge(
            file_path=args.model,
            module=args.module,
            src=args.src,
            dst=args.dst,
            kind=args.kind,
            out_path=(args.out or None),
            in_place=args.in_place,
        )
        return

if __name__ == "__main__":
    main()
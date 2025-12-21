import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Pretty-print JSON and extract first N lines.")
    parser.add_argument("json_path", help="Path to the JSON file (e.g. trained_model.json)")
    parser.add_argument("-n", "--lines", type=int, default=10, help="How many lines to output (default: 10)")
    parser.add_argument("--chars", type=int, default=0, help="If > 0, print the first N raw characters from the file (no JSON parsing)")
    parser.add_argument("--tokens", type=int, default=0, help="If > 0, print the first N top_tokens entries (token strings) instead of lines")
    parser.add_argument("--module", default="", help="Optional module name to extract tokens from (default: all modules)")
    parser.add_argument("-o", "--out", help="Optional output text file to write the lines to")
    parser.add_argument("--indent", type=int, default=2, help="Indent level for pretty-printing (default: 2)")
    args = parser.parse_args()

    json_path = Path(args.json_path)

    if args.chars and args.chars > 0:
        raw = json_path.read_text(encoding="utf-8")
        output = raw[: args.chars]
        print(output)
        if args.out:
            Path(args.out).write_text(output + "\n", encoding="utf-8")
        return

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if args.tokens and args.tokens > 0:
        modules = data.get("modules", {})
        if not isinstance(modules, dict):
            raise ValueError("Expected 'modules' to be an object/dict")

        wanted_module = args.module.strip()
        collected = []
        for module_name, module_data in modules.items():
            if wanted_module and module_name != wanted_module:
                continue
            if not isinstance(module_data, dict):
                continue
            top_tokens = module_data.get("top_tokens", [])
            if not isinstance(top_tokens, list):
                continue
            for entry in top_tokens:
                if isinstance(entry, dict) and isinstance(entry.get("token"), str):
                    collected.append((module_name, entry["token"]))
                if len(collected) >= args.tokens:
                    break
            if len(collected) >= args.tokens:
                break

        output_lines = []
        for i, (module_name, token) in enumerate(collected, start=1):
            output_lines.append(f"{i}. [{module_name}] {token}")
        output = "\n".join(output_lines)
    else:
        pretty = json.dumps(data, indent=args.indent, ensure_ascii=False)
        pretty_lines = pretty.splitlines()

        head = pretty_lines[:args.lines]
        output = "\n".join(head)

    print(output)

    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()

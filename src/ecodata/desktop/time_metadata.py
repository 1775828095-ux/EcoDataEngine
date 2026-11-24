"""Time support from declared metadata and local model controls."""
from pathlib import PurePosixPath
import re


def cell_methods(value, axes):
    """Keep the order of CF time operations, separate from spatial operations."""
    text = re.sub(r"\([^)]*\)", "", str(value))
    operations = []
    pattern = r"((?:[\w.-]+\s*:\s*)+)([a-z_]+)(?:\s+((?:within|over)\s+(?:days|years)))?"
    for names, method, qualifier in re.findall(pattern, text, re.I):
        if any(name.strip() in axes for name in names.split(":") if name.strip()):
            operations.append(" ".join((method + " " + qualifier).lower().split()))
    return operations


def declared_parameters(text, keys):
    if not text:
        return {}
    lines = [line.split("!", 1)[0].lower().split() for line in text.splitlines()]
    lines = [line for line in lines if line]
    for header, values in zip(lines, lines[1:]):
        if set(keys) <= set(header) and len(header) == len(values):
            try:
                return {key: int(values[header.index(key)]) for key in keys}
            except ValueError:
                return {}
    return {}


def attach_swat_print_periods(rows, docs):
    """Attach declared years from the matching scenario, not measured output coverage."""
    texts = dict(docs)
    controls = {}
    single_annual = set()
    for name, text in docs:
        path = PurePosixPath(name)
        if path.name.casefold() != "file.cio":
            continue
        groups = [line.split("!", 1)[0].split() for line in text.splitlines()]
        simulation = next((line for line in groups if len(line) >= 3 and line[0].lower() == "simulation"), [])
        if not simulation or any(PurePosixPath(ref).name != ref or "\\" in ref or ":" in ref for ref in simulation[1:3]):
            continue
        period = declared_parameters(texts.get(str(path.parent / simulation[1])), ("yrc_start", "yrc_end"))
        print_text = texts.get(str(path.parent / simulation[2]))
        printing = declared_parameters(print_text, ("nyskip", "yrc_start", "yrc_end"))
        if not period or not printing or printing["nyskip"] < 0 or not 0 < period["yrc_start"] <= period["yrc_end"] <= 9999:
            continue
        start = max(period["yrc_start"] + printing["nyskip"], printing["yrc_start"])
        end = min(period["yrc_end"], printing["yrc_end"] or period["yrc_end"])
        if start <= end:
            controls[path.parent] = dict(start_year=start, end_year=end, basis="swat_simulation_and_print_controls")
            if declared_parameters(print_text, ("aa_int_cnt",)).get("aa_int_cnt") == 0:
                single_annual.add(path.parent)

    for row in rows:
        if row["format"] == "SWAT+ 输出表":
            parent = PurePosixPath(row["name"]).parent
            directory = parent.parent if parent.name.casefold() == "output" else parent
            if directory in controls:
                row["declared_output_print_period"] = controls[directory]
                if row.get("declared_output_period") == "average_annual" and directory in single_annual:
                    row["single_average_annual_window"] = True

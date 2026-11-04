"""Capture an observed, fully pinned dependency closure for this CPU revision.

The lock records installed distributions, not unverified wheel hashes. It does
not invent a full lock for the earlier WSL machine or the historical v7.9 run.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
ROOTS = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]


def main(destination=ROOT):
    out = destination / "provenance"
    out.mkdir(exist_ok=True)
    found = {}
    todo = ROOTS.copy()
    while todo:
        name = canonicalize_name(todo.pop())
        if name in found:
            continue
        dist = metadata.distribution(name)
        dependencies = []
        for spec in dist.requires or []:
            requirement = Requirement(spec)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                dep = canonicalize_name(requirement.name)
                dependencies.append({"name": dep, "specifier": str(requirement.specifier)})
                todo.append(dep)
        records = {}
        for item in dist.files or []:
            if str(item).endswith((".dist-info/METADATA", ".dist-info/RECORD")):
                path = Path(dist.locate_file(item))
                records[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        found[name] = {"version": dist.version, "requires_in_this_environment": dependencies,
                       "installed_distribution_metadata_sha256": records}
    for name, record in found.items():
        for dep in record["requires_in_this_environment"]:
            if dep["specifier"]:
                requirement = Requirement(dep["name"] + dep["specifier"])
                if found[dep["name"]]["version"] not in requirement.specifier:
                    raise RuntimeError(f"Unsatisfied installed requirement: {name}: {requirement}")
    lock = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "python_implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(), "roots": ROOTS,
        "packages": dict(sorted(found.items())),
        "scope": "Complete installed runtime dependency closure of the five experiment/plotting roots under this Python/platform and default extras. Version-pinned install list plus installed METADATA/RECORD hashes.",
        "limitations": "No downloaded wheel hashes or universal cross-platform solver lock is claimed. Historical WSL has only its recorded Python and four principal versions; this does not retrospectively reconstruct that environment. Document-authoring tools are outside the experimental runtime lock."
    }
    (out / "dependency_lock.json").write_text(json.dumps(lock, indent=2)+"\n")
    lines = ["# Observed CPU runtime closure for Python 3.12 on Linux.",
             "# See provenance/dependency_lock.json for scope and installed metadata hashes.",
             "# Fully pinned versions; this is not a wheel-hash lock."]
    lines += [f"{name}=={record['version']}" for name, record in sorted(found.items())]
    (destination / "requirements-experiments.lock").write_text("\n".join(lines)+"\n")
    inventory = sorted({(canonicalize_name(d.metadata['Name']), d.version) for d in metadata.distributions()})
    (out / "full_environment_inventory.json").write_text(json.dumps({"python": sys.version, "platform": platform.platform(),
        "packages": [{"name": n, "version": v} for n, v in inventory],
        "role": "Observed environment inventory, not a request to install unrelated packages."}, indent=2)+"\n")
    print(json.dumps({"dependency_closure_packages": len(found), "full_environment_packages": len(inventory), "all_requirements_satisfied": True}))


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT)
    args=parser.parse_args()
    main(args.destination)

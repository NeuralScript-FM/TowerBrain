
"""
Tower Brain — Results Explainer

Reads controller outputs from the outputs/ folder and generates a
Markdown explanation of the simulation results.

Works with:
- A flat controller_summary.json
- A scenario-indexed controller_summary.json
- Controller KPI CSV files
- Controller history CSV files

Ollama is optional. Use --no-llm to generate a report without it.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_FILE = OUTPUT_DIR / "tower_brain_report.md"
SUMMARY_FILE = OUTPUT_DIR / "controller_summary.json"

DEFAULT_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------
# FILE HELPERS
# ---------------------------------------------------------

def read_json(path):
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: Could not read {path.name}: {exc}")
        return None


def read_csv(path):
    if not path or not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    except (OSError, csv.Error) as exc:
        print(f"Warning: Could not read {path.name}: {exc}")
        return []


def find_output_file(keyword):
    """Find the most relevant CSV matching a keyword."""
    if not OUTPUT_DIR.exists():
        return None

    matches = [
        path for path in OUTPUT_DIR.glob("*.csv")
        if keyword.lower() in path.stem.lower()
    ]

    if not matches:
        return None

    # Prefer files containing the current controller naming convention.
    matches.sort(
        key=lambda p: (
            "tower_brain" not in p.stem.lower(),
            p.stat().st_mtime
        ),
        reverse=False
    )
    return matches[0]


def to_number(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def format_number(value, decimals=2):
    number = to_number(value)
    if number is None:
        return "Not available"
    return f"{number:,.{decimals}f}"


# ---------------------------------------------------------
# SUMMARY NORMALIZATION
# ---------------------------------------------------------

def select_scenario_data(summary, scenario):
    """
    Accept both:
      {"scenario": "summer", "kpis": {...}}

    and:
      {"summer": {...}, "heatwave": {...}}
    """

    if not isinstance(summary, dict):
        return {}

    # Flat summary: use it directly.
    if "kpis" in summary or "scenario" in summary:
        return summary

    # Scenario-indexed summary.
    if scenario in summary:
        selected = summary[scenario]
        if isinstance(selected, dict):
            return selected

    # Case-insensitive scenario lookup.
    for key, value in summary.items():
        if str(key).lower() == scenario.lower():
            if isinstance(value, dict):
                return value

    # Do not crash just because the requested scenario key is absent.
    return summary


def normalize_kpis(data):
    """Extract KPI values from common summary structures."""

    if not isinstance(data, dict):
        return {}

    kpis = data.get("kpis", data)

    if not isinstance(kpis, dict):
        return {}

    # Some files wrap the values in another result object.
    if isinstance(kpis.get("results"), dict):
        kpis = kpis["results"]

    return kpis


# ---------------------------------------------------------
# CSV METRICS
# ---------------------------------------------------------

def metrics_from_history(rows):
    """Calculate useful metrics from the simulation history CSV."""

    if not rows:
        return {}

    metrics = {}

    energy_values = []
    power_values = []
    comfort_values = []

    for row in rows:
        power = to_number(row.get("total_kw"))

        if power is not None:
            power_values.append(power)

        # Each history row represents a 15-minute simulation step.
        if power is not None:
            energy_values.append(power * 0.25)

        comfort = to_number(row.get("comfort_pct"))
        if comfort is not None:
            comfort_values.append(comfort)

    if energy_values:
        metrics["total_kwh_from_history"] = sum(energy_values)

    if power_values:
        metrics["peak_kw_from_history"] = max(power_values)

    if comfort_values:
        metrics["average_comfort_pct_from_history"] = (
            sum(comfort_values) / len(comfort_values)
        )

    metrics["history_rows"] = len(rows)

    return metrics


def merge_metrics(kpis, history_metrics):
    """Use KPI values first, then fall back to history calculations."""

    result = dict(kpis)

    aliases = {
        "total_kwh": [
            "total_kwh",
            "energy_kwh",
            "total_energy_kwh"
        ],
        "peak_kw": [
            "peak_kw_15min",
            "peak_kw",
            "maximum_kw"
        ],
        "comfort_pct": [
            "comfort_pct",
            "comfort_percentage"
        ],
        "peak_day": ["peak_day"],
        "peak_time": ["peak_time"],
        "peak_w_per_m2": ["peak_w_per_m2"],
        "avg_weekday_kwh": [
            "avg_weekday_kwh",
            "average_weekday_kwh"
        ],
    }

    for standard_name, possible_names in aliases.items():
        if result.get(standard_name) not in (None, ""):
            continue

        for name in possible_names:
            if result.get(name) not in (None, ""):
                result[standard_name] = result[name]
                break

    if result.get("total_kwh") in (None, ""):
        result["total_kwh"] = history_metrics.get(
            "total_kwh_from_history"
        )

    if result.get("peak_kw") in (None, ""):
        result["peak_kw"] = history_metrics.get(
            "peak_kw_from_history"
        )

    if result.get("comfort_pct") in (None, ""):
        result["comfort_pct"] = history_metrics.get(
            "average_comfort_pct_from_history"
        )

    return result


# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------

def build_facts(scenario):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = read_json(SUMMARY_FILE)
    selected_summary = select_scenario_data(summary, scenario)
    kpis = normalize_kpis(selected_summary)

    history_file = find_output_file("history")
    kpi_file = find_output_file("kpi")

    # If JSON does not contain KPI values, try the KPI CSV.
    kpi_rows = read_csv(kpi_file)

    if not kpis and kpi_rows:
        matching_row = None

        for row in kpi_rows:
            row_scenario = row.get("scenario", "")
            if row_scenario.lower() == scenario.lower():
                matching_row = row
                break

        if matching_row is None:
            matching_row = kpi_rows[0]

        kpis = matching_row

    history_rows = read_csv(history_file)
    history_metrics = metrics_from_history(history_rows)
    kpis = merge_metrics(kpis, history_metrics)

    if not kpis and not history_rows:
        raise FileNotFoundError(
            "No controller results were found in outputs/. "
            "Run python controller.py first."
        )

    return {
        "scenario": scenario,
        "kpis": kpis,
        "history_file": history_file.name if history_file else None,
        "kpi_file": kpi_file.name if kpi_file else None,
        "history_metrics": history_metrics,
        "history_rows": len(history_rows),
    }


# ---------------------------------------------------------
# REPORT CONTENT
# ---------------------------------------------------------

def create_factual_report(facts):
    scenario = facts["scenario"]
    kpis = facts["kpis"]

    total_kwh = format_number(kpis.get("total_kwh"))
    peak_kw = format_number(kpis.get("peak_kw"))
    comfort_pct = format_number(kpis.get("comfort_pct"), 1)

    peak_day = kpis.get("peak_day", "Not available")
    peak_time = kpis.get("peak_time", "Not available")

    avg_weekday = format_number(
        kpis.get("avg_weekday_kwh")
    )

    peak_w_m2 = format_number(
        kpis.get("peak_w_per_m2")
    )

    lines = [
        "# Tower Brain — Simulation Results",
        "",
        f"## 1. Results",
        "",
        f"- **Scenario:** {scenario}",
        f"- **Total energy consumption:** {total_kwh} kWh",
        f"- **Peak electrical demand:** {peak_kw} kW",
        f"- **Comfort percentage:** {comfort_pct}%",
        f"- **Peak demand day:** {peak_day}",
        f"- **Peak demand time:** {peak_time}",
        f"- **Average daily energy:** {avg_weekday} kWh",
        f"- **Peak power density:** {peak_w_m2} W/m²",
        "",
        "## 2. Interpretation",
        "",
        (
            f"The simulation ran using the **{scenario}** scenario. "
            "The recorded energy consumption was "
            f"{total_kwh} kWh, with a maximum 15-minute electrical "
            f"demand of {peak_kw} kW."
        ),
        "",
        (
            f"The recorded comfort percentage was {comfort_pct}%. "
            "This indicates the proportion of occupied person-time "
            "within the simulator's defined comfort temperature range, "
            "when that KPI is supplied by the simulator."
        ),
        "",
        "## 3. Data availability",
        "",
        f"- History file: {facts['history_file'] or 'Not found'}",
        f"- KPI file: {facts['kpi_file'] or 'Not found'}",
        f"- History records: {facts['history_rows']}",
        "",
        "## 4. Important limitations",
        "",
        "- These results come from a simulation, not live building measurements.",
        "- Missing metrics are reported as unavailable rather than estimated.",
        "- Energy and demand results depend on the selected scenario, timetable, "
        "controller settings, and simulation parameters.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------
# OPTIONAL OLLAMA EXPLANATION
# ---------------------------------------------------------

def explain_with_ollama(model, factual_report):
    if requests is None:
        raise RuntimeError(
            "The requests package is not installed. "
            "Run: python -m pip install requests"
        )

    prompt = f"""
You are a technical building-energy simulation analyst.

Explain the following Tower Brain simulation report.

Rules:
- Use only the supplied report facts.
- Do not invent energy savings, percentages, comparisons, or causes.
- Clearly distinguish measured simulation outputs from interpretation.
- Explain the results in clear, professional language.
- Mention limitations where relevant.
- Do not claim that the system was tested in a real building.

REPORT:
{factual_report}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )

    response.raise_for_status()
    data = response.json()

    return data.get("response", "").strip()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Tower Brain simulation results explanation."
    )

    parser.add_argument(
        "--scenario",
        default="summer",
        choices=["summer", "heatwave", "november"],
        help="Simulation scenario to explain."
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name."
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Generate the report without using Ollama."
    )

    args = parser.parse_args()

    print("\nTower Brain Results Explainer")
    print("-----------------------------")

    try:
        facts = build_facts(args.scenario)

        factual_report = create_factual_report(facts)

        if not args.no_llm:
            print(f"\nGenerating explanation using {args.model}...")

            try:
                ai_explanation = explain_with_ollama(
                    args.model,
                    factual_report
                )

                if ai_explanation:
                    factual_report += (
                        "\n\n## 5. AI-generated explanation\n\n"
                        + ai_explanation
                        + "\n"
                    )
                else:
                    print("Ollama returned an empty response.")

            except Exception as exc:
                print(f"\nOllama explanation unavailable: {exc}")
                print("Continuing with the factual report.")

        REPORT_FILE.write_text(
            factual_report,
            encoding="utf-8"
        )

        print("\nReport generated successfully.")
        print(f"Saved to: {REPORT_FILE}")
        print("\n" + factual_report)

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

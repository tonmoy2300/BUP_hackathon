from __future__ import annotations

import pulp


class OptimizationError(RuntimeError):
    """Raised when the LP solver can't find an optimal 24-hour schedule.

    Message is intentionally generic -- solver internals (CBC status codes,
    infeasibility diagnostics, constraint dumps, etc.) are never included.
    """


_HOURS_IN_DAY = 24


def _apply_directives(
    hours: list[dict], battery: dict, validated_directives: list[dict]
) -> tuple[list[float], list[float], set[int], set[int], list[float | None]]:
    """Derive per-hour parameters (0-23) from base data plus active directives.

    Returns (effective_solar, min_energy_floor, no_charge_hours,
    no_discharge_hours, grid_cap).
    """
    hours_by_h = {int(entry["hour"]): entry for entry in hours}

    effective_solar = [
        float(hours_by_h[h]["solar_kwh"]) for h in range(_HOURS_IN_DAY)
    ]
    min_energy_floor = [float(battery["minimum_energy_kwh"])] * _HOURS_IN_DAY
    no_charge_hours: set[int] = set()
    no_discharge_hours: set[int] = set()
    grid_cap: list[float | None] = [None] * _HOURS_IN_DAY

    for directive in validated_directives:
        if not directive.get("applies"):
            continue

        directive_type = directive["directive_type"]
        adjustment = directive.get("structured_adjustment") or {}

        if directive_type == "solar_reduction":
            factor = adjustment["factor"]
            for h in adjustment["hours"]:
                # Multiply, so overlapping solar_reduction directives compound.
                effective_solar[h] *= factor
        elif directive_type == "minimum_battery_reserve":
            floor = adjustment["minimum_energy_kwh"]
            for h in adjustment["hours"]:
                min_energy_floor[h] = max(min_energy_floor[h], floor)
        elif directive_type == "no_charge_window":
            no_charge_hours.update(adjustment["hours"])
        elif directive_type == "no_discharge_window":
            no_discharge_hours.update(adjustment["hours"])
        elif directive_type == "max_grid_window":
            cap = adjustment["max_grid_kwh"]
            for h in adjustment["hours"]:
                grid_cap[h] = cap if grid_cap[h] is None else min(grid_cap[h], cap)

    return effective_solar, min_energy_floor, no_charge_hours, no_discharge_hours, grid_cap


def solve_schedule(
    hours: list[dict], battery: dict, validated_directives: list[dict]
) -> dict:
    """Solve the 24-hour battery/grid dispatch LP with PuLP.

    `hours` is 24 dicts with hour/demand_kwh/solar_kwh/tariff_bdt_per_kwh.
    `battery` has capacity_kwh/initial_energy_kwh/minimum_energy_kwh/
    max_charge_kwh_per_hour/max_discharge_kwh_per_hour. `validated_directives`
    is the output of orchestrator.validate_directive_list -- already
    guardrails-checked, so directive shapes are trusted here.

    Returns {"hourly_plan": [...], "total_grid_kwh", "total_cost_bdt",
    "peak_grid_kwh"}, all floats rounded to 6 decimals.

    Raises OptimizationError, with a safe generic message, if the LP has no
    optimal solution (e.g. directives make it infeasible) or the solver
    itself fails to run.
    """
    n = _HOURS_IN_DAY
    hours_by_h = {int(entry["hour"]): entry for entry in hours}
    demand = [float(hours_by_h[h]["demand_kwh"]) for h in range(n)]
    tariff = [float(hours_by_h[h]["tariff_bdt_per_kwh"]) for h in range(n)]

    effective_solar, min_energy_floor, no_charge_hours, no_discharge_hours, grid_cap = (
        _apply_directives(hours, battery, validated_directives)
    )

    capacity_kwh = float(battery["capacity_kwh"])
    initial_energy_kwh = float(battery["initial_energy_kwh"])
    max_charge_per_hour = float(battery["max_charge_kwh_per_hour"])
    max_discharge_per_hour = float(battery["max_discharge_kwh_per_hour"])

    problem = pulp.LpProblem("gridwise_dispatch", pulp.LpMinimize)

    grid: dict[int, pulp.LpVariable] = {}
    solar_used: dict[int, pulp.LpVariable] = {}
    charge: dict[int, pulp.LpVariable] = {}
    discharge: dict[int, pulp.LpVariable] = {}
    energy: dict[int, pulp.LpVariable] = {}

    for h in range(n):
        grid[h] = pulp.LpVariable(f"grid_{h}", lowBound=0, upBound=grid_cap[h])
        solar_used[h] = pulp.LpVariable(
            f"solar_used_{h}", lowBound=0, upBound=effective_solar[h]
        )
        charge_ub = 0.0 if h in no_charge_hours else max_charge_per_hour
        charge[h] = pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=charge_ub)
        discharge_ub = 0.0 if h in no_discharge_hours else max_discharge_per_hour
        discharge[h] = pulp.LpVariable(
            f"discharge_{h}", lowBound=0, upBound=discharge_ub
        )
        energy[h] = pulp.LpVariable(
            f"energy_{h}", lowBound=min_energy_floor[h], upBound=capacity_kwh
        )

    for h in range(n):
        problem += (
            grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h],
            f"energy_balance_{h}",
        )
        prev_energy = energy[h - 1] if h > 0 else initial_energy_kwh
        problem += (
            energy[h] == prev_energy + charge[h] - discharge[h],
            f"battery_transition_{h}",
        )

    problem += energy[n - 1] == initial_energy_kwh, "end_of_day_neutrality"

    problem += pulp.lpSum(grid[h] * tariff[h] for h in range(n)), "total_grid_cost"

    try:
        status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
    except Exception as exc:
        raise OptimizationError(
            "Unable to compute an optimal dispatch schedule for the given inputs."
        ) from exc

    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError(
            "Unable to compute an optimal dispatch schedule for the given inputs."
        )

    grid_raw = [grid[h].value() for h in range(n)]
    solar_used_raw = [solar_used[h].value() for h in range(n)]
    charge_raw = [charge[h].value() for h in range(n)]
    discharge_raw = [discharge[h].value() for h in range(n)]
    energy_raw = [energy[h].value() for h in range(n)]

    hourly_plan = []
    for h in range(n):
        net = charge_raw[h] - discharge_raw[h]
        if net > 1e-6:
            battery_action = "charge"
            battery_kwh = net
        elif net < -1e-6:
            battery_action = "discharge"
            battery_kwh = -net
        else:
            battery_action = "idle"
            battery_kwh = 0.0

        hourly_plan.append(
            {
                "hour": h,
                "grid_kwh": round(grid_raw[h], 6),
                "solar_used_kwh": round(solar_used_raw[h], 6),
                "battery_action": battery_action,
                "battery_kwh": round(battery_kwh, 6),
                "battery_energy_after_kwh": round(energy_raw[h], 6),
            }
        )

    total_grid_kwh = round(sum(grid_raw), 6)
    total_cost_bdt = round(sum(g * t for g, t in zip(grid_raw, tariff)), 6)
    peak_grid_kwh = round(max(grid_raw), 6)

    return {
        "hourly_plan": hourly_plan,
        "total_grid_kwh": total_grid_kwh,
        "total_cost_bdt": total_cost_bdt,
        "peak_grid_kwh": peak_grid_kwh,
    }

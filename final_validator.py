"""final_validator.py — post-solver invariant checker.

Replays the LP output and raises FinalValidationError (with a safe generic
message) if any of the following checks fail (tolerance 0.01 kWh):

1. Per-hour energy balance:
       grid + solar_used + discharge == demand + charge
2. solar_used is within [0, effective_solar] (after solar_reduction directives).
3. battery_energy_after is within [effective min floor, capacity].
4. charge/discharge are within [0, max_per_hour].
5. No charge in no_charge hours; no discharge in no_discharge hours.
6. grid <= cap in max_grid_window hours.
7. Battery energy transitions chain correctly from initial_energy_kwh.
8. Hour 23 ends at initial_energy_kwh (end-of-day neutrality).
9. Totals match the hourly sums.
10. Total cost matches sum(grid * tariff).
"""

from __future__ import annotations

from optimizer import _apply_directives

_TOL = 0.01  # kWh tolerance for all checks


class FinalValidationError(RuntimeError):
    """Raised when the solver output fails a post-hoc invariant check.

    The message is intentionally generic so it's safe to propagate via HTTP.
    The caller (main.py) logs the real exception before returning the safe 500.
    """


def _fail(msg: str) -> None:
    raise FinalValidationError(
        f"Schedule failed validation: {msg}"
    )


def validate_schedule(
    hours: list[dict],
    battery: dict,
    validated_directives: list[dict],
    schedule: dict,
) -> None:
    """Validate the solver output ``schedule`` against inputs and directives.

    Parameters
    ----------
    hours:
        The 24 hourly dicts passed to the solver (hour, demand_kwh,
        solar_kwh, tariff_bdt_per_kwh).
    battery:
        Battery parameters dict (capacity_kwh, initial_energy_kwh,
        minimum_energy_kwh, max_charge_kwh_per_hour,
        max_discharge_kwh_per_hour).
    validated_directives:
        The list returned by orchestrator.validate_directive_list.
    schedule:
        The dict returned by optimizer.solve_schedule (keys: hourly_plan,
        total_grid_kwh, total_cost_bdt, peak_grid_kwh).

    Raises
    ------
    FinalValidationError
        If any invariant check fails.
    """
    # Derive effective per-hour parameters (reuses optimizer's own logic).
    (
        effective_solar,
        min_energy_floor,
        no_charge_hours,
        no_discharge_hours,
        grid_cap,
    ) = _apply_directives(hours, battery, validated_directives)

    hours_by_h = {int(h["hour"]): h for h in hours}
    demand = [float(hours_by_h[h]["demand_kwh"]) for h in range(24)]
    tariff = [float(hours_by_h[h]["tariff_bdt_per_kwh"]) for h in range(24)]

    capacity = float(battery["capacity_kwh"])
    initial_energy = float(battery["initial_energy_kwh"])
    max_charge_per_hour = float(battery["max_charge_kwh_per_hour"])
    max_discharge_per_hour = float(battery["max_discharge_kwh_per_hour"])

    plan = {int(e["hour"]): e for e in schedule["hourly_plan"]}

    running_energy = initial_energy
    sum_grid = 0.0
    sum_cost = 0.0
    peak_grid = 0.0

    for h in range(24):
        e = plan[h]
        grid = float(e["grid_kwh"])
        solar_used = float(e["solar_used_kwh"])
        battery_action = e["battery_action"]
        battery_kwh = float(e["battery_kwh"])
        energy_after = float(e["battery_energy_after_kwh"])

        # Decode charge / discharge from action+kwh.
        if battery_action == "charge":
            charge = battery_kwh
            discharge = 0.0
        elif battery_action == "discharge":
            charge = 0.0
            discharge = battery_kwh
        else:  # idle
            charge = 0.0
            discharge = 0.0

        # 1. Energy balance per hour.
        balance_lhs = grid + solar_used + discharge
        balance_rhs = demand[h] + charge
        if abs(balance_lhs - balance_rhs) > _TOL:
            _fail(
                f"h{h}: energy balance violated "
                f"(grid={grid} + solar={solar_used} + dis={discharge} "
                f"= {balance_lhs:.4f} != demand={demand[h]} + chg={charge} "
                f"= {balance_rhs:.4f})"
            )

        # 2. solar_used bounds.
        if solar_used < -_TOL:
            _fail(f"h{h}: solar_used={solar_used} is negative")
        if solar_used > effective_solar[h] + _TOL:
            _fail(
                f"h{h}: solar_used={solar_used} > effective_solar={effective_solar[h]}"
            )

        # 3. Battery energy bounds.
        if energy_after < min_energy_floor[h] - _TOL:
            _fail(
                f"h{h}: battery_energy_after={energy_after} < "
                f"min_floor={min_energy_floor[h]}"
            )
        if energy_after > capacity + _TOL:
            _fail(
                f"h{h}: battery_energy_after={energy_after} > "
                f"capacity={capacity}"
            )

        # 4. Charge/discharge limits.
        if charge < -_TOL:
            _fail(f"h{h}: charge={charge} is negative")
        if discharge < -_TOL:
            _fail(f"h{h}: discharge={discharge} is negative")
        if charge > max_charge_per_hour + _TOL:
            _fail(
                f"h{h}: charge={charge} > max_charge_per_hour={max_charge_per_hour}"
            )
        if discharge > max_discharge_per_hour + _TOL:
            _fail(
                f"h{h}: discharge={discharge} > "
                f"max_discharge_per_hour={max_discharge_per_hour}"
            )

        # 5. no_charge / no_discharge windows.
        if h in no_charge_hours and charge > _TOL:
            _fail(f"h{h}: charge={charge} in no_charge_window")
        if h in no_discharge_hours and discharge > _TOL:
            _fail(f"h{h}: discharge={discharge} in no_discharge_window")

        # 6. Grid cap.
        if grid_cap[h] is not None and grid > grid_cap[h] + _TOL:
            _fail(
                f"h{h}: grid={grid} > cap={grid_cap[h]} (max_grid_window)"
            )

        # 7. Energy transition chain.
        expected_energy = running_energy + charge - discharge
        if abs(energy_after - expected_energy) > _TOL:
            _fail(
                f"h{h}: battery_energy_after={energy_after} != "
                f"prev_energy({running_energy}) + charge({charge}) "
                f"- discharge({discharge}) = {expected_energy:.4f}"
            )
        running_energy = energy_after

        sum_grid += grid
        sum_cost += grid * tariff[h]
        peak_grid = max(peak_grid, grid)

    # 8. End-of-day neutrality.
    if abs(running_energy - initial_energy) > _TOL:
        _fail(
            f"hour 23 energy_after={running_energy} != "
            f"initial_energy={initial_energy} (end-of-day neutrality)"
        )

    # 9. Totals match hourly sums.
    if abs(schedule["total_grid_kwh"] - sum_grid) > _TOL:
        _fail(
            f"total_grid_kwh={schedule['total_grid_kwh']} != "
            f"sum of hourly grid={sum_grid:.4f}"
        )
    if abs(schedule["peak_grid_kwh"] - peak_grid) > _TOL:
        _fail(
            f"peak_grid_kwh={schedule['peak_grid_kwh']} != "
            f"max of hourly grid={peak_grid:.4f}"
        )

    # 10. Cost matches sum(grid * tariff).
    if abs(schedule["total_cost_bdt"] - sum_cost) > _TOL:
        _fail(
            f"total_cost_bdt={schedule['total_cost_bdt']} != "
            f"sum(grid*tariff)={sum_cost:.4f}"
        )

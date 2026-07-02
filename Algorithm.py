import pulp
import numpy as np

# ============================================================
# CONFIG
# ============================================================
num_cars = 100
num_phases = 3
time_steps = range(12)

max_total_power = 120
max_phase_power = 40
max_car_power = 7.2
floor_power_per_hour = 0.3

lambda_1 = 1
lambda_2 = 1

NUM_NIGHTS = 5

# Fairness system
FAIRNESS_WEIGHT = 5
FAIRNESS_DECAY = 0.9
MAX_FAIRNESS_DEBT = 4.0
FAIRNESS_THRESHOLD = 3.8   # FIXED (reduced guarantee spam)

# Minimum system
MIN_VIABLE_RATIO = 0.35

# Guarantee system (budgeted + controlled)
TOTAL_GUARANTEE_BUDGET = 250

# Slack penalty
SLACK_PENALTY = 200

# Token system
MAX_TOKENS_SPENDABLE_PER_NIGHT = 3
TOKEN_TO_POINTS_RATE = 2

# ============================================================
# V2G (Vehicle-to-Grid) SYSTEM - voluntary, consent-based
# ============================================================
# Cars must explicitly opt in. A car is only eligible to donate power
# back to the pool if (a) it is enrolled, and (b) it was well-served
# (met its own MIN_VIABLE_RATIO) the PREVIOUS night - used as a proxy
# for "this car likely has charge to spare," avoiding the circular
# problem of needing tonight's own results before tonight's own
# constraints can be set. Donors are rewarded with tokens.
V2G_ENROLLMENT_RATE = 0.3           # 30% of residents opt into the program
V2G_MAX_DONATION_RATE = 2.0         # kW max draw from any single donating car per hour
TOKEN_REWARD_PER_KWH_DONATED = 0.5  # tokens earned per kWh donated back

np.random.seed(42)

# ============================================================
# INIT STATE
# ============================================================
car_tiers = {i: np.random.randint(1, 4) for i in range(num_cars)}
token_balance = {i: np.random.randint(0, 10) for i in range(num_cars)}
fairness_debt = {i: 0.0 for i in range(num_cars)}

# V2G enrollment is a one-time decision per car, persists across all nights
v2g_enrolled = {i: np.random.random() < V2G_ENROLLMENT_RATE for i in range(num_cars)}

# Night 1 has no history yet, so nobody is eligible to donate on night 1
was_well_served_last_night = {i: False for i in range(num_cars)}

grid_prices = {t: np.random.uniform(5, 15) for t in time_steps}
battery_schedule = {t: np.random.uniform(10, 40) for t in time_steps}
domestic_load = {t: np.random.uniform(20, 50) for t in time_steps}

history = []

# ============================================================
# HELPERS
# ============================================================
def simulate_token_spending(num_cars, token_balance, max_spend):
    spent = {}
    for i in range(num_cars):
        spend = min(np.random.randint(0, max_spend + 1), token_balance[i])
        spent[i] = spend
        token_balance[i] -= spend
    return spent, token_balance


def assign_to_least_loaded_phase(num_cars, car_e_req, num_phases):
    phase_load = {p: 0 for p in range(num_phases)}
    car_phase = {}

    for i in sorted(car_e_req, key=car_e_req.get, reverse=True):
        best = min(phase_load, key=phase_load.get)
        car_phase[i] = best
        phase_load[best] += car_e_req[i]

    return car_phase


def gini_coefficient(values):
    """
    Measures inequality in how satisfaction (delivered/demand ratio) is
    distributed across all cars tonight. Returns a number from 0 (perfectly
    equal - everyone got the same %) to 1 (maximally unequal - one car got
    everything, everyone else got nothing). This gives a single comparable
    number to track fairness across nights, rather than just eyeballing
    tables of who got what.
    """
    sorted_vals = np.sort(np.array(values, dtype=float))
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    if cumulative[-1] == 0:
        return 0.0  # everyone got exactly zero - defined as "equal" (degenerate case)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


# ============================================================
# MAIN LOOP
# ============================================================
for night in range(1, NUM_NIGHTS + 1):

    print("\n" + "=" * 70)
    print(f"NIGHT {night}")
    print("=" * 70)

    car_hours_left = {i: np.random.randint(3, 13) for i in range(num_cars)}

    car_e_req = {i: np.random.uniform(15, 65) for i in range(num_cars)}

    car_min_viable = {
        i: car_e_req[i] * MIN_VIABLE_RATIO
        for i in range(num_cars)
    }

    # tokens
    tokens_spent, token_balance = simulate_token_spending(
        num_cars, token_balance, MAX_TOKENS_SPENDABLE_PER_NIGHT
    )

    token_boost = {
        i: tokens_spent[i] * TOKEN_TO_POINTS_RATE
        for i in range(num_cars)
    }

    car_phase = assign_to_least_loaded_phase(
        num_cars, car_e_req, num_phases
    )

    # Determine tonight's V2G donor eligibility using LAST NIGHT's outcome
    eligible_donors = [
        i for i in range(num_cars)
        if v2g_enrolled[i] and was_well_served_last_night[i]
    ]

    # ============================================================
    # MODEL
    # ============================================================
    prob = pulp.LpProblem(f"EV_Night_{night}", pulp.LpMinimize)

    P = pulp.LpVariable.dicts(
        "P",
        ((i, t) for i in range(num_cars) for t in time_steps),
        lowBound=0,
        upBound=max_car_power
    )

    slack = pulp.LpVariable.dicts(
        "slack",
        (i for i in range(num_cars)),
        lowBound=0
    )

    # V2G donation variable - power flowing OUT of eligible donor cars,
    # back into the shared pool. Non-donors are locked to 0.
    D = pulp.LpVariable.dicts(
        "Donation",
        ((i, t) for i in range(num_cars) for t in time_steps),
        lowBound=0,
        upBound=V2G_MAX_DONATION_RATE
    )
    for i in range(num_cars):
        if i not in eligible_donors:
            for t in time_steps:
                D[i, t].upBound = 0

    # Slack specifically for the donor safety constraint below. Only
    # declared nonzero for eligible donors - see explanation at the
    # constraint itself for why this is needed to avoid infeasibility.
    donor_slack = pulp.LpVariable.dicts(
        "donor_slack",
        (i for i in range(num_cars)),
        lowBound=0
    )
    for i in range(num_cars):
        if i not in eligible_donors:
            donor_slack[i].upBound = 0

    # ============================================================
    # WEIGHTS
    # ============================================================
    W = {}

    for i in range(num_cars):
        W[i] = (
            car_tiers[i] * 20
            + car_e_req[i] / car_hours_left[i]
            + fairness_debt[i] * FAIRNESS_WEIGHT
            + token_boost[i]
        )

    # ============================================================
    # OBJECTIVE
    # ============================================================
    cost = pulp.lpSum(
        grid_prices[t] * P[i, t]
        for i in range(num_cars)
        for t in time_steps
    )

    reward = pulp.lpSum(
        W[i] * P[i, t]
        for i in range(num_cars)
        for t in time_steps
    )

    slack_penalty = pulp.lpSum(
        SLACK_PENALTY * slack[i]
        for i in range(num_cars)
    )

    # Same penalty severity as the regular slack - donor_slack should
    # only ever be used as an absolute last resort, never exploited freely.
    donor_slack_penalty = pulp.lpSum(
        SLACK_PENALTY * donor_slack[i]
        for i in range(num_cars)
    )

    prob += cost - reward + slack_penalty + donor_slack_penalty

    # ============================================================
    # GRID LIMITS (now includes V2G donations as extra available power)
    # ============================================================
    for t in time_steps:

        total_donations_this_hour = pulp.lpSum(D[i, t] for i in range(num_cars))

        total_avail = max(
            0,
            max_total_power - domestic_load[t] + battery_schedule[t]
        )

        prob += pulp.lpSum(P[i, t] for i in range(num_cars)) <= total_avail + total_donations_this_hour

        for p in range(num_phases):

            cars = [i for i in range(num_cars) if car_phase[i] == p]
            donors_on_phase = [i for i in eligible_donors if car_phase[i] == p]
            phase_donations_this_hour = pulp.lpSum(D[i, t] for i in donors_on_phase)

            phase_avail = max(
                0,
                max_phase_power
                - domestic_load[t] / num_phases
                + battery_schedule[t] / num_phases
            )

            prob += pulp.lpSum(P[i, t] for i in cars) <= phase_avail + phase_donations_this_hour

    # ============================================================
    # ENERGY CONSTRAINTS
    # ============================================================
    for i in range(num_cars):

        total = pulp.lpSum(
            P[i, t]
            for t in time_steps
            if t < car_hours_left[i]
        )

        prob += total + slack[i] >= car_min_viable[i]
        prob += total <= car_e_req[i]

        for t in time_steps:
            if t >= car_hours_left[i]:
                prob += P[i, t] == 0
                prob += D[i, t] == 0   # can't donate once departed either
            else:
                prob += P[i, t] >= floor_power_per_hour

        # Donor safety constraint: a donor cannot donate more energy total
        # than would bring them below their OWN minimum viable charge.
        # NOTE: eligibility is based on LAST night's outcome, but tonight's
        # `total` is itself a decision variable that could still end up
        # below car_min_viable[i] for unrelated reasons (different random
        # demand, different competition). A hard constraint here without
        # donor_slack would require "donate <= negative number" while also
        # requiring "donate >= 0" - infeasible. donor_slack absorbs that
        # gap, with a real penalty so it's only used as a last resort.
        if i in eligible_donors:
            total_donated = pulp.lpSum(D[i, t] for t in time_steps if t < car_hours_left[i])
            prob += total_donated <= total - car_min_viable[i] + donor_slack[i]

    # ============================================================
    # GUARANTEE (CONTROLLED + FIXED)
    # ============================================================
    candidates = [
        i for i in range(num_cars)
        if fairness_debt[i] >= FAIRNESS_THRESHOLD
    ]

    candidates.sort(key=lambda x: fairness_debt[x], reverse=True)

    candidates = candidates[:20]   # FIX: prevent explosion

    used = 0

    for i in candidates:

        if used >= TOTAL_GUARANTEE_BUDGET:
            break

        g = min(5, car_e_req[i])

        if used + g > TOTAL_GUARANTEE_BUDGET:
            g = TOTAL_GUARANTEE_BUDGET - used

        if g <= 0:
            break

        prob += pulp.lpSum(P[i, t] for t in time_steps if t < car_hours_left[i]) >= g

        used += g

    # ============================================================
    # SOLVE
    # ============================================================
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    print("Status:", pulp.LpStatus[status])

    if pulp.LpStatus[status] != "Optimal":
        print("WARNING: solver did not find an optimal solution tonight.")
        print("Reported values below may be unreliable - treat with caution.")

    # ============================================================
    # RESULTS
    # ============================================================
    delivered = {
        i: sum(P[i, t].varValue or 0 for t in time_steps)
        for i in range(num_cars)
    }

    donated = {
        i: sum(D[i, t].varValue or 0 for t in time_steps)
        for i in range(num_cars)
    }

    satisfaction = {
        i: delivered[i] / car_e_req[i]
        for i in range(num_cars)
    }

    # Award token rewards for any V2G donations made tonight
    for i in eligible_donors:
        if donated[i] > 0.001:
            reward_tokens = donated[i] * TOKEN_REWARD_PER_KWH_DONATED
            token_balance[i] += round(reward_tokens)

    # Update eligibility flag for TOMORROW based on TONIGHT's satisfaction
    was_well_served_last_night = {
        i: satisfaction[i] >= MIN_VIABLE_RATIO for i in range(num_cars)
    }

    # ============================================================
    # FAIRNESS UPDATE
    # ============================================================
    new_debt = {}

    for i in range(num_cars):

        r = satisfaction[i]

        if r >= 0.9:
            new_debt[i] = fairness_debt[i] * 0.5
        else:
            new_debt[i] = min(
                MAX_FAIRNESS_DEBT,
                fairness_debt[i] * FAIRNESS_DECAY + (1 - r)
            )

    fairness_debt = new_debt

    # ============================================================
    # CLEAN OUTPUT (FIXED FORMATTING)
    # ============================================================
    avg = np.mean(list(satisfaction.values())) * 100
    gini = gini_coefficient(list(satisfaction.values()))

    print(f"\nAverage Satisfaction: {avg:.2f}%")
    print(f"Gini Coefficient (fairness of distribution): {gini:.3f}  (0 = perfectly fair, 1 = maximally unfair)")

    worst = sorted(range(num_cars), key=lambda i: satisfaction[i])[:5]

    print("\nWorst Served Vehicles:")
    print("-" * 50)
    print(f"{'ID':>5} | {'Delivered':>10} | {'Demand':>10} | {'%':>6}")
    print("-" * 50)

    for i in worst:
        print(f"{i:5d} | {delivered[i]:10.2f} | {car_e_req[i]:10.2f} | {satisfaction[i]*100:6.2f}")

    print("-" * 50)

    # store history
    history.append({
        "night": night,
        "delivered": delivered,
        "demand": car_e_req,
        "donated": donated,
        "eligible_donors": len(eligible_donors),
        "total_donated": sum(donated.values()),
        "gini": gini,
    })

    print("\nPHASE-WISE POWER BREAKDOWN (with grid balance)")
    print("=" * 80)

    for t in time_steps:

        print(f"\nTIME SLOT {t}")
        print("-" * 80)

        # system-level split
        dom_total = domestic_load[t]
        battery_boost = battery_schedule[t]
        donations_total = sum(D[i, t].varValue or 0 for i in range(num_cars))

        total_available = max_total_power - dom_total + battery_boost + donations_total

        print(f"Domestic Load: {dom_total:.2f}")
        print(f"Battery Support: {battery_boost:.2f}")
        print(f"V2G Donations: {donations_total:.2f}")
        print(f"TOTAL EV AVAILABLE POWER: {total_available:.2f}")

        for p in range(num_phases):

            cars = [i for i in range(num_cars) if car_phase[i] == p]
            donors_on_phase = [i for i in eligible_donors if car_phase[i] == p]

            phase_domestic = dom_total / num_phases
            phase_battery = battery_boost / num_phases
            phase_donations = sum(D[i, t].varValue or 0 for i in donors_on_phase)

            phase_capacity = max(
                0,
                max_phase_power - phase_domestic + phase_battery
            ) + phase_donations

            ev_used = sum(P[i, t].varValue or 0 for i in cars)

            remaining = max(0, phase_capacity - ev_used)

            print(f"\n  Phase {p}:")
            print(f"    Domestic share: {phase_domestic:.2f}")
            print(f"    V2G donations this phase: {phase_donations:.2f}")
            print(f"    Phase capacity for EVs: {phase_capacity:.2f}")
            print(f"    EV used: {ev_used:.2f}")
            print(f"    Remaining headroom: {remaining:.2f}")
            print(f"    Active cars: {len(cars)}")

    print("\nTOP 15 VEHICLE BREAKDOWN")
    print("=" * 70)

    print(f"{'ID':>4} | {'Tier':>4} | {'Demand':>8} | {'Delivered':>9} | {'Donated':>8} | {'Token':>6} | {'%':>6}")

    for i in range(15):

        deliv = sum(P[i, t].varValue or 0 for t in time_steps)
        don = sum(D[i, t].varValue or 0 for t in time_steps)

        pct = (deliv / car_e_req[i]) * 100

        print(
        f"{i:4d} | "
        f"{car_tiers[i]:4d} | "
        f"{car_e_req[i]:8.2f} | "
        f"{deliv:9.2f} | "
        f"{don:8.2f} | "
        f"{token_balance[i]:6d} | "
        f"{pct:6.2f}"
        )

    print("\nTOKEN USAGE SNAPSHOT (sample)")
    print("=" * 70)

    sample = list(range(10))

    for i in sample:
        print(
            f"Vehicle {i}: Tokens left = {token_balance[i]} | V2G enrolled = {v2g_enrolled[i]} | Eligible tonight = {i in eligible_donors}"
        )

    print("\nSYSTEM LOAD PER TIME SLOT")
    print("=" * 70)

    for t in time_steps:

        total = sum(P[i, t].varValue or 0 for i in range(num_cars))
        don_t = sum(D[i, t].varValue or 0 for i in range(num_cars))

        print(f"Time {t}: Total EV load = {total:.2f} | V2G donated = {don_t:.2f}")

    print("\nV2G SUMMARY TONIGHT")
    print("=" * 70)
    print(f"Eligible donors: {len(eligible_donors)}")
    print(f"Total donated back to pool: {sum(donated.values()):.2f} kWh")

    if eligible_donors:
        top_donors = sorted(eligible_donors, key=lambda i: donated[i], reverse=True)[:5]
        print("\nTop V2G Donors Tonight:")
        print("-" * 60)
        print(f"{'ID':>5} | {'Donated kWh':>12} | {'Tokens Earned':>14} | {'New Balance':>12}")
        print("-" * 60)
        for i in top_donors:
            if donated[i] > 0.001:
                earned = round(donated[i] * TOKEN_REWARD_PER_KWH_DONATED)
                print(f"{i:5d} | {donated[i]:12.2f} | {earned:14d} | {token_balance[i]:12d}")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

for i in [0, 1, 2]:
    print(f"\nVehicle {i}:")
    for h in history:
        d = h["delivered"][i]
        r = h["demand"][i]
        print(f"Night {h['night']}: {d:.2f}/{r:.2f} ({(d/r)*100:.1f}%)")

print("\nV2G IMPACT OVER TIME")
print("=" * 70)
for h in history:
    print(f"Night {h['night']}: {h['eligible_donors']} eligible donors, "
          f"{h['total_donated']:.2f} kWh donated back to the pool")

print("\nFAIRNESS TREND (GINI COEFFICIENT) OVER TIME")
print("=" * 70)
for h in history:
    print(f"Night {h['night']}: Gini = {h['gini']:.3f}")

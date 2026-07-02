# AI-in-smart-charging
A constraint-based smart charging scheduler for residential EV infrastructure, built with PuLP (LP solver) and NumPy. Designed for apartment complexes running a shared 3-phase transformer with a hard power cap, where unmanaged concurrent charging would cause grid overloads.


Problem Context

A 100-vehicle apartment complex shares a transformer with a maximum EV allocation of 120 kW across 3 phases (40 kW per phase). Each vehicle can draw up to 7.2 kW. Naive concurrent charging would create demand of up to 720 kW — 6× the available supply — causing overloads, wire heating, and equipment degradation.

This system replaces ad-hoc charging with an LP-optimized nightly schedule that balances fairness, urgency, and grid safety simultaneously.


Core Architecture

The scheduler runs as a nightly loop (covering a 12-hour charging window, 6 PM – 6 AM). Each iteration builds and solves a Linear Programme that allocates power across 100 vehicles × 12 time slots, subject to grid, phase, and per-vehicle constraints.

Decision Variables

VariableDescriptionP[i, t]Power (kW) allocated to vehicle i at time slot tD[i, t]Power (kW) donated back to the pool by V2G-eligible vehicle i at slot tslack[i]Soft constraint absorber — allows minimum viable charge to be missed at a high penalty cost rather than making the problem infeasibledonor_slack[i]Safety absorber for V2G donors — prevents infeasibility when a donor's own received charge is uncertain


Implemented Features

1. LP-Based Power Allocation (PuLP / CBC)

The core is a minimisation problem combining three terms:

Minimise:  grid_cost − weighted_reward + slack_penalty + donor_slack_penalty


Grid cost discourages allocation during expensive grid price hours
Weighted reward pulls power toward high-priority vehicles (tier, urgency, fairness debt, token boost)
Slack penalties (200 per unit) ensure minimum viable charge is only missed as an absolute last resort


2. Three-Tier Vehicle Priority System

Each vehicle is assigned a tier (1–3) at enrolment. Tier level contributes directly to its weight in the objective function:

pythonW[i] = (tier * 20) + (demand / hours_left) + (fairness_debt * 5) + token_boost

Higher-tier vehicles receive more allocation weight. However, tier alone does not guarantee service — fairness debt and urgency can elevate lower-tier vehicles above higher-tier ones.

3. SoC-Proxy Demand Modelling

Each vehicle has a randomly sampled nightly energy requirement (car_e_req) representing the energy needed to reach target SoC. The algorithm respects both an upper bound (don't overcharge) and a minimum viable floor (35% of demand, configurable via MIN_VIABLE_RATIO).

4. Urgency Weighting via Demand Rate

Urgency is encoded implicitly through the ratio car_e_req[i] / car_hours_left[i] — vehicles that need a lot of energy in a short window automatically receive a higher weight in the objective, increasing their allocation priority without requiring a manual "urgent" flag.

5. 3-Phase Load Balancing

Vehicles are pre-assigned to phases using a greedy least-loaded heuristic — vehicles with the highest energy demand are assigned first, always to the phase currently carrying the least load. This distributes load across phases before the LP even runs, reducing per-phase imbalance.

Per-phase power caps are enforced as hard constraints inside the LP:

sum(P[i, t] for i on phase p) ≤ phase_capacity[t]

where phase capacity accounts for domestic load share and battery contribution.

6. Dynamic Available Power (Domestic Load Awareness)

The system does not assume a fixed EV budget. Available power per time slot is computed as:

EV_budget[t] = max_total_power − domestic_load[t] + battery_schedule[t]

Domestic load is sampled per time slot (20–50 kW range), meaning EV allocation automatically contracts during high-demand domestic periods and expands during quiet overnight hours.

7. Token Economy

Residents hold a token balance. Each night they can spend up to 3 tokens. Each token spent converts to 2 priority points added to their allocation weight:

pythontoken_boost[i] = tokens_spent[i] * TOKEN_TO_POINTS_RATE

Tokens provide a transparent, opt-in mechanism for residents to temporarily increase their priority — without the system being purely pay-to-win, since fairness debt and tier still co-determine outcomes.

8. Fairness Debt System

Every vehicle accumulates a fairness debt score that tracks how under-served it has been in recent nights:


If satisfaction ≥ 90% → debt decays aggressively (×0.5)
Otherwise → debt grows proportionally to the shortfall: debt = min(MAX_DEBT, debt × 0.9 + (1 − satisfaction))


Fairness debt feeds directly into the allocation weight (debt × 5), meaning chronically under-served vehicles progressively gain priority over well-served ones — without any manual intervention.

Debt is capped at MAX_FAIRNESS_DEBT = 4.0 to prevent runaway dominance.

9. Guarantee System

Vehicles whose fairness debt exceeds FAIRNESS_THRESHOLD = 3.8 are entered into a guarantee pool. The system commits a hard minimum energy grant (up to 5 kWh each) to the top 20 most-indebted vehicles, subject to a total nightly budget of 250 kWh across all guarantees.

This converts the soft fairness weight into a hard constraint for the most neglected vehicles, ensuring the LP cannot legally ignore them.

10. V2G (Vehicle-to-Grid) — Consent-Based

Vehicles can voluntarily enrol in the V2G programme (30% enrolment rate modelled). Enrolled vehicles may donate power back to the shared pool, increasing the available budget for other vehicles.

Key design decisions:


Eligibility is based on last night's outcome — a vehicle must have been well-served the previous night to donate tonight, used as a proxy for "likely has charge to spare," avoiding circular dependency on tonight's own unknowns
Donor safety constraint — a vehicle cannot donate more than the difference between what it received tonight and its own minimum viable charge
Token rewards — donors earn 0.5 tokens per kWh donated, creating an incentive loop without mandating participation
Phase-aware donations — V2G power is credited to the donor's own phase, not to the global pool, preserving phase balance


11. Battery Storage Integration

A stationary battery schedule (battery_schedule[t]) adds supplemental power each time slot. This models a community battery bank (e.g. second-life EV batteries) that charges during low-cost/solar hours and discharges into the EV budget at night — effectively time-shifting renewable generation into the charging window.

12. Grid Price Awareness

Hourly grid prices (grid_prices[t], 5–15 units) are included in the cost term of the objective. The LP naturally defers allocation to cheaper hours when possible, acting as implicit demand-response without requiring a separate scheduling layer.

13. Fairness Measurement (Gini Coefficient)

Each night, the Gini coefficient of satisfaction ratios across all 100 vehicles is computed and reported. This gives a single comparable metric for distributional fairness across nights — 0 means everyone got the same fraction of their demand, 1 means one vehicle got everything.


Configuration Reference

pythonnum_cars = 100                      # Fleet size
num_phases = 3                      # Electrical phases
time_steps = range(12)              # 12 × 1-hour slots (6 PM – 6 AM)
max_total_power = 120               # kW hard cap (transformer limit)
max_phase_power = 40                # kW per phase
max_car_power = 7.2                 # kW per vehicle (on-board charger limit)
floor_power_per_hour = 0.3          # kW minimum trickle per active slot
MIN_VIABLE_RATIO = 0.35             # Minimum acceptable satisfaction fraction
FAIRNESS_WEIGHT = 5                 # Multiplier for fairness debt in priority weight
FAIRNESS_DECAY = 0.9                # Debt decay rate per night (well-served vehicles)
MAX_FAIRNESS_DEBT = 4.0             # Debt ceiling
FAIRNESS_THRESHOLD = 3.8            # Debt level that triggers hard guarantee
TOTAL_GUARANTEE_BUDGET = 250        # Total kWh budget across all guarantees per night
SLACK_PENALTY = 200                 # Cost per unit of slack (infeasibility absorber)
MAX_TOKENS_SPENDABLE_PER_NIGHT = 3  # Token spend cap per vehicle per night
TOKEN_TO_POINTS_RATE = 2            # Priority points earned per token spent
V2G_ENROLLMENT_RATE = 0.3          # Fraction of fleet enrolled in V2G
V2G_MAX_DONATION_RATE = 2.0        # kW max draw from any single donor per hour
TOKEN_REWARD_PER_KWH_DONATED = 0.5 # Tokens earned per kWh donated via V2G


Output Per Night


Average satisfaction % across all vehicles
Gini coefficient (fairness of distribution)
Worst-served vehicles (bottom 5)
Phase-wise power breakdown per time slot (domestic load, battery support, V2G, EV draw, headroom)
Top 15 vehicle breakdown (tier, demand, delivered, donated, token balance, satisfaction %)
V2G summary (eligible donors, total kWh donated, top donors with token rewards)
System load per time slot



Requirements

pulp
numpy

Install with:

bashpip install pulp numpy

Run with:

bashpython charging_scheduler.py

CBC solver is bundled with PuLP — no separate solver installation needed.


Limitations & Known Gaps


Domestic load and battery schedule are currently randomly sampled — production use would replace these with real sensor feeds or forecasting models
SoC is modelled via car_e_req as a proxy; direct BMS integration is not implemented
Phase assignment is static per night (assigned once before the LP runs); dynamic reassignment mid-night is not supported
Solar generation is not modelled directly — battery schedule serves as a stand-in for time-shifted solar



Potential Extensions


Solar PV integration with day-ahead generation forecasting
Predictive departure time modelling from historical resident patterns
Real-time transformer temperature monitoring as a hard safety override
Harmonic distortion (THD) tracking for high-EV-density phases
Persistent resident profiles and multi-week fairness tracking
API layer for integration with OCPP-compliant charging hardware

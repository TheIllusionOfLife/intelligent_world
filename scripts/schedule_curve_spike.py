#!/usr/bin/env python3


def generate_schedule_data(
    steps: int = 500,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.995,
    w2: float = 0.1,
    decay_factor: float = 0.999,
    w2_floor: float = 0.02,
) -> dict[str, object]:
    if steps < 0:
        return {
            "temperature": [],
            "w2_effective": [],
            "w2_leads_temperature": False,
        }

    temperature = []
    w2_effective = []
    for step in range(steps + 1):
        temp = max(1e-9, initial_temperature * (cooling_rate**step))
        penalty = max(w2_floor, w2 * (decay_factor**step))
        temperature.append(temp)
        w2_effective.append(penalty)

    w2_decay_ratio = w2_effective[-1] / w2_effective[0] if w2_effective and w2_effective[0] else 1.0
    temp_decay_ratio = temperature[-1] / temperature[0] if temperature and temperature[0] else 1.0

    return {
        "temperature": temperature,
        "w2_effective": w2_effective,
        "w2_leads_temperature": w2_decay_ratio < temp_decay_ratio,
    }


if __name__ == "__main__":
    print(generate_schedule_data())

from typing import Dict, List

COL_INDEX = {
    "row_id": 0,
    "time": 1,
    "co": 2,
    "drum_level_1": 3,
    "drum_level_2": 4,
    "drum_level_3": 5,
    "main_motor_current": 6,
    "south_kiln_main_motor_freq": 7,
    "whrb_drum_pressure": 8,
    "whrb_outlet_pressure": 9,
    "esp_inlet_pressure": 10,
    "esp_outlet_pressure": 11,
    "fan_outlet_pressure": 12,
    "disc_feeder_freq": 13,
    "kiln_tail_temp": 14,
    "kiln_head_temp": 15,
    "whrb_inlet_temp": 16,
    "esp_inlet_temp": 17,
    "esp_outlet_temp": 18,
    "fan_outlet_temp": 19,
}

NODE_FEATURES: Dict[str, List[int]] = {
    "kiln_head": [
        COL_INDEX["kiln_head_temp"],
        COL_INDEX["main_motor_current"],
        COL_INDEX["south_kiln_main_motor_freq"],
    ],
    "kiln_tail": [COL_INDEX["kiln_tail_temp"]],
    "boiler": [
        COL_INDEX["whrb_drum_pressure"],
        COL_INDEX["whrb_outlet_pressure"],
        COL_INDEX["whrb_inlet_temp"],
        COL_INDEX["drum_level_1"],
        COL_INDEX["drum_level_2"],
        COL_INDEX["drum_level_3"],
    ],
    "esp": [
        COL_INDEX["esp_inlet_pressure"],
        COL_INDEX["esp_outlet_pressure"],
        COL_INDEX["esp_inlet_temp"],
        COL_INDEX["esp_outlet_temp"],
    ],
    "fan": [COL_INDEX["fan_outlet_pressure"], COL_INDEX["fan_outlet_temp"]],
    "feed": [COL_INDEX["disc_feeder_freq"]],
}

NODE_ORDER = ["kiln_head", "kiln_tail", "boiler", "esp", "fan", "feed"]

# Hyperedges requested by process topology.
HYPEREDGES = [
    ["feed", "kiln_head"],
    ["kiln_head", "kiln_tail"],
    ["kiln_tail", "boiler"],
    ["boiler", "esp"],
    ["esp", "fan"],
    ["kiln_head", "kiln_tail", "boiler"],
    ["boiler", "esp", "fan"],
]

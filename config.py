# -*- coding: utf-8 -*-

BIG_M = 100.0

CASE_DATA = {
    "case_name": "Refinery-7U-DirectBufferBlending-24H",
    "time": {
        "num_periods": 24,
        "period_length_hours": 1.0,
    },

    "crudes": ["SAM", "OMN", "IRL", "SHL", "SAX", "CBA"],

    "crude_supply": {
        "SAM": {"min": 0.0, "max": 15.0},
        "OMN": {"min": 0.0, "max": 15.0},
        "IRL": {"min": 0.0, "max": 15.0},
        "SHL": {"min": 0.0, "max": 15.0},
        "SAX": {"min": 0.0, "max": 15.0},
        "CBA": {"min": 0.0, "max": 15.0},
    },

    "crude_price_base": {
        "SAM": 2270.78,
        "OMN": 2342.94,
        "IRL": 2368.68,
        "SHL": 1911.68,
        "SAX": 2430.24,
        "CBA": 2354.09,
    },

    "crude_quality": {
        "SAM": {"SUL": 2.000, "TAN": 0.05, "MET": 284.5129, "VI1": 1.4695},
        "OMN": {"SUL": 1.400, "TAN": 0.50, "MET": 77.3966, "VI1": 2.0892},
        "IRL": {"SUL": 0.352, "TAN": 0.27, "MET": 332.8887, "VI1": 2.0524},
        "SHL": {"SUL": 1.100, "TAN": 2.25, "MET": 58.2460, "VI1": 2.1141},
        "SAX": {"SUL": 0.290, "TAN": 0.46, "MET": 78.2005, "VI1": 1.9479},
        "CBA": {"SUL": 0.390, "TAN": 0.80, "MET": 74.8573, "VI1": 2.0006},
    },

    "units": {
        "CDU1": {
            "type": "primary_distillation",
            "capacity_min": 0.0,
            "capacity_max": 45.0,
            "modes": ["normal"],
        },
        "CDU2": {
            "type": "primary_distillation",
            "capacity_min": 28.05,
            "capacity_max": 28.50,
            "modes": ["normal"],
        },
        "DFHC": {
            "type": "diesel_hydrotreating",
            "capacity_min": 0.0,
            "capacity_max": 11.55,
            "modes": ["normal"],
        },
        "FCC1": {
            "type": "fluid_catalytic_cracking",
            "capacity_min": 0.0,
            "capacity_max": 10.80,
            "modes": ["gasoline_mode", "diesel_mode"],
        },
        "FCC2": {
            "type": "fluid_catalytic_cracking",
            "capacity_min": 0.0,
            "capacity_max": 10.80,
            "modes": ["gasoline_mode", "diesel_mode"],
        },
        "ROHU": {
            "type": "residue_oil_hydrotreating",
            "capacity_min": 15.00,
            "capacity_max": 15.90,
            "modes": ["residue_hydrotreating", "hydrocracking_feed"],
        },
        "DHC": {
            "type": "diesel_hydrocracking",
            "capacity_min": 7.50,
            "capacity_max": 7.80,
            "modes": ["normal"],
        },
    },

    "inventory_nodes": {
        "crude_tanks": {
            "type": "feed_storage",
            "init": 0.0,
            "min": 0.0,
            "max": 90.0,
            "inventory_cost": 68.0,
        },
        "naphtha_buffer": {
            "type": "intermediate_buffer",
            "init": 0.0,
            "min": 0.0,
            "max": 20.0,
            "inventory_cost": 347.38,
        },
        "distillate_buffer": {
            "type": "intermediate_buffer",
            "init": 0.0,
            "min": 0.0,
            "max": 20.0,
            "inventory_cost": 274.76,
        },
        "fcc_feed_buffer": {
            "type": "intermediate_buffer",
            "init": 0.0,
            "min": 0.0,
            "max": 25.0,
            "inventory_cost": 272.99,
        },
        "residual_buffer": {
            "type": "intermediate_buffer",
            "init": 0.0,
            "min": 0.0,
            "max": 20.0,
            "inventory_cost": 205.15,
        },
        "hydro_feed_buffer": {
            "type": "intermediate_buffer",
            "init": 0.0,
            "min": 0.0,
            "max": 12.0,
            "inventory_cost": 205.15,
        },
        "gasoline_component_pool": {
            "type": "component_pool",
            "init": 0.0,
            "min": 0.0,
            "max": 30.0,
            "inventory_cost": 500.0,
        },
        "diesel_jet_component_pool": {
            "type": "component_pool",
            "init": 0.0,
            "min": 0.0,
            "max": 30.0,
            "inventory_cost": 450.0,
        },
        "lpg_component_pool": {
            "type": "component_pool",
            "init": 0.0,
            "min": 0.0,
            "max": 20.0,
            "inventory_cost": 350.0,
        },
    },

    "product_pools": {
        "gasoline_pool": {
            "products": ["W92", "W95"],
            "init": 2.0,
            "min": 2.0,
            "max": 30.0,
            "inventory_cost": 718.77,
        },
        "diesel_pool": {
            "products": ["JET", "EEN", "L10"],
            "init": 3.0,
            "min": 3.0,
            "max": 30.0,
            "inventory_cost": 647.00,
        },
        "lpg_pool": {
            "products": ["PLG", "PPP", "STR"],
            "init": 0.0,
            "min": 0.0,
            "max": 20.0,
            "inventory_cost": 420.0,
        },
    },

    "cdu_yields_aggregated": {
        "CDU1": {
            "naphtha_buffer": 0.151,
            "distillate_buffer": 0.193,
            "fcc_feed_buffer": 0.338,
            "residual_buffer": 0.233,
            "byproduct_or_untracked": 0.085,
        },
        "CDU2": {
            "naphtha_buffer": 0.152,
            "distillate_buffer": 0.189,
            "fcc_feed_buffer": 0.258,
            "residual_buffer": 0.318,
            "byproduct_or_untracked": 0.083,
        },
    },

    "secondary_unit_yields_aggregated": {
        "DFHC": {
            "gasoline_component_pool": 0.0542,
            "diesel_jet_component_pool": 0.9095,
            "lpg_component_pool": 0.0000,
            "byproduct_or_untracked": 0.0363,
        },
        "FCC1": {
            "gasoline_component_pool": 0.5830,
            "diesel_jet_component_pool": 0.1610,
            "lpg_component_pool": 0.1180,
            "byproduct_or_untracked": 0.1380,
        },
        "FCC2": {
            "gasoline_component_pool": 0.5570,
            "diesel_jet_component_pool": 0.1650,
            "lpg_component_pool": 0.1410,
            "byproduct_or_untracked": 0.1370,
        },
        "ROHU": {
            "residue_hydrotreating": {
                "gasoline_component_pool": 0.0060,
                "diesel_jet_component_pool": 0.0999,
                "hydro_feed_buffer": 0.8739,
                "lpg_component_pool": 0.0000,
                "byproduct_or_untracked": 0.0202,
            },
            "hydrocracking_feed": {
                "gasoline_component_pool": 0.0010,
                "diesel_jet_component_pool": 0.0333,
                "hydro_feed_buffer": 0.9632,
                "lpg_component_pool": 0.0000,
                "byproduct_or_untracked": 0.0025,
            },
        },
        "DHC": {
            "gasoline_component_pool": 0.1313,
            "diesel_jet_component_pool": 0.8372,
            "lpg_component_pool": 0.0214,
            "byproduct_or_untracked": 0.0101,
        },
    },

    "direct_routes": [
        {"from": "crude_tanks", "to": "CDU1", "stream": "crude"},
        {"from": "crude_tanks", "to": "CDU2", "stream": "crude"},
        {"from": "CDU1", "to": "DFHC", "stream": "distillate_direct"},
        {"from": "CDU2", "to": "DFHC", "stream": "distillate_direct"},
        {"from": "CDU1", "to": "FCC1", "stream": "vgo_direct"},
        {"from": "CDU1", "to": "FCC2", "stream": "vgo_direct"},
        {"from": "CDU2", "to": "FCC1", "stream": "vgo_direct"},
        {"from": "CDU2", "to": "FCC2", "stream": "vgo_direct"},
        {"from": "residual_buffer", "to": "ROHU", "stream": "residue"},
        {"from": "hydro_feed_buffer", "to": "DHC", "stream": "hydrotreated_intermediate"},
        {"from": "naphtha_buffer", "to": "gasoline_component_pool", "stream": "naphtha_component"},
        {"from": "DFHC", "to": "gasoline_component_pool", "stream": "light_component"},
        {"from": "DFHC", "to": "diesel_jet_component_pool", "stream": "diesel_component"},
        {"from": "FCC1", "to": "gasoline_component_pool", "stream": "fcc_gasoline_component"},
        {"from": "FCC1", "to": "diesel_jet_component_pool", "stream": "fcc_light_cycle_oil"},
        {"from": "FCC1", "to": "lpg_component_pool", "stream": "fcc_lpg"},
        {"from": "FCC2", "to": "gasoline_component_pool", "stream": "fcc_gasoline_component"},
        {"from": "FCC2", "to": "diesel_jet_component_pool", "stream": "fcc_light_cycle_oil"},
        {"from": "FCC2", "to": "lpg_component_pool", "stream": "fcc_lpg"},
        {"from": "ROHU", "to": "hydro_feed_buffer", "stream": "hydrotreated_intermediate"},
        {"from": "ROHU", "to": "diesel_jet_component_pool", "stream": "hydrotreated_distillate"},
        {"from": "DHC", "to": "gasoline_component_pool", "stream": "hydrocracked_naphtha"},
        {"from": "DHC", "to": "diesel_jet_component_pool", "stream": "hydrocracked_distillate"},
        {"from": "DHC", "to": "lpg_component_pool", "stream": "hydrocracked_lpg"},
    ],

    "buffer_routes": [
        {"from": "CDU1", "to": "naphtha_buffer", "stream": "naphtha_overflow"},
        {"from": "CDU2", "to": "naphtha_buffer", "stream": "naphtha_overflow"},
        {"from": "CDU1", "to": "distillate_buffer", "stream": "distillate_overflow"},
        {"from": "CDU2", "to": "distillate_buffer", "stream": "distillate_overflow"},
        {"from": "CDU1", "to": "fcc_feed_buffer", "stream": "vgo_overflow"},
        {"from": "CDU2", "to": "fcc_feed_buffer", "stream": "vgo_overflow"},
        {"from": "CDU1", "to": "residual_buffer", "stream": "residue_overflow"},
        {"from": "CDU2", "to": "residual_buffer", "stream": "residue_overflow"},
        {"from": "distillate_buffer", "to": "DFHC", "stream": "distillate_buffered"},
        {"from": "distillate_buffer", "to": "DHC", "stream": "distillate_buffered"},
        {"from": "fcc_feed_buffer", "to": "FCC1", "stream": "fcc_feed_buffered"},
        {"from": "fcc_feed_buffer", "to": "FCC2", "stream": "fcc_feed_buffered"},
    ],

    "blending_routes": [
        {"from": "gasoline_component_pool", "to": "gasoline_pool", "unit": "BlendingUnit"},
        {"from": "diesel_jet_component_pool", "to": "diesel_pool", "unit": "BlendingUnit"},
        {"from": "lpg_component_pool", "to": "lpg_pool", "unit": "BlendingUnit"},
    ],

    "product_grades": {
        "W92": {
            "pool": "gasoline_pool",
            "price": 4593.60,
            "demand_min": 14.31,
            "demand_max": BIG_M,
        },
        "W95": {
            "pool": "gasoline_pool",
            "price": 4990.06,
            "demand_min": 5.00,
            "demand_max": 5.00,
        },
        "JET": {
            "pool": "diesel_pool",
            "price": 4380.53,
            "demand_min": 6.20,
            "demand_max": 6.20,
        },
        "EEN": {
            "pool": "diesel_pool",
            "price": 3509.19,
            "demand_min": 3.50,
            "demand_max": BIG_M,
        },
        "L10": {
            "pool": "diesel_pool",
            "price": 4447.00,
            "demand_min": 0.00,
            "demand_max": BIG_M,
        },
        "PLG": {
            "pool": "lpg_pool",
            "price": 3577.98,
            "demand_min": 0.00,
            "demand_max": BIG_M,
        },
        "PPP": {
            "pool": "lpg_pool",
            "price": 7176.65,
            "demand_min": 0.93,
            "demand_max": BIG_M,
        },
        "STR": {
            "pool": "lpg_pool",
            "price": 6676.44,
            "demand_min": 0.69,
            "demand_max": 0.69,
        },
    },

    "demand_groups": {
        "gasoline_demand": {
            "pool": "gasoline_pool",
            "products": ["W92", "W95"],
            "base_min_total": 19.31,
        },
        "diesel_jet_demand": {
            "pool": "diesel_pool",
            "products": ["JET", "EEN", "L10"],
            "base_min_total": 9.70,
        },
        "lpg_demand": {
            "pool": "lpg_pool",
            "products": ["PLG", "PPP", "STR"],
            "base_min_total": 1.62,
        },
    },

    "product_quality_specs": {
        "W92": {
            "SUL": [0.0, 0.0009],
            "RON": [93.0, 150.0],
            "OLV": [0.0, 25.0],
            "BNZ": [0.0, 1.0],
            "OXY": [0.0, 14.0],
            "ARW": [0.0, 38.0],
        },
        "W95": {
            "SUL": [0.0, 0.0009],
            "RON": [96.0, 150.0],
            "OLV": [0.0, 25.0],
            "BNZ": [0.0, 1.0],
            "OXY": [0.0, 14.0],
            "ARW": [0.0, 38.0],
        },
        "JET": {
            "SUL": [0.0, 0.18],
            "ARW": [0.0, 19.0],
        },
        "EEN": {
            "SUL": [0.0, 0.08],
            "OLV": [0.0, 1.0],
            "ARW": [0.0, 9.0],
        },
        "L10": {
            "SUL": [0.0, 0.0045],
            "CTI": [46.0, 100.0],
        },
    },

    "blending_components": {
        "W92": [
            "MTB1", "BMT1", "RF1", "HAR1", "BHR1", "H1N_W92",
            "SZB1", "H8N", "TOL1", "XYL1", "ZBB", "HLN1",
        ],
        "W95": [
            "MTB2", "BMT2", "HAR2", "BHR2", "SZB2", "TOL2",
            "XYL2", "ALK", "HLN2",
        ],
        "JET": ["H5J", "HCJ"],
        "EEN": ["H2N_EEN", "H4N", "WN1"],
        "L10": ["H2D_L10", "H2E", "H2K", "H4D_L10", "H4E", "HCD"],
    },

    "hydrogen_feed_constraints": {
        "DFHC": {
            "hydrogen_stream": "RFH3",
            "process_streams": ["KE11", "LD1", "HD1_DFHC", "HD22", "H6D_DFHC"],
            "ratio": 0.0130,
        },
        "ROHU": {
            "hydrogen_stream": "H2T3",
            "process_streams": ["VR21", "IV22_ROHU", "K1O1", "C1D2", "D1O_ROHU", "V41_ROHU"],
            "ratio": 0.0115,
        },
        "DHC": {
            "hydrogen_stream": "RFH2",
            "process_streams": ["HD21", "K1D_DHC", "K1N", "C2D2", "LD2"],
            "ratio": 0.0251,
        },
    },

    "utility_price": {
        "CCC": 1.00,
        "WAT": 0.95,
        "CWT": 0.26,
        "DOW": 8.60,
        "DSW": 3.30,
        "KWH": 0.54,
        "LTM": 118.0,
        "MTM": 146.0,
        "HTM": 151.0,
        "UGS": 0.0,
        "UFL": 0.0,
    },

    "unit_utility_coefficients": {
        "CDU1": [1.4600, 0.0596, 1.8465, 0.0075, 0.0040, 5.6700, 0.0000, 0.0083, 0.0000, 0.0059, 0.0000],
        "CDU2": [1.4600, 0.0596, 1.8465, 0.0075, 0.0040, 5.6700, 0.0000, 0.0083, 0.0000, 0.0059, 0.0000],
        "DFHC": [3.5900, 0.0056, 3.5916, 0.0790, 0.0014, 17.5600, 0.0000, -0.1006, 0.0000, 0.0069, 0.0000],
        "FCC1": [21.2400, 0.1656, 34.2539, 0.4932, 0.0132, 33.3900, 0.0000, -0.0435, -0.0840, 0.0000, 0.0000],
        "FCC2": [22.9300, 0.0636, 27.2964, 0.4028, 0.0113, 24.6100, 0.0000, -0.0939, -0.1467, 0.0000, 0.0000],
        "ROHU": [13.5500, 0.0941, 2.6883, 0.1586, 0.1035, 19.7500, -0.0289, -0.1412, 0.1649, 0.0031, 0.0000],
        "DHC": [6.6000, 0.0037, 5.4405, 0.0089, 0.0335, 16.4000, 0.0000, 0.0569, 0.0000, 0.0017, 0.0000],
    },

    # --- Switching and load-stability cost parameters ---
    # c_sw: one-time cost incurred whenever a unit changes on/off or mode between
    #        consecutive periods.  Values calibrated so total switching cost is
    #        a meaningful (but not dominant) fraction of the ~1.58 M profit.
    "switching_cost": {
        "CDU1": 800.0,
        "CDU2": 800.0,
        "DFHC": 500.0,
        "FCC1": 1200.0,
        "FCC2": 1200.0,
        "ROHU": 1000.0,
        "DHC": 600.0,
    },

    # c_stab: per-unit-load penalty for load changes that exceed Delta_max.
    "load_stability_cost": {
        "CDU1": 5.0,
        "CDU2": 5.0,
        "DFHC": 3.0,
        "FCC1": 5.0,
        "FCC2": 5.0,
        "ROHU": 4.0,
        "DHC": 3.0,
    },

    # Delta_max: maximum load change (tons/period) allowed without penalty.
    "max_load_change": {
        "CDU1": 10.0,
        "CDU2": 5.0,
        "DFHC": 3.0,
        "FCC1": 3.0,
        "FCC2": 3.0,
        "ROHU": 2.0,
        "DHC": 2.0,
    },
}

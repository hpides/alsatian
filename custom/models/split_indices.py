from global_utils.model_names import *

SPLIT_INDEXES = {
    RESNET_152: [
        56,  # only linear layer
        53,  # last 1 bottleneck layer onwards
        52,  # last 2 bottleneck layer onwards
        51,  # last 3 bottleneck layer onwards
        50,  # last 4 bottleneck layer onwards
        49,  # last 5 bottleneck layer onwards
        48,  # last 6 bottleneck layer onwards
        47,  # last 7 bottleneck layer onwards
        46,  # last 8 bottleneck layer onwards
        45,  # last 9 bottleneck layer onwards
        44,  # last 10 bottleneck layer onwards
        43,  # last 11 bottleneck layer onwards
        42,  # last 12 bottleneck layer onwards
        41,  # last 13 bottleneck layer onwards
        40,  # last 14 bottleneck layer onwards
        39,  # last 15 bottleneck layer onwards
        38,  # last 16 bottleneck layer onwards
        37,  # last 17 bottleneck layer onwards
        36,  # last 18 bottleneck layer onwards
        35,  # last 19 bottleneck layer onwards
        34,  # last 20 bottleneck layer onwards
        33,  # last 21 bottleneck layer onwards
        32,  # last 22 bottleneck layer onwards
        31,  # last 23 bottleneck layer onwards
        30,  # last 24 bottleneck layer onwards
        29,  # last 25 bottleneck layer onwards
        28,  # last 26 bottleneck layer onwards
        27,  # last 27 bottleneck layer onwards
        26,  # last 28 bottleneck layer onwards
        25,  # last 29 bottleneck layer onwards
        24,  # last 30 bottleneck layer onwards
        23,  # last 31 bottleneck layer onwards
        22,  # last 32 bottleneck layer onwards
        21,  # last 33 bottleneck layer onwards
        20,  # last 34 bottleneck layer onwards
        19,  # last 35 bottleneck layer onwards
        18,  # last 36 bottleneck layer onwards
        17,  # last 37 bottleneck layer onwards
        16,  # last 38 bottleneck layer onwards
        15,  # last 39 bottleneck layer onwards
        14,  # last 40 bottleneck layer onwards
        13,  # last 41 bottleneck layer onwards
        12,  # last 42 bottleneck layer onwards
        11,  # last 43 bottleneck layer onwards
        10,  # last 44 bottleneck layer onwards
        9,  # last 45 bottleneck layer onwards
        8,  # last 46 bottleneck layer onwards
        7,  # last 47 bottleneck layer onwards
        6,  # last 48 bottleneck layer onwards
        5,  # last 49 bottleneck layer onwards
        4,  # last 50 bottleneck layer onwards
    ],
    RESNET_101: [
        39,  # only linear layer
        36,  # last 1 bottleneck layer onwards
        35,  # last 2 bottleneck layer onwards
        34,  # last 3 bottleneck layer onwards
        33,  # last 4 bottleneck layer onwards
        32,  # last 5 bottleneck layer onwards
        31,  # last 6 bottleneck layer onwards
        30,  # last 7 bottleneck layer onwards
        29,  # last 8 bottleneck layer onwards
        28,  # last 9 bottleneck layer onwards
        27,  # last 10 bottleneck layer onwards
        26,  # last 11 bottleneck layer onwards
        25,  # last 12 bottleneck layer onwards
        24,  # last 13 bottleneck layer onwards
        23,  # last 14 bottleneck layer onwards
        22,  # last 15 bottleneck layer onwards
        21,  # last 16 bottleneck layer onwards
        20,  # last 17 bottleneck layer onwards
        19,  # last 18 bottleneck layer onwards
        18,  # last 19 bottleneck layer onwards
        17,  # last 20 bottleneck layer onwards
        16,  # last 21 bottleneck layer onwards
        15,  # last 22 bottleneck layer onwards
        14,  # last 23 bottleneck layer onwards
        13,  # last 24 bottleneck layer onwards
        12,  # last 25 bottleneck layer onwards
        11,  # last 26 bottleneck layer onwards
        10,  # last 27 bottleneck layer onwards
        9,  # last 28 bottleneck layer onwards
        8,  # last 29 bottleneck layer onwards
        7,  # last 30 bottleneck layer onwards
        6,  # last 31 bottleneck layer onwards
        5,  # last 32 bottleneck layer onwards
        4,  # last 33 bottleneck layer onwards
    ],
    RESNET_50: [
        22,  # only linear layer
        19,  # last 1 bottleneck layer onwards
        17,  # last 2 bottleneck layer onwards
        16,  # last 3 bottleneck layer onwards
        15,  # last 4 bottleneck layer onwards
        14,  # last 5 bottleneck layer onwards
        13,  # last 6 bottleneck layer onwards
        12,  # last 7 bottleneck layer onwards
        11,  # last 8 bottleneck layer onwards
        10,  # last 9 bottleneck layer onwards
        9,  # last 10 bottleneck layer onwards
        8,  # last 11 bottleneck layer onwards
        7,  # last 12 bottleneck layer onwards
        6,  # last 13 bottleneck layer onwards
        5,  # last 14 bottleneck layer onwards
        4,  # last 15 bottleneck layer onwards
    ],
    RESNET_34: [  # interestingly these indexes seem to be the same as resnet50 but this one has approx 4M less params
        22,  # only linear layer
        19,  # last 1 bottleneck layer onwards
        17,  # last 2 bottleneck layer onwards
        16,  # last 3 bottleneck layer onwards
        15,  # last 4 bottleneck layer onwards
        14,  # last 5 bottleneck layer onwards
        13,  # last 6 bottleneck layer onwards
        12,  # last 7 bottleneck layer onwards
        11,  # last 8 bottleneck layer onwards
        10,  # last 9 bottleneck layer onwards
        9,  # last 10 bottleneck layer onwards
        8,  # last 11 bottleneck layer onwards
        7,  # last 12 bottleneck layer onwards
        6,  # last 13 bottleneck layer onwards
        5,  # last 14 bottleneck layer onwards
        4,  # last 15 bottleneck layer onwards
    ],
    RESNET_18: [
        14,  # only linear layer
        11,  # last 1 bottleneck layer onwards
        10,  # last 2 bottleneck layer onwards
        9,  # last 3 bottleneck layer onwards
        8,  # last 4 bottleneck layer onwards
        7,  # last 5 bottleneck layer onwards
        6,  # last 6 bottleneck layer onwards
        5,  # last 7 bottleneck layer onwards
        4,  # last 8 bottleneck layer onwards
    ],
    MOBILE_V2: [
        25,  # dropout + linear
        20,  # last part before features, but no block
        19,  # last 1 inverted residual block
        18,  # last 2 inverted residual block
        17,  # last 3 inverted residual block
        16,  # last 4 inverted residual block
        15,  # last 5 inverted residual block
        14,  # last 6 inverted residual block
        13,  # last 7 inverted residual block
        12,  # last 8 inverted residual block
        11,  # last 9 inverted residual block
        10,  # last 10 inverted residual block
        9,  # last 11 inverted residual block
        8,  # last 12 inverted residual block
        7,  # last 13 inverted residual block
        6,  # last 14 inverted residual block
        5,  # last 15 inverted residual block
        4,  # last 16 inverted residual block
        3  # last 17 inverted residual block
    ],
    EFF_NET_V2_S: [
        48,  # linear layer + dropout
        43,  # last part before features, but no block
        42,  # last 1 MBConv/FusedMBConv block
        41,  # last 2 MBConv/FusedMBConv block
        40,  # last 3 MBConv/FusedMBConv block
        39,  # last 4 MBConv/FusedMBConv block
        38,  # last 5 MBConv/FusedMBConv block
        37,  # last 6 MBConv/FusedMBConv block
        36,  # last 7 MBConv/FusedMBConv block
        35,  # last 8 MBConv/FusedMBConv block
        34,  # last 9 MBConv/FusedMBConv block
        33,  # last 10 MBConv/FusedMBConv block
        32,  # last 11 MBConv/FusedMBConv block
        31,  # last 12 MBConv/FusedMBConv block
        30,  # last 13 MBConv/FusedMBConv block
        29,  # last 14 MBConv/FusedMBConv block
        28,  # last 15 MBConv/FusedMBConv block
        27,  # last 16 MBConv/FusedMBConv block
        26,  # last 17 MBConv/FusedMBConv block
        25,  # last 18 MBConv/FusedMBConv block
        24,  # last 19 MBConv/FusedMBConv block
        23,  # last 20 MBConv/FusedMBConv block
        22,  # last 21 MBConv/FusedMBConv block
        21,  # last 22 MBConv/FusedMBConv block
        20,  # last 23 MBConv/FusedMBConv block
        19,  # last 24 MBConv/FusedMBConv block
        18,  # last 25 MBConv/FusedMBConv block
        17,  # last 26 MBConv/FusedMBConv block
        16,  # last 27 MBConv/FusedMBConv block
        15,  # last 28 MBConv/FusedMBConv block
        14,  # last 29 MBConv/FusedMBConv block
        13,  # last 30 MBConv/FusedMBConv block
        12,  # last 31 MBConv/FusedMBConv block
        11,  # last 32 MBConv/FusedMBConv block
        10,  # last 33 MBConv/FusedMBConv block
        9,  # last 34 MBConv/FusedMBConv block
        8,  # last 35 MBConv/FusedMBConv block
        7,  # last 36 MBConv/FusedMBConv block
        6,  # last 37 MBConv/FusedMBConv block
        5,  # last 38 MBConv/FusedMBConv block
        4,  # last 39 MBConv/FusedMBConv block
        3  # last 40 MBConv/FusedMBConv block
    ],
    EFF_NET_V2_L: [
        87,  # linear layer + dropout
        82,  # last part before features, but no block
        81,  # last 1 MBConv/FusedMBConv block
        80,  # last 2 MBConv/FusedMBConv block
        79,  # last 3 MBConv/FusedMBConv block
        78,  # last 4 MBConv/FusedMBConv block
        77,  # last 5 MBConv/FusedMBConv block
        76,  # last 6 MBConv/FusedMBConv block
        75,  # last 7 MBConv/FusedMBConv block
        74,  # last 8 MBConv/FusedMBConv block
        73,  # last 9 MBConv/FusedMBConv block
        72,  # last 10 MBConv/FusedMBConv block
        71,  # last 11 MBConv/FusedMBConv block
        70,  # last 12 MBConv/FusedMBConv block
        69,  # last 13 MBConv/FusedMBConv block
        68,  # last 14 MBConv/FusedMBConv block
        67,  # last 15 MBConv/FusedMBConv block
        66,  # last 16 MBConv/FusedMBConv block
        65,  # last 17 MBConv/FusedMBConv block
        64,  # last 18 MBConv/FusedMBConv block
        63,  # last 19 MBConv/FusedMBConv block
        62,  # last 20 MBConv/FusedMBConv block
        61,  # last 21 MBConv/FusedMBConv block
        60,  # last 22 MBConv/FusedMBConv block
        59,  # last 23 MBConv/FusedMBConv block
        58,  # last 24 MBConv/FusedMBConv block
        57,  # last 25 MBConv/FusedMBConv block
        56,  # last 26 MBConv/FusedMBConv block
        55,  # last 27 MBConv/FusedMBConv block
        54,  # last 28 MBConv/FusedMBConv block
        53,  # last 29 MBConv/FusedMBConv block
        52,  # last 30 MBConv/FusedMBConv block
        51,  # last 31 MBConv/FusedMBConv block
        50,  # last 32 MBConv/FusedMBConv block
        49,  # last 33 MBConv/FusedMBConv block
        48,  # last 34 MBConv/FusedMBConv block
        47,  # last 35 MBConv/FusedMBConv block
        46,  # last 36 MBConv/FusedMBConv block
        45,  # last 37 MBConv/FusedMBConv block
        44,  # last 38 MBConv/FusedMBConv block
        43,  # last 39 MBConv/FusedMBConv block
        42,  # last 40 MBConv/FusedMBConv block
        41,  # last 41 MBConv/FusedMBConv block
        40,  # last 42 MBConv/FusedMBConv block
        39,  # last 43 MBConv/FusedMBConv block
        38,  # last 44 MBConv/FusedMBConv block
        37,  # last 45 MBConv/FusedMBConv block
        36,  # last 46 MBConv/FusedMBConv block
        35,  # last 47 MBConv/FusedMBConv block
        34,  # last 48 MBConv/FusedMBConv block
        33,  # last 49 MBConv/FusedMBConv block
        32,  # last 50 MBConv/FusedMBConv block
        31,  # last 51 MBConv/FusedMBConv block
        30,  # last 52 MBConv/FusedMBConv block
        29,  # last 53 MBConv/FusedMBConv block
        28,  # last 54 MBConv/FusedMBConv block
        27,  # last 55 MBConv/FusedMBConv block
        26,  # last 56 MBConv/FusedMBConv block
        25,  # last 57 MBConv/FusedMBConv block
        24,  # last 58 MBConv/FusedMBConv block
        23,  # last 59 MBConv/FusedMBConv block
        22,  # last 60 MBConv/FusedMBConv block
        21,  # last 61 MBConv/FusedMBConv block
        20,  # last 62 MBConv/FusedMBConv block
        19,  # last 63 MBConv/FusedMBConv block
        18,  # last 64 MBConv/FusedMBConv block
        17,  # last 65 MBConv/FusedMBConv block
        16,  # last 66 MBConv/FusedMBConv block
        15,  # last 67 MBConv/FusedMBConv block
        14,  # last 68 MBConv/FusedMBConv block
        13,  # last 69 MBConv/FusedMBConv block
        12,  # last 70 MBConv/FusedMBConv block
        11,  # last 71 MBConv/FusedMBConv block
        10,  # last 72 MBConv/FusedMBConv block
        9,  # last 73 MBConv/FusedMBConv block
        8,  # last 74 MBConv/FusedMBConv block
        7,  # last 75 MBConv/FusedMBConv block
        6,  # last 76 MBConv/FusedMBConv block
        5,  # last 77 MBConv/FusedMBConv block
        4,  # last 78 MBConv/FusedMBConv block
        3  # last 79 MBConv/FusedMBConv block
    ],
    VIT_B_16: [
        17,  # last linear layer
        14,  # last 1 encoder block + classifier token layer
        13,  # last 2 encoder block
        12,  # last 3 encoder block
        11,  # last 4 encoder block
        10,  # last 5 encoder block
        9,  # last 6 encoder block
        8,  # last 7 encoder block
        7,  # last 8 encoder block
        6,  # last 9 encoder block
        5,  # last 10 encoder block
        4,  # last 11 encoder block
        3  # last 12 encoder block
    ],
    VIT_B_32: [
        17,  # last linear layer
        14,  # last 1 encoder block + classifier token layer
        13,  # last 2 encoder block
        12,  # last 3 encoder block
        11,  # last 4 encoder block
        10,  # last 5 encoder block
        9,  # last 6 encoder block
        8,  # last 7 encoder block
        7,  # last 8 encoder block
        6,  # last 9 encoder block
        5,  # last 10 encoder block
        4,  # last 11 encoder block
        3  # last 12 encoder block
    ],
    VIT_L_16: [
        29,  # last linear layer
        26,  # last 1 encoder block + classifier token layer
        25,  # last 2 encoder block
        24,  # last 3 encoder block
        23,  # last 4 encoder block
        22,  # last 5 encoder block
        21,  # last 6 encoder block
        20,  # last 7 encoder block
        19,  # last 8 encoder block
        18,  # last 9 encoder block
        17,  # last 10 encoder block
        16,  # last 11 encoder block
        15,  # last 12 encoder block
        14,  # last 13 encoder block
        13,  # last 14 encoder block
        12,  # last 15 encoder block
        11,  # last 16 encoder block
        10,  # last 17 encoder block
        9,  # last 18 encoder block
        8,  # last 19 encoder block
        7,  # last 20 encoder block
        6,  # last 21 encoder block
        5,  # last 22 encoder block
        4,  # last 23 encoder block
        3  # last 24 encoder block
    ],
    VIT_L_32: [
        29,  # last linear layer
        26,  # last 1 encoder block + classifier token layer
        25,  # last 2 encoder block
        24,  # last 3 encoder block
        23,  # last 4 encoder block
        22,  # last 5 encoder block
        21,  # last 6 encoder block
        20,  # last 7 encoder block
        19,  # last 8 encoder block
        18,  # last 9 encoder block
        17,  # last 10 encoder block
        16,  # last 11 encoder block
        15,  # last 12 encoder block
        14,  # last 13 encoder block
        13,  # last 14 encoder block
        12,  # last 15 encoder block
        11,  # last 16 encoder block
        10,  # last 17 encoder block
        9,  # last 18 encoder block
        8,  # last 19 encoder block
        7,  # last 20 encoder block
        6,  # last 21 encoder block
        5,  # last 22 encoder block
        4,  # last 23 encoder block
        3  # last 24 encoder block
    ],
    BERT: [
        14,  # last linear layer if there is one
        13,  # poller
        12,  # last encoder block
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
    ]
}

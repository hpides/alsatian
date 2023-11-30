from global_utils.model_names import *

SPLIT_INDEXES = {
    RESNET_152: [
        56,  # only linear layer
        53,  # last 1 bottleneck layer onwards
        51,  # last 3 bottleneck layer onwards
    ],
    RESNET_101: [
        39,  # only linear layer
        36,  # last 1 bottleneck layer onwards
        34,  # last 3 bottleneck layer onwards
    ],
    RESNET_50: [
        22,  # only linear layer
        19,  # last 1 bottleneck layer onwards
        17,  # last 3 bottleneck layer onwards
    ],
    RESNET_34: [  # interestingly these indexes seem to be the same as resnet50 but this one has approx 4M less params
        22,  # only linear layer
        19,  # last 1 bottleneck layer onwards
        17,  # last 3 bottleneck layer onwards
    ],
    RESNET_18: [
        14,  # only linear layer
        11,  # last 1 bottleneck layer onwards
        9,  # last 3 bottleneck layer onwards
    ],
    MOBILE_V2: [
        25,  # dropout + linear
        21,
        19,
    ],
    VIT_B_16: [
        17,
        14,
        12,
    ],
    VIT_B_32: [
        17,
        14,
        12,
    ],
    VIT_L_16: [
        29,
        26,
        24,
    ],
    VIT_L_32: [
        29,
        26,
        24,
    ],
    EFF_NET_V2_S: [
        48,
        42,
        40
    ],
    EFF_NET_V2_L: [
        87,
        81,
        79
    ]

}

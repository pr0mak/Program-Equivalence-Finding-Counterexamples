# import matplotlib.pyplot as plt
# import numpy as np

# epochs = np.arange(1, 51)

# train_loss = 1.5 * np.exp(-0.12 * epochs) + 0.1

# val_loss = 1.4 * np.exp(-0.10 * epochs) + 0.15 + 0.0008 * (epochs - 12)**2

# plt.figure(figsize=(8, 5))
# plt.plot(epochs, train_loss, label='Training Loss', color='#1f77b4', linewidth=2.5)
# plt.plot(epochs, val_loss, label='Validation Loss', color="#ff0e0e", linewidth=2.5)


# plt.axvline(x=16, color='red', linestyle='--', linewidth=2, label='Early Stopping (Epoch 15)')

# plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
# plt.xlabel('Epochs', fontsize=12)
# plt.ylabel('Cross-Entropy Loss', fontsize=12)
# plt.legend(fontsize=11)
# plt.grid(True, linestyle='--', alpha=0.7)

# plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')

import re
import matplotlib.pyplot as plt

log_data = """
Epoch [1/100] - Train Loss: 0.4871 | Val Loss: 0.4164
Epoch [2/100] - Train Loss: 0.4299 | Val Loss: 0.4064
Epoch [3/100] - Train Loss: 0.4151 | Val Loss: 0.3941
Epoch [4/100] - Train Loss: 0.3998 | Val Loss: 0.3805
Epoch [5/100] - Train Loss: 0.3865 | Val Loss: 0.3676
Epoch [6/100] - Train Loss: 0.3736 | Val Loss: 0.3576
Epoch [7/100] - Train Loss: 0.3614 | Val Loss: 0.3454
Epoch [8/100] - Train Loss: 0.3515 | Val Loss: 0.3381
Epoch [9/100] - Train Loss: 0.3424 | Val Loss: 0.3303
Epoch [10/100] - Train Loss: 0.3336 | Val Loss: 0.3248
Epoch [11/100] - Train Loss: 0.3260 | Val Loss: 0.3197
Epoch [12/100] - Train Loss: 0.3199 | Val Loss: 0.3157
Epoch [13/100] - Train Loss: 0.3142 | Val Loss: 0.3115
Epoch [14/100] - Train Loss: 0.3095 | Val Loss: 0.3075
Epoch [15/100] - Train Loss: 0.3043 | Val Loss: 0.3057
Epoch [16/100] - Train Loss: 0.2996 | Val Loss: 0.3025
Epoch [17/100] - Train Loss: 0.2969 | Val Loss: 0.3010
Epoch [18/100] - Train Loss: 0.2918 | Val Loss: 0.2985
Epoch [19/100] - Train Loss: 0.2876 | Val Loss: 0.2972
Epoch [20/100] - Train Loss: 0.2847 | Val Loss: 0.2941
Epoch [21/100] - Train Loss: 0.2817 | Val Loss: 0.2935
Epoch [22/100] - Train Loss: 0.2779 | Val Loss: 0.2913
Epoch [23/100] - Train Loss: 0.2764 | Val Loss: 0.2910
Epoch [24/100] - Train Loss: 0.2743 | Val Loss: 0.2887
Epoch [25/100] - Train Loss: 0.2709 | Val Loss: 0.2890
Epoch [26/100] - Train Loss: 0.2686 | Val Loss: 0.2895
Epoch [27/100] - Train Loss: 0.2654 | Val Loss: 0.2870
Epoch [28/100] - Train Loss: 0.2641 | Val Loss: 0.2863
Epoch [29/100] - Train Loss: 0.2627 | Val Loss: 0.2853
Epoch [30/100] - Train Loss: 0.2594 | Val Loss: 0.2842
Epoch [31/100] - Train Loss: 0.2557 | Val Loss: 0.2843
Epoch [32/100] - Train Loss: 0.2532 | Val Loss: 0.2846
Epoch [33/100] - Train Loss: 0.2518 | Val Loss: 0.2825
Epoch [34/100] - Train Loss: 0.2488 | Val Loss: 0.2831
Epoch [35/100] - Train Loss: 0.2476 | Val Loss: 0.2824
Epoch [36/100] - Train Loss: 0.2451 | Val Loss: 0.2830
Epoch [37/100] - Train Loss: 0.2442 | Val Loss: 0.2815
Epoch [38/100] - Train Loss: 0.2427 | Val Loss: 0.2804
Epoch [39/100] - Train Loss: 0.2405 | Val Loss: 0.2815
Epoch [40/100] - Train Loss: 0.2382 | Val Loss: 0.2829
Epoch [41/100] - Train Loss: 0.2375 | Val Loss: 0.2799
Epoch [42/100] - Train Loss: 0.2351 | Val Loss: 0.2796
Epoch [43/100] - Train Loss: 0.2340 | Val Loss: 0.2816
Epoch [44/100] - Train Loss: 0.2310 | Val Loss: 0.2820
Epoch [45/100] - Train Loss: 0.2295 | Val Loss: 0.2823
Epoch [46/100] - Train Loss: 0.2285 | Val Loss: 0.2813
Epoch [47/100] - Train Loss: 0.2265 | Val Loss: 0.2806
Epoch [48/100] - Train Loss: 0.2246 | Val Loss: 0.2799
Epoch [49/100] - Train Loss: 0.2235 | Val Loss: 0.2801
Epoch [50/100] - Train Loss: 0.2238 | Val Loss: 0.2824
Epoch [51/100] - Train Loss: 0.2202 | Val Loss: 0.2809
Epoch [52/100] - Train Loss: 0.2189 | Val Loss: 0.2815
Epoch [53/100] - Train Loss: 0.2146 | Val Loss: 0.2825
Epoch [54/100] - Train Loss: 0.2134 | Val Loss: 0.2806
Epoch [55/100] - Train Loss: 0.2139 | Val Loss: 0.2817
Epoch [56/100] - Train Loss: 0.2123 | Val Loss: 0.2823
Epoch [57/100] - Train Loss: 0.2100 | Val Loss: 0.2815
Epoch [58/100] - Train Loss: 0.2098 | Val Loss: 0.2809
Epoch [59/100] - Train Loss: 0.2072 | Val Loss: 0.2816
Epoch [60/100] - Train Loss: 0.2062 | Val Loss: 0.2817
Epoch [61/100] - Train Loss: 0.2031 | Val Loss: 0.2823
Epoch [62/100] - Train Loss: 0.2034 | Val Loss: 0.2834
Epoch [63/100] - Train Loss: 0.2026 | Val Loss: 0.2845
Epoch [64/100] - Train Loss: 0.2011 | Val Loss: 0.2858
Epoch [65/100] - Train Loss: 0.1994 | Val Loss: 0.2850
Epoch [66/100] - Train Loss: 0.2000 | Val Loss: 0.2816
Epoch [67/100] - Train Loss: 0.1988 | Val Loss: 0.2843
Epoch [68/100] - Train Loss: 0.1957 | Val Loss: 0.2830
Epoch [69/100] - Train Loss: 0.1951 | Val Loss: 0.2838
Epoch [70/100] - Train Loss: 0.1929 | Val Loss: 0.2853
Epoch [71/100] - Train Loss: 0.1924 | Val Loss: 0.2846
Epoch [72/100] - Train Loss: 0.1925 | Val Loss: 0.2863
Epoch [73/100] - Train Loss: 0.1922 | Val Loss: 0.2866
Epoch [74/100] - Train Loss: 0.1891 | Val Loss: 0.2852
Epoch [75/100] - Train Loss: 0.1876 | Val Loss: 0.2895
Epoch [76/100] - Train Loss: 0.1862 | Val Loss: 0.2865
Epoch [77/100] - Train Loss: 0.1847 | Val Loss: 0.2894
Epoch [78/100] - Train Loss: 0.1845 | Val Loss: 0.2897
Epoch [79/100] - Train Loss: 0.1850 | Val Loss: 0.2873
Epoch [80/100] - Train Loss: 0.1822 | Val Loss: 0.2903
Epoch [81/100] - Train Loss: 0.1805 | Val Loss: 0.2895
Epoch [82/100] - Train Loss: 0.1814 | Val Loss: 0.2900
Epoch [83/100] - Train Loss: 0.1771 | Val Loss: 0.2912
Epoch [84/100] - Train Loss: 0.1796 | Val Loss: 0.2925
Epoch [85/100] - Train Loss: 0.1770 | Val Loss: 0.2924
Epoch [86/100] - Train Loss: 0.1760 | Val Loss: 0.2921
Epoch [87/100] - Train Loss: 0.1771 | Val Loss: 0.2939
Epoch [88/100] - Train Loss: 0.1757 | Val Loss: 0.2933
Epoch [89/100] - Train Loss: 0.1715 | Val Loss: 0.2973
Epoch [90/100] - Train Loss: 0.1723 | Val Loss: 0.2960
Epoch [91/100] - Train Loss: 0.1710 | Val Loss: 0.2963
Epoch [92/100] - Train Loss: 0.1700 | Val Loss: 0.3001
Epoch [93/100] - Train Loss: 0.1712 | Val Loss: 0.2990
Epoch [94/100] - Train Loss: 0.1702 | Val Loss: 0.3007
Epoch [95/100] - Train Loss: 0.1677 | Val Loss: 0.2984
Epoch [96/100] - Train Loss: 0.1684 | Val Loss: 0.2972
Epoch [97/100] - Train Loss: 0.1673 | Val Loss: 0.3001
Epoch [98/100] - Train Loss: 0.1670 | Val Loss: 0.3000
Epoch [99/100] - Train Loss: 0.1649 | Val Loss: 0.3057
Epoch [100/100] - Train Loss: 0.1645 | Val Loss: 0.3036
"""

epochs = [int(x) for x in re.findall(r"Epoch \[(\d+)/100\]", log_data)]
train_loss = [float(x) for x in re.findall(r"Train Loss: ([\d.]+)", log_data)]
val_loss = [float(x) for x in re.findall(r"Val Loss: ([\d.]+)", log_data)]


plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', color='#C72125', linewidth=2) # Στα χρώματα του TU/e

plt.axvline(x=42, color='gray', linestyle='--', linewidth=1.5, label='Early Stopping (Epoch 42)')
plt.scatter(42, 0.2796, color='#C72125', s=50, zorder=5)

plt.title('Model Loss Over 100 Epochs (Divergence Analysis)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

plt.savefig('loss_curve_final.png', bbox_inches='tight')
plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

def create_architecture_diagram(output_path="fig_architecture.png"):
    fig, ax = plt.subplots(figsize=(19.5, 11), dpi=300)
    ax.set_xlim(0, 1950)
    ax.set_ylim(0, 1100)
    ax.axis('off')
    
    # Crisp white background
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Main Header Block
    ax.text(975, 1055, "Deep Transfer Learning CNN Architecture", 
            ha='center', va='center', fontsize=22, fontweight='bold', color='#0F172A',
            fontfamily='DejaVu Sans')
    ax.text(975, 1022, "MobileNetV2 Champion Backbone with Multi-Stage Regularized Dense Classification Head", 
            ha='center', va='center', fontsize=12.5, fontweight='normal', color='#64748B',
            fontfamily='DejaVu Sans')

    # ----------------------------------------------------
    # Stage Containers (Background sections)
    # ----------------------------------------------------
    # Stage 1 Container: Feature Extractor (Top)
    stg1_bg = FancyBboxPatch((40, 580), 1870, 410, boxstyle="round,pad=10,rounding_size=16",
                             facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(stg1_bg)
    
    # Stage 1 Header Badge
    stg1_tag = FancyBboxPatch((75, 955), 680, 26, boxstyle="round,pad=2,rounding_size=6",
                              facecolor='#E2E8F0', edgecolor='#CBD5E1', linewidth=1, zorder=2)
    ax.add_patch(stg1_tag)
    ax.text(90, 968, "STAGE 1: PRE-TRAINED FEATURE EXTRACTION & SPATIAL POOLING (FROZEN WEIGHTS)", 
            ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1E293B',
            fontfamily='DejaVu Sans', zorder=3)

    # Stage 2 Container: Classification Head (Bottom)
    stg2_bg = FancyBboxPatch((40, 55), 1870, 485, boxstyle="round,pad=10,rounding_size=16",
                             facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5, linestyle='--', zorder=1)
    ax.add_patch(stg2_bg)
    
    # Stage 2 Header Badge (Starts at x=100, leaving clear space on left for bus wire at x=58)
    stg2_tag = FancyBboxPatch((100, 498), 710, 26, boxstyle="round,pad=2,rounding_size=6",
                              facecolor='#E2E8F0', edgecolor='#CBD5E1', linewidth=1, zorder=2)
    ax.add_patch(stg2_tag)
    ax.text(115, 511, "STAGE 2: CUSTOM DENSE CLASSIFICATION & MULTI-TIER REGULARIZATION HEAD (TRAINABLE)", 
            ha='left', va='center', fontsize=9.5, fontweight='bold', color='#1E293B',
            fontfamily='DejaVu Sans', zorder=3)

    # Helper function for shadow + card
    def draw_card(x, y, w, h, title, subtitle, details, badge, bg_color, border_color, title_color='#0F172A', badge_color=None):
        if badge_color is None:
            badge_color = border_color
            
        # Subtle Drop Shadow
        shadow = FancyBboxPatch((x + 3, y - 3), w, h, boxstyle="round,pad=6,rounding_size=12",
                                facecolor='#000000', edgecolor='none', alpha=0.06, zorder=2)
        ax.add_patch(shadow)

        # Outer Card
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=6,rounding_size=12",
                              facecolor=bg_color, edgecolor=border_color, linewidth=2.0, zorder=3)
        ax.add_patch(card)
        
        # Badge Pill
        if badge:
            badge_len = len(badge)
            badge_w = badge_len * 7.2 + 22
            badge_box = FancyBboxPatch((x + w/2 - badge_w/2, y + h - 14), badge_w, 22,
                                       boxstyle="round,pad=2,rounding_size=8",
                                       facecolor=badge_color, edgecolor='none', zorder=4)
            ax.add_patch(badge_box)
            ax.text(x + w/2, y + h - 3, badge, ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color='#FFFFFF', fontfamily='DejaVu Sans', zorder=5)

        # Title
        ax.text(x + w/2, y + h*0.64, title, ha='center', va='center',
                fontsize=11.5, fontweight='bold', color=title_color, fontfamily='DejaVu Sans', zorder=4)
        
        # Subtitle
        if subtitle:
            ax.text(x + w/2, y + h*0.46, subtitle, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#334155', fontfamily='DejaVu Sans', zorder=4)
            
        # Details
        if details:
            ax.text(x + w/2, y + h*0.22, details, ha='center', va='center',
                    fontsize=8.5, fontweight='normal', color='#64748B', fontfamily='DejaVu Sans', zorder=4)

    # Helper function to draw crisp arrow with well-separated above-the-line label
    def draw_arrow(x1, y1, x2, y2, label=""):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>,head_length=8,head_width=5',
                                color='#475569', linewidth=2.0, zorder=6)
        ax.add_patch(arrow)
        if label:
            # Badge pill behind label for 100% legibility
            lw = len(label) * 7.5 + 16
            lbl_bg = FancyBboxPatch(((x1 + x2)/2 - lw/2, y1 + 18), lw, 20,
                                    boxstyle="round,pad=2,rounding_size=6",
                                    facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1, zorder=7)
            ax.add_patch(lbl_bg)
            ax.text((x1 + x2)/2, y1 + 28, label, ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color='#334155', fontfamily='DejaVu Sans', zorder=8)

    # ====================================================
    # ROW 1: PRE-TRAINED FEATURE EXTRACTION & POOLING
    # ====================================================
    row1_y = 630
    card_h1 = 280
    arrow_y1 = row1_y + card_h1 / 2

    # 1. Input Image Box (x=75, w=280)
    draw_card(75, row1_y, 280, card_h1, 
              "Input Image", 
              "224 × 224 × 3", 
              "RGB Fruit Image\nNormalized to [0, 1]\n(Apples, Bananas, Oranges)", 
              "INPUT TENSOR", 
              "#F0F9FF", "#0284C7")
    
    draw_arrow(355, arrow_y1, 420, arrow_y1, "224×224×3")

    # 2. Pretrained CNN Backbone (x=420, w=490)
    draw_card(420, row1_y, 490, card_h1, 
              "Pretrained CNN Backbone", 
              "MobileNetV2 / EfficientNetB0", 
              "53 Inverted Residual Blocks (MobileNetV2)\nDepthwise Separable Convolutions\nFrozen ImageNet Weights (2.26M Base Params)", 
              "CHAMPION EXTRACTOR", 
              "#FEF3C7", "#D97706", badge_color="#B45309")

    draw_arrow(910, arrow_y1, 975, arrow_y1, "7×7×1280")

    # 3. Spatial Feature Maps (x=975, w=355)
    draw_card(975, row1_y, 355, card_h1, 
              "Spatial Feature Maps", 
              "7 × 7 × 1280", 
              "Deep Semantic Descriptors\nCaptures Fungal Mold, Browning\nNecrosis, Bruises & Peel Decay", 
              "FEATURE MAPS", 
              "#FDF4FF", "#9333EA", badge_color="#7E22CE")

    draw_arrow(1330, arrow_y1, 1395, arrow_y1, "Pool 7×7")

    # 4. Global Average Pooling 2D (x=1395, w=480)
    draw_card(1395, row1_y, 480, card_h1, 
              "Global Avg Pooling 2D", 
              "1D Embedding: 1 × 1280", 
              "Spatial Compression: 7×7 → 1×1\nEliminates 62M Flatten Parameters\nEnforces Positional Invariance", 
              "SPATIAL POOLING", 
              "#EDE9FE", "#7C3AED", badge_color="#6D28D9")

    # ====================================================
    # BUS WIRE CONNECTOR: FROM GAP TO STAGE 2
    # ====================================================
    gap_cx = 1395 + 480 / 2  # 1635
    route_verts = [
        (gap_cx, row1_y),         # Exit bottom of GAP
        (gap_cx, 555),            # Drop down to bus channel
        (58, 555),                # Traverse left all the way across
        (58, 285),                # Drop down to Card 1 vertical center
        (75, 285)                 # Enter left edge of Batch Normalization 1
    ]
    route_codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO]
    route_path = Path(route_verts, route_codes)
    route_arrow = FancyArrowPatch(path=route_path,
                                  arrowstyle='-|>,head_length=8,head_width=5',
                                  color='#2563EB', linewidth=2.4, zorder=6)
    ax.add_patch(route_arrow)

    # Connector Pill Label placed on horizontal segment
    bus_badge = FancyBboxPatch((810, 542), 330, 26, boxstyle="round,pad=2,rounding_size=6",
                               facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.2, zorder=7)
    ax.add_patch(bus_badge)
    ax.text(975, 555, "Dense 1280-D Feature Representation", ha='center', va='center',
            fontsize=9, fontweight='bold', color='#1D4ED8', fontfamily='DejaVu Sans', zorder=8)

    # ====================================================
    # ROW 2: CUSTOM DENSE CLASSIFICATION HEAD (7 CARDS)
    # ====================================================
    row2_y = 155
    card_h2 = 260
    arrow_y2 = row2_y + card_h2 / 2  # 285

    # Card 1: Batch Normalization 1 (x=75, w=220)
    draw_card(75, row2_y, 220, card_h2,
              "Batch Normalization", 
              "Stabilizer 1", 
              "Normalizes 1280-D Input\nReduces Covariate Shift\nAccelerates Convergence", 
              "NORMALIZATION", 
              "#F1F5F9", "#475569")

    draw_arrow(295, arrow_y2, 335, arrow_y2)

    # Card 2: Dense Layer 1 (x=335, w=235)
    draw_card(335, row2_y, 235, card_h2,
              "Dense Layer 1", 
              "256 Neurons (ReLU)", 
              "Dense Linear Projection\nReLU Non-linearity\n327,936 Trainable Params", 
              "FULLY CONNECTED", 
              "#ECFDF5", "#059669")

    draw_arrow(570, arrow_y2, 610, arrow_y2)

    # Card 3: Dropout Layer 1 (x=610, w=225)
    draw_card(610, row2_y, 225, card_h2,
              "Dropout Layer 1", 
              "Rate = 0.40 (40%)", 
              "Stochastically Zeros Units\nPrevents Co-adaptation\nRobust Feature Subsets", 
              "REGULARIZATION", 
              "#FFF7ED", "#EA580C")

    draw_arrow(835, arrow_y2, 875, arrow_y2)

    # Card 4: Batch Normalization 2 (x=875, w=220)
    draw_card(875, row2_y, 220, card_h2,
              "Batch Normalization", 
              "Stabilizer 2", 
              "Conditions Dense 1 Outputs\nMaintains Unit Variance\nPrevents Vanishing Gradient", 
              "NORMALIZATION", 
              "#F1F5F9", "#475569")

    draw_arrow(1095, arrow_y2, 1135, arrow_y2)

    # Card 5: Dense Layer 2 (x=1135, w=235)
    draw_card(1135, row2_y, 235, card_h2,
              "Dense Layer 2", 
              "64 Neurons (ReLU)", 
              "Intermediate Compression\nNon-linear Decision Space\n16,448 Trainable Params", 
              "FULLY CONNECTED", 
              "#ECFDF5", "#059669")

    draw_arrow(1370, arrow_y2, 1410, arrow_y2)

    # Card 6: Dropout Layer 2 (x=1410, w=225)
    draw_card(1410, row2_y, 225, card_h2,
              "Dropout Layer 2", 
              "Rate = 0.20 (20%)", 
              "Secondary Regularizer\nFinal Overfitting Defense\nPreserves Generalization", 
              "REGULARIZATION", 
              "#FFF7ED", "#EA580C")

    draw_arrow(1635, arrow_y2, 1675, arrow_y2)

    # Card 7: Binary Output Layer (x=1675, w=200)
    draw_card(1675, row2_y, 200, card_h2,
              "Binary Output", 
              "1 Neuron (Sigmoid)", 
              "P(Rotten | Image) ∈ [0, 1]", 
              "PREDICTION HEAD", 
              "#FEF2F2", "#DC2626")

    # Mini Decision Badges inside Binary Output Card
    fresh_badge = FancyBboxPatch((1692, row2_y + 42), 166, 22, boxstyle="round,pad=2,rounding_size=6",
                                 facecolor='#16A34A', edgecolor='none', zorder=4)
    ax.add_patch(fresh_badge)
    ax.text(1775, row2_y + 53, "Fresh: P < 0.50", ha='center', va='center',
            fontsize=8, fontweight='bold', color='#FFFFFF', fontfamily='DejaVu Sans', zorder=5)

    rotten_badge = FancyBboxPatch((1692, row2_y + 15), 166, 22, boxstyle="round,pad=2,rounding_size=6",
                                  facecolor='#DC2626', edgecolor='none', zorder=4)
    ax.add_patch(rotten_badge)
    ax.text(1775, row2_y + 26, "Rotten: P ≥ 0.50", ha='center', va='center',
            fontsize=8, fontweight='bold', color='#FFFFFF', fontfamily='DejaVu Sans', zorder=5)

    # ====================================================
    # FOOTER: BENCHMARK & PERFORMANCE CALLOUT
    # ====================================================
    footer_box = FancyBboxPatch((75, 75), 1800, 48, boxstyle="round,pad=3,rounding_size=8",
                                facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.2, zorder=2)
    ax.add_patch(footer_box)
    
    footer_text = (
        "Champion Model Metrics (Held-Out Test Set, N = 2,698 images):   "
        "Accuracy: 97.96%   |   ROC-AUC: 0.9985   |   Precision: 97.89%   |   Recall: 97.96%   |   "
        "F1-Score: 0.9792   |   Inference Latency: ~18ms"
    )
    ax.text(975, 99, footer_text, ha='center', va='center', fontsize=9.5, fontweight='bold',
            color='#1E293B', fontfamily='DejaVu Sans', zorder=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close()
    print(f"Diagram successfully saved to {output_path}")

if __name__ == "__main__":
    create_architecture_diagram("fig_architecture.png")

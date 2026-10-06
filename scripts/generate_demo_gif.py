"""Generate reference-grade terminal GIF demo for EWM Engine."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = Path("assets")
ASSETS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = ASSETS_DIR / "demo.gif"

FONT_PATH = "C:/WINDOWS/Fonts/consola.ttf"
FONT_SIZE = 14
TITLE_FONT_SIZE = 12

try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    font_bold = ImageFont.truetype("C:/WINDOWS/Fonts/consolab.ttf", FONT_SIZE)
    title_font = ImageFont.truetype(FONT_PATH, TITLE_FONT_SIZE)
except Exception:
    font = ImageFont.load_default()
    font_bold = font
    title_font = font

WIDTH = 1000
HEIGHT = 580
PAD_X = 28
PAD_Y = 56
LINE_HEIGHT = 19

# Theme colors (Catppuccin Mocha style)
BG_COLOR = (24, 24, 37)  # #181825
HEADER_COLOR = (30, 30, 46)  # #1e1e2e
BORDER_COLOR = (49, 50, 68)  # #313244
TEXT_DEFAULT = (205, 214, 244)  # #cdd6f4
TEXT_MUTED = (147, 153, 178)  # #9399b2
TEXT_PROMPT = (137, 180, 250)  # #89b4fa
TEXT_SUCCESS = (166, 227, 161)  # #a6e3a1 (green)
TEXT_CYAN = (137, 220, 235)  # #89dceb
TEXT_YELLOW = (249, 226, 175)  # #f9e2af
TEXT_RED = (243, 139, 168)  # #f38ba8
TEXT_PURPLE = (203, 166, 247)  # #cba6f7


def draw_window_base():
    img = Image.new("RGB", (WIDTH, HEIGHT), (15, 17, 26))
    draw = ImageDraw.Draw(img)

    # Window background with border
    draw.rounded_rectangle(
        [(8, 8), (WIDTH - 8, HEIGHT - 8)],
        radius=10,
        fill=BG_COLOR,
        outline=BORDER_COLOR,
        width=1,
    )

    # Titlebar
    draw.rounded_rectangle(
        [(8, 8), (WIDTH - 8, 42)],
        radius=10,
        fill=HEADER_COLOR,
    )
    # Square off the bottom of the titlebar
    draw.rectangle([(8, 32), (WIDTH - 8, 42)], fill=HEADER_COLOR)
    draw.line([(8, 42), (WIDTH - 8, 42)], fill=BORDER_COLOR, width=1)

    # Traffic light buttons
    draw.ellipse([(24, 18), (36, 30)], fill=(243, 139, 168))  # Red
    draw.ellipse([(44, 18), (56, 30)], fill=(249, 226, 175))  # Yellow
    draw.ellipse([(64, 18), (76, 30)], fill=(166, 227, 161))  # Green

    # Window title
    title_text = "ewm-engine — bash — 1000x580"
    draw.text((WIDTH // 2 - 90, 16), title_text, fill=TEXT_MUTED, font=title_font)
    return img


def render_terminal_frame(lines_with_style, cursor=None):
    img = draw_window_base()
    draw = ImageDraw.Draw(img)

    curr_y = PAD_Y
    for line_parts in lines_with_style:
        curr_x = PAD_X
        for text, color, is_bold in line_parts:
            f = font_bold if is_bold else font
            draw.text((curr_x, curr_y), text, fill=color, font=f)
            # advance x
            bbox = draw.textbbox((curr_x, curr_y), text, font=f)
            curr_x = bbox[2]
        curr_y += LINE_HEIGHT

    if cursor:
        cx, cy = cursor
        draw.rectangle([(cx, cy), (cx + 8, cy + 15)], fill=TEXT_PURPLE)

    return img


# Build frames
frames = []
durations = []

prompt_prefix = [
    ("vinicius@ewm", TEXT_SUCCESS, True),
    (":", TEXT_MUTED, False),
    ("~/Enterprise-World-Model-Engine", TEXT_PROMPT, True),
    ("$ ", TEXT_DEFAULT, False),
]

cmd_text = "ewm example civicflow"

# Initial prompt
f1 = render_terminal_frame([[*prompt_prefix]], cursor=(PAD_X + 380, PAD_Y))
frames.append(f1)
durations.append(400)

# Typing animation
typed = ""
for ch in cmd_text:
    typed += ch
    if len(typed) % 3 == 0 or typed == cmd_text:
        line = [*prompt_prefix, (typed, TEXT_DEFAULT, False)]
        cx = PAD_X + 380 + (len(typed) * 8.4)
        frames.append(render_terminal_frame([line], cursor=(int(cx), PAD_Y)))
        durations.append(70)

# Enter pressed
line_entered = [*prompt_prefix, (cmd_text, TEXT_DEFAULT, False)]
frames.append(render_terminal_frame([line_entered]))
durations.append(300)

# Simulation output building
output_blocks = [
    # Block 1: Startup
    [
        [("Running CivicFlow Flood Response Example...", TEXT_CYAN, False)],
        [("=" * 78, TEXT_MUTED, False)],
        [("CIVICFLOW: DISASTER RELIEF RESOURCE ALLOCATION WORLD MODEL", TEXT_PURPLE, True)],
        [("=" * 78, TEXT_MUTED, False)],
        [
            (
                "RESEARCH DISCLAIMER: FOR SCIENTIFIC & EDUCATIONAL DEMONSTRATION ONLY.",
                TEXT_YELLOW,
                False,
            )
        ],
    ],
    # Block 2: Table header
    [
        [],
        [("=== Scenario Comparison (Baseline: Policy A (Myopic Nearest)) ===", TEXT_PROMPT, True)],
        [("-" * 105, TEXT_MUTED, False)],
        [
            (
                f"{'Metric':<30} | {'Scenario':<30} | {'Mean (Std)':<18} | {'Delta vs Base [CI] (*sig)':<25}",
                TEXT_DEFAULT,
                True,
            )
        ],
        [("-" * 105, TEXT_MUTED, False)],
    ],
    # Block 3: Metrics rows
    [
        [
            (
                f"{'unserved_water':<30} | {'Policy A (Myopic Nearest)':<30} | {'3830.00 (+/-0.0)':<18} | {'-':<25}",
                TEXT_MUTED,
                False,
            )
        ],
        [
            (
                f"{'':<30} | {'Policy B (Proactive Regional)':<30} | {'2860.00 (+/-0.0)':<18} | ",
                TEXT_DEFAULT,
                False,
            ),
            ("-970.00 (-25.3%) [-970, -970] *", TEXT_SUCCESS, True),
        ],
        [("-" * 105, BORDER_COLOR, False)],
        [
            (
                f"{'unserved_rations':<30} | {'Policy A (Myopic Nearest)':<30} | {'1790.00 (+/-0.0)':<18} | {'-':<25}",
                TEXT_MUTED,
                False,
            )
        ],
        [
            (
                f"{'':<30} | {'Policy B (Proactive Regional)':<30} | {'1310.00 (+/-0.0)':<18} | ",
                TEXT_DEFAULT,
                False,
            ),
            ("-480.00 (-26.8%) [-480, -480] *", TEXT_SUCCESS, True),
        ],
        [("-" * 105, BORDER_COLOR, False)],
        [
            (
                f"{'violations_count':<30} | {'Policy A (Myopic Nearest)':<30} | {'0.00 (+/-0.0)':<18} | {'-':<25}",
                TEXT_MUTED,
                False,
            )
        ],
        [
            (
                f"{'':<30} | {'Policy B (Proactive Regional)':<30} | {'0.00 (+/-0.0)':<18} | ",
                TEXT_DEFAULT,
                False,
            ),
            ("+0.00 (+0.0%) [0.00, 0.00]", TEXT_CYAN, False),
        ],
        [("-" * 105, TEXT_MUTED, False)],
    ],
    # Block 4: Systemic trace & prompt return
    [
        [],
        [("=== Systemic Dependency Trace (Policy B - Trajectory 0) ===", TEXT_PROMPT, True)],
        [
            (
                "Trace DAG: 24 state/action nodes · 27 causal edges · EvidenceLevel: verified",
                TEXT_CYAN,
                False,
            )
        ],
        [
            (
                "Result: Policy B reduces unserved relief demand by 25.7% with 0 violations.",
                TEXT_SUCCESS,
                True,
            )
        ],
        [],
        [*prompt_prefix],
    ],
]

accumulated = [line_entered]
for b_idx, block in enumerate(output_blocks):
    accumulated.extend(block)
    cursor_pos = None
    if b_idx == len(output_blocks) - 1:
        cursor_pos = (PAD_X + 380, PAD_Y + (len(accumulated) - 1) * LINE_HEIGHT)

    frame = render_terminal_frame(accumulated, cursor=cursor_pos)
    frames.append(frame)
    durations.append(700 if b_idx < 3 else 3800)

print(f"Total frames: {len(frames)}")

# Quantize and save as animated GIF
first = frames[0].convert("P", palette=Image.ADAPTIVE)
rest = [f.convert("P", palette=Image.ADAPTIVE) for f in frames[1:]]

first.save(
    OUTPUT_PATH,
    save_all=True,
    append_images=rest,
    duration=durations,
    loop=0,
    optimize=True,
)

print(f"Saved {OUTPUT_PATH} (size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")

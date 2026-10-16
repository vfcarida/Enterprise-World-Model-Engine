"""Generate social preview card (1280x640) for GitHub and documentation.

Generates both assets/social-preview.png and assets/social-preview.svg.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def generate_png_social_preview(output_path: Path) -> None:
    width = 1280
    height = 640

    img = Image.new("RGBA", (width, height), (11, 15, 25, 255))
    draw = ImageDraw.Draw(img)

    # Gradient background simulation with subtle glow
    for y in range(height):
        ratio = y / height
        r = int(11 + (18 - 11) * ratio)
        g = int(15 + (24 - 15) * ratio)
        b = int(25 + (38 - 25) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # Glow accents (indigo top-left, cyan bottom-right)
    for r_i in range(250, 0, -5):
        alpha = int(12 * (r_i / 250))
        draw.ellipse(
            [(-100 + 250 - r_i, -100 + 250 - r_i), (-100 + 250 + r_i, -100 + 250 + r_i)],
            fill=(99, 102, 241, alpha),
        )
        draw.ellipse(
            [(width - 150 - r_i, height - 150 - r_i), (width - 150 + r_i, height - 150 + r_i)],
            fill=(6, 182, 212, alpha),
        )

    # Outer border
    draw.rounded_rectangle(
        [(16, 16), (width - 16, height - 16)], radius=16, outline=(43, 53, 79, 255), width=2
    )

    # Fonts
    try:
        font_title = ImageFont.truetype("arialbd.ttf", 46)
        font_sub = ImageFont.truetype("arial.ttf", 22)
        font_card_head = ImageFont.truetype("arialbd.ttf", 18)
        font_card_body = ImageFont.truetype("arial.ttf", 15)
        font_badge = ImageFont.truetype("arialbd.ttf", 16)
        font_mono = ImageFont.truetype("consola.ttf", 16)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title
        font_card_head = font_title
        font_card_body = font_title
        font_badge = font_title
        font_mono = font_title

    # Header section
    draw.text((64, 56), "Enterprise World Model Engine", font=font_title, fill=(240, 246, 252, 255))
    draw.text(
        (64, 116),
        "Deterministic, auditable what-if simulation kernel for socio-technical systems — no GPU/LLM required.",
        font=font_sub,
        fill=(148, 163, 184, 255),
    )

    # Install pill / badge
    pill_box = [(64, 160), (380, 198)]
    draw.rounded_rectangle(
        pill_box, radius=8, fill=(30, 41, 59, 255), outline=(99, 102, 241, 255), width=1
    )
    draw.text((78, 168), "$ pip install ewm-engine", font=font_mono, fill=(165, 180, 252, 255))

    # Architecture Tag
    draw.rounded_rectangle(
        [(400, 160), (560, 198)], radius=8, fill=(16, 24, 39, 255), outline=(51, 65, 85, 255)
    )
    draw.text((416, 169), "Python >= 3.11", font=font_badge, fill=(203, 213, 225, 255))

    draw.rounded_rectangle(
        [(576, 160), (700, 198)], radius=8, fill=(16, 24, 39, 255), outline=(51, 65, 85, 255)
    )
    draw.text((592, 169), "Apache-2.0", font=font_badge, fill=(203, 213, 225, 255))

    draw.rounded_rectangle(
        [(716, 160), (840, 198)], radius=8, fill=(16, 24, 39, 255), outline=(51, 65, 85, 255)
    )
    draw.text((732, 169), "SLSA L2+", font=font_badge, fill=(56, 189, 248, 255))

    # Three Core Pillars (Cards)
    cards = [
        (
            "Structural World Model",
            "Known state + explicit rules",
            "Typed relational graph & bounded continuous resources. Physical and accounting conservation laws hold by construction with 0.000000 drift.",
            (99, 102, 241),
        ),
        (
            "Epistemic Honesty",
            "Scored on uncertainty",
            "Non-parametric bootstrap CIs, finite-sample conformal prediction bands, anytime-valid E-values, and automated refusal when unidentifiable.",
            (16, 185, 129),
        ),
        (
            "Cryptographic Audit",
            "Bit-for-bit replay",
            "Every state transition, parameter branch, and policy intervention produces an immutable SHA-256 fingerprint for forensic reproducibility.",
            (6, 182, 212),
        ),
    ]

    card_y = 230
    card_h = 320
    card_w = 360
    card_gap = 26

    for i, (title, tag, desc, color) in enumerate(cards):
        x0 = 64 + i * (card_w + card_gap)
        x1 = x0 + card_w
        y0 = card_y
        y1 = y0 + card_h

        # Card body
        draw.rounded_rectangle(
            [(x0, y0), (x1, y1)],
            radius=12,
            fill=(17, 24, 39, 230),
            outline=(31, 41, 55, 255),
            width=1,
        )
        # Top color accent line
        draw.line([(x0 + 16, y0 + 1), (x1 - 16, y0 + 1)], fill=(*color, 255), width=2)

        # Card Title
        draw.text((x0 + 20, y0 + 24), title, font=font_card_head, fill=(243, 244, 246, 255))
        draw.text((x0 + 20, y0 + 52), tag, font=font_card_body, fill=(*color, 255))

        # Divider
        draw.line([(x0 + 20, y0 + 80), (x1 - 20, y0 + 80)], fill=(31, 41, 55, 255), width=1)

        # Body text with manual wrap
        words = desc.split()
        lines = []
        cur = []
        for w in words:
            cur.append(w)
            if len(" ".join(cur)) > 34:
                lines.append(" ".join(cur[:-1]))
                cur = [w]
        if cur:
            lines.append(" ".join(cur))

        line_y = y0 + 100
        for line in lines:
            draw.text((x0 + 20, line_y), line, font=font_card_body, fill=(156, 163, 175, 255))
            line_y += 24

    # Footer note
    draw.text(
        (64, height - 52),
        "Principal Architect: Vinicius Carida (@vfcarida) · github.com/vfcarida/Enterprise-World-Model-Engine",
        font=font_card_body,
        fill=(100, 116, 139, 255),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    print(f"Social preview PNG saved to: {output_path}")


def generate_svg_social_preview(output_path: Path) -> None:
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640" style="font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0b0f19" />
      <stop offset="100%" stop-color="#121826" />
    </linearGradient>
    <linearGradient id="indigo-glow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#6366f1" stop-opacity="0.25" />
      <stop offset="100%" stop-color="#6366f1" stop-opacity="0.0" />
    </linearGradient>
    <linearGradient id="cyan-glow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#06b6d4" stop-opacity="0.25" />
      <stop offset="100%" stop-color="#06b6d4" stop-opacity="0.0" />
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="1280" height="640" fill="url(#bg)" rx="16" stroke="#2b354f" stroke-width="2" />
  <circle cx="100" cy="80" r="320" fill="url(#indigo-glow)" />
  <circle cx="1180" cy="560" r="280" fill="url(#cyan-glow)" />

  <!-- Header -->
  <text x="64" y="96" font-size="46" font-weight="800" fill="#f0f6fc" letter-spacing="-0.5">Enterprise World Model Engine</text>
  <text x="64" y="140" font-size="22" font-weight="400" fill="#94a3b8">Deterministic, auditable what-if simulation kernel for socio-technical systems — no GPU/LLM required.</text>

  <!-- Badges -->
  <g transform="translate(64, 168)">
    <rect width="310" height="38" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="1" />
    <text x="18" y="24" font-family="monospace" font-size="16" font-weight="600" fill="#a5b4fc">$ pip install ewm-engine</text>

    <rect x="326" width="160" height="38" rx="8" fill="#111827" stroke="#334155" stroke-width="1" />
    <text x="344" y="24" font-size="15" font-weight="600" fill="#cbd5e1">Python &gt;= 3.11</text>

    <rect x="502" width="120" height="38" rx="8" fill="#111827" stroke="#334155" stroke-width="1" />
    <text x="518" y="24" font-size="15" font-weight="600" fill="#cbd5e1">Apache-2.0</text>

    <rect x="638" width="120" height="38" rx="8" fill="#111827" stroke="#334155" stroke-width="1" />
    <text x="654" y="24" font-size="15" font-weight="600" fill="#38bdf8">SLSA L2+</text>
  </g>

  <!-- Pillar Cards -->
  <!-- Card 1 -->
  <g transform="translate(64, 236)">
    <rect width="360" height="310" rx="12" fill="#111827" fill-opacity="0.9" stroke="#1f2937" stroke-width="1" />
    <line x1="16" y1="0" x2="344" y2="0" stroke="#6366f1" stroke-width="3" />
    <text x="24" y="38" font-size="20" font-weight="700" fill="#f3f4f6">Structural World Model</text>
    <text x="24" y="66" font-size="14" font-weight="600" fill="#6366f1">Known state + explicit rules</text>
    <line x1="24" y1="84" x2="336" y2="84" stroke="#1f2937" stroke-width="1" />
    <text x="24" y="116" font-size="15" fill="#9ca3af">Typed relational graph and</text>
    <text x="24" y="142" font-size="15" fill="#9ca3af">bounded continuous resources.</text>
    <text x="24" y="168" font-size="15" fill="#9ca3af">Physical &amp; accounting conservation</text>
    <text x="24" y="194" font-size="15" fill="#9ca3af">laws hold by construction with</text>
    <text x="24" y="220" font-size="15" font-weight="700" fill="#10b981">0.000000 mass balance drift.</text>
  </g>

  <!-- Card 2 -->
  <g transform="translate(450, 236)">
    <rect width="360" height="310" rx="12" fill="#111827" fill-opacity="0.9" stroke="#1f2937" stroke-width="1" />
    <line x1="16" y1="0" x2="344" y2="0" stroke="#10b981" stroke-width="3" />
    <text x="24" y="38" font-size="20" font-weight="700" fill="#f3f4f6">Epistemic Honesty</text>
    <text x="24" y="66" font-size="14" font-weight="600" fill="#10b981">Scored on uncertainty</text>
    <line x1="24" y1="84" x2="336" y2="84" stroke="#1f2937" stroke-width="1" />
    <text x="24" y="116" font-size="15" fill="#9ca3af">Non-parametric bootstrap CIs,</text>
    <text x="24" y="142" font-size="15" fill="#9ca3af">finite-sample conformal coverage</text>
    <text x="24" y="168" font-size="15" fill="#9ca3af">bands, anytime-valid E-values,</text>
    <text x="24" y="194" font-size="15" fill="#9ca3af">and automated causal refusal</text>
    <text x="24" y="220" font-size="15" fill="#9ca3af">when non-identifiable.</text>
  </g>

  <!-- Card 3 -->
  <g transform="translate(836, 236)">
    <rect width="360" height="310" rx="12" fill="#111827" fill-opacity="0.9" stroke="#1f2937" stroke-width="1" />
    <line x1="16" y1="0" x2="344" y2="0" stroke="#06b6d4" stroke-width="3" />
    <text x="24" y="38" font-size="20" font-weight="700" fill="#f3f4f6">Cryptographic Audit</text>
    <text x="24" y="66" font-size="14" font-weight="600" fill="#06b6d4">Bit-for-bit replayability</text>
    <line x1="24" y1="84" x2="336" y2="84" stroke="#1f2937" stroke-width="1" />
    <text x="24" y="116" font-size="15" fill="#9ca3af">Deterministic SeedSequence PRNG.</text>
    <text x="24" y="142" font-size="15" fill="#9ca3af">Every state transition and policy</text>
    <text x="24" y="168" font-size="15" fill="#9ca3af">branch generates an immutable</text>
    <text x="24" y="194" font-size="15" fill="#9ca3af">canonical SHA-256 fingerprint</text>
    <text x="24" y="220" font-size="15" fill="#9ca3af">for regulatory compliance.</text>
  </g>

  <!-- Footer -->
  <text x="64" y="596" font-size="14" fill="#64748b">Principal Architect: Vinicius Caridá (@vfcarida) · github.com/vfcarida/Enterprise-World-Model-Engine</text>
</svg>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content.strip(), encoding="utf-8")
    print(f"Social preview SVG saved to: {output_path}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    assets_dir = root / "assets"
    docs_assets_dir = root / "docs" / "assets"

    generate_png_social_preview(assets_dir / "social-preview.png")
    generate_svg_social_preview(assets_dir / "social-preview.svg")

    # Mirror to docs/assets
    generate_png_social_preview(docs_assets_dir / "social-preview.png")
    generate_svg_social_preview(docs_assets_dir / "social-preview.svg")


if __name__ == "__main__":
    main()

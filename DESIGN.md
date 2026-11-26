---
name: Rabbit
description: A ledger of verdicts in white, brown and black slabs, carried by three squares and a pair of ears.
colors:
  ink: "#000000"
  paper: "#FFFFFF"
  surface-light: "#F7F2ED"
  surface-raised-light: "#EFE7E0"
  ink-secondary-light: "#4A3B32"
  muted-light: "#75655A"
  brown: "#6F4528"
  brown-deep: "#52321C"
  brown-surface: "#482B17"
  brown-surface-raised: "#5E3B22"
  muted-brown: "#D6C3B4"
  surface-dark: "#0F0B08"
  surface-raised-dark: "#1C1511"
  ink-secondary-dark: "#EDE4DC"
  muted-dark: "#A8968A"
  tan: "#C99A73"
  tan-light: "#DDB896"
typography:
  display:
    fontFamily: "Inter Variable, Inter, system-ui, sans-serif"
    fontSize: "clamp(2.5rem, 1rem + 5.5vw, 6rem)"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.04em"
  headline:
    fontFamily: "Inter Variable, Inter, system-ui, sans-serif"
    fontSize: "1.375rem"
    fontWeight: 500
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  title:
    fontFamily: "Inter Variable, Inter, system-ui, sans-serif"
    fontSize: "clamp(2rem, 1.1rem + 3.2vw, 4rem)"
    fontWeight: 800
    lineHeight: 1.05
    letterSpacing: "-0.04em"
  body:
    fontFamily: "Inter Variable, Inter, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "JetBrains Mono Variable, JetBrains Mono, SF Mono, Menlo, monospace"
    fontSize: "0.75rem"
    fontWeight: 500
    lineHeight: 1.55
    letterSpacing: "0.04em"
rounded:
  sm: "6px"
  md: "8px"
  card: "10px"
  lg: "14px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  2xl: "48px"
  3xl: "80px"
  4xl: "112px"
components:
  button-primary:
    backgroundColor: "{colors.brown}"
    textColor: "{colors.paper}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    height: "36px"
    padding: "0 14px"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    height: "36px"
    padding: "0 14px"
  card:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.ink}"
    rounded: "{rounded.card}"
    padding: "24px"
  ledger-row:
    textColor: "{colors.ink}"
    padding: "24px 8px"
---

# Design System: Rabbit

## Overview

**Creative North Star: "The Ledger of Verdicts"**

The site is a ledger, not a brochure. Every claim sits in a row with its evidence in the same glance, and the page is built from full-width slabs: white leads, brown marks chapters and black closes the page. There is no shadow and no illustration: state is carried by three squares, and the mark is a pair of ears.

Density is calm and typographic. Display type is large and tight, rows are separated by hairlines, and measurements are set in mono. The tone is that of a lab notebook that shows its losses next to its wins.

**Key Characteristics:**
- White dominates, brown is second (accent and slab), black is third (text and footer).
- Slabs (`.on-paper`, `.on-ink`, `.on-black`) that swap the whole semantic palette in scope, whatever the system theme.
- One marker system, three squares, always with a text label.
- Ears as the chapter mark: the hero pair, the seam pair, the EarsMark glyph.
- Hairlines instead of shadows; no gradients in the chrome.

## Colors

White, brown and black, like a rabbit, with warm neutrals derived from the brown. Colour never carries meaning; fill, outline and hatch do.

### Primary
- **Paper** (#FFFFFF): the dominant canvas, and the text on brown and black slabs.

### Secondary
- **Brown** (#6F4528): the accent on white: primary button, highlighted values.
- **Deep Brown** (#52321C): the canvas of the brown slab (`.on-ink`) and the button hover on white. On it, the accent is white.

### Tertiary
- **Ink** (#000000): text on white slabs, the 1px head rule of a ledger, the focus outline and selection on white, the canvas of the footer (`.on-black`) and of dark mode.

### Neutral
- **On white:** surface #F7F2ED, raised surface #EFE7E0, secondary text #4A3B32, muted #75655A.
- **On brown:** surface #482B17, raised surface #5E3B22, secondary text #EDE4DC, muted #D6C3B4.
- **On black** (footer and dark mode): surface #0F0B08, raised surface #1C1511, secondary text #EDE4DC, muted #A8968A, accent #C99A73, accent hover #DDB896, black text on the accent. Do not set #C99A73 or #A8968A as text on #6F4528 (below 4.5:1).
- Hairline is brown at 18% on white (faintest border 9%) and white at 16% on brown and black (faintest 7%).

### Named Rules
**The Slab Rule.** A section is `.on-paper`, `.on-ink` (brown) or `.on-black` (footer only), never a tint between. White is the default: the hero, the aim and the policy are white; brown marks the navigation, the layer names and the run section; the benchmarks are white, so the bars read as dark on light. Two white slabs in a row are separated by a hairline. A slab resets the whole palette in scope, accent and aliases included, so components read the variables and never a literal colour.

**The One Accent Rule.** Brown is the only hue. On white it marks actions and highlighted values; otherwise emphasis is weight, size, underline thickness or a marker.

**The Not By Colour Rule.** State is never carried by colour alone; every state has a shape and a word.

## Typography

**Display Font:** Inter Variable (with Inter, system-ui)
**Body Font:** Inter Variable
**Label/Mono Font:** JetBrains Mono Variable (with SF Mono, Menlo)

**Character:** Inter speaks; JetBrains Mono is reserved for data, measurements, code and technical labels. Whether Inter stays the voice of headings is undecided by the owner.

### Hierarchy
- **Display** (800, clamp(2.5rem, 1rem + 5.5vw, 6rem), 1.0, -0.04em): the aim sentence at 16ch. The hero word is larger still (up to 17rem, -0.06em).
- **Title** (800, clamp(2rem, 1.1rem + 3.2vw, 4rem), 1.05, -0.04em): slab titles such as the policy title.
- **Headline** (500, 1.375rem, 1.1, -0.02em): verdict case names and sub-heads.
- **Body** (400, 1rem, 1.55): prose on a 64ch measure; lead 1.0625rem at 1.65 in secondary text; small 0.875rem.
- **Data** (mono, max(0.9em, 13px), tabular numerals): numbers and measurements inline.
- **Label** (mono 500, 0.75rem, 0.04em, uppercase): table heads, terminal titles, copy buttons.

### Named Rules
**The Mono Is Data Rule.** JetBrains Mono appears only on figures, code, logs and table heads, never on prose or headings.

**The Tabular Rule.** Tables and measurements use tabular numerals; numeric columns align right.

## Layout

Slabs run edge to edge; content sits in a `.wrapper` of `min(100% - 48px, 1200px)` (`.wrapper-wide` 1280px). Spacing is a 4px grid, with slab padding of 112px at wide viewports, 80px below 960px and 48px below 640px. The ledger is the layout form: a grid of marker, claim and detail, evidence. At 960px the aim grid (5fr/7fr) collapses to one column; the verdict ledger drops its last column at 960px and becomes one line per row at 640px. Sections clear the 64px sticky bar with `scroll-margin-top`.

### Named Rules
**The Verdict Row Rule.** A claim never appears without its evidence in the same row, linked.

## Elevation & Depth

Flat. Depth is the alternation of slabs and the 1px hairline; there are no shadows on surfaces, and focus is a 2px ink outline offset 3px. Slab seams are hard edges, cropped by the ears where they occur. Motion is limited to short colour transitions (120ms and 180ms, cubic-bezier(0.25, 0, 0.5, 1)) and one authored moment: the markers of the verdict ledger fill by scroll (opacity and scale only), resolved to their final state under reduced motion.

### Named Rules
**The Hairline Rule.** Rows and sections are separated by a 1px hairline; the head of a ledger takes a 1px solid ink rule.

## Shapes

Small, quiet radii (6, 8, 10, 14px) on buttons, cards and code blocks; ledger rows and slabs are square. The three markers are squares, never circles. The ears are two tall capsules (rounded rects 5 wide on a 32 grid), filled as the nav mark and stroked 2 to 3px (non-scaling) as hero and seam ears, cropped by the edge of the slab.

## Components

### Markers
Three squares, one system. **Solid** is measured or refused. **Hollow** (1.5px outline) is open or let through. **Hatch** (outline plus diagonal strokes) is unavailable or not measured. Default size 0.875rem, 1.125 to 1.5rem in ledgers. Each has a text label, hidden for the eye or shown in the legend.

### Buttons
- **Shape:** 8px radius, 36px tall, 14px side padding, 14px medium text.
- **Primary:** accent fill (`--accent`), `--on-accent` text, hover `--accent-hover`: brown on white, white on brown, tan on black.
- **Secondary:** transparent with a hairline border; hover the surface of the slab.

### Cards / Containers
6 to 10px radius, hairline border, Soft Surface fill, 24px padding, no shadow. Used sparingly; the page is rows, not a card grid.

### Navigation
Sticky 64px bar on a brown slab with a hairline bottom edge, white text, underlined current link.

### Benchmark cards
One card per timing in an auto-fill grid (columns at least 20rem wide), grouped under an ink rule. A card has the case, the factor against Linux with mitigations off (the stricter baseline) set large, the factor against Linux 7.3.0 under it, three bars (Rabbit, Linux with mitigations off, Linux 7.3.0) and "QEMU TCG, an emulator · lower is better". Each card has its own linear scale from zero, with a 2px minimum bar. The Rabbit bar is solid accent with a semibold accent label; the Linux bars are raised surface with a muted outline. Every bar carries its name and exact value. A loss or a tie is worded as one; a recorded row has no factor and no bars, only its values. White background, hairline border, 10px radius, no shadow.

### Aim and verdict ledgers (signature)
A top rule in ink, rows separated by hairlines, marker + claim + detail + underlined evidence link. Row hover is a Soft Surface wash and a thicker evidence underline. The verdict ledger has four columns (case, Rabbit verdict, Linux verdict, evidence) and a legend of the three squares.

### Ears
The hero ears bleed from the hero; the seam ears mark chapters at a slab seam above 1100px, outline only, never over content.

## Do's and Don'ts

### Do:
- **Do** let white lead, use brown slabs to mark chapters and black for the footer, and let variables do the theming.
- **Do** pair every marker with a label and every row with a link to its file.
- **Do** keep "QEMU TCG, an emulator" beside every benchmark figure, and call ENOSYS unavailable (hatch), never protection.
- **Do** keep text out of the corner where the seam ears reach.

### Don't:
- **Don't** add a second hue, a shadow or a gradient to the chrome.
- **Don't** use a card grid under a slogan, or big-number metric blocks. The one exception is the benchmark chart, where the large factor always sits with its baseline in words, its three bars and its context.
- **Don't** use mono for prose or headings.
- **Don't** use circles or colour for state.

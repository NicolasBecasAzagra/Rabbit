---
name: Rabbit
description: A ledger of verdicts in pure black and pure white slabs, carried by three squares and a pair of ears.
colors:
  ink: "#000000"
  paper: "#FFFFFF"
  surface-light: "#F2F2F2"
  ink-secondary-light: "#404040"
  muted-light: "#6B6B6B"
  surface-dark: "#0A0A0A"
  surface-raised-dark: "#171717"
  ink-secondary-dark: "#E5E5E5"
  muted-dark: "#8F8F8F"
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
    backgroundColor: "{colors.ink}"
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

The site is a ledger, not a brochure. Every claim sits in a row with its evidence in the same glance, and the page is built from full-width slabs of pure black and pure white that alternate by section. There is no accent colour, no shadow and no illustration: state is carried by three squares, and the mark is a pair of ears.

Density is calm and typographic. Display type is large and tight, rows are separated by hairlines, and measurements are set in mono. The tone is that of a lab notebook that shows its losses next to its wins.

**Key Characteristics:**
- Strictly black and white; the old accent names all alias to ink.
- Alternating slabs (`.on-paper`, `.on-ink`) that swap the whole semantic palette in scope, whatever the system theme.
- One marker system, three squares, always with a text label.
- Ears as the chapter mark: the hero pair, the seam pair, the EarsMark glyph.
- Hairlines instead of shadows; no gradients in the chrome.

## Colors

Pure ink and paper with a short grey ladder. Colour never carries meaning; fill, outline and hatch do.

### Primary
- **Ink** (#000000): text on white slabs, the canvas of black slabs, the primary button, the 1px head rule of a ledger, the focus outline, selection.
- **Paper** (#FFFFFF): the canvas of white slabs, text on black slabs, the text of a primary button.

### Neutral
- **Soft Surface** (#F2F2F2): cards and hover wash on white slabs.
- **Graphite** (#404040): secondary text and button hover on white.
- **Quiet Grey** (#6B6B6B): muted text, table heads on white.
- **Near Black** (#0A0A0A) and **Raised Black** (#171717): surface and raised surface on black slabs.
- **Bone** (#E5E5E5): secondary text and button hover on black.
- **Ash** (#8F8F8F): muted text on black.
- Hairline is ink at 14% on white and white at 16% on black (`--hairline`); the faintest border is 7%.

### Named Rules
**The Slab Rule.** A section is `.on-paper` or `.on-ink`, never a tint between. A slab resets the semantic pair (`--ink`, `--paper`, `--surface`, `--muted`, `--hairline`), so components read the variables and never a literal colour.

**The No Accent Rule.** There is no hue. Emphasis is weight, size, underline thickness or a marker.

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
- **Body** (400, 1rem, 1.55): prose on a 64ch measure; lead 1.0625rem at 1.65 in Graphite; small 0.875rem.
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
- **Primary:** ink fill, inverse text; hover Graphite (Bone on black).
- **Secondary:** transparent with a hairline border; hover Soft Surface.

### Cards / Containers
6 to 10px radius, hairline border, Soft Surface fill, 24px padding, no shadow. Used sparingly; the page is rows, not a card grid.

### Navigation
Sticky 64px bar on a black slab with a hairline bottom edge, ink-on-black text, underlined links.

### Aim and verdict ledgers (signature)
A top rule in ink, rows separated by hairlines, marker + claim + detail + underlined evidence link. Row hover is a Soft Surface wash and a thicker evidence underline. The verdict ledger has four columns (case, Rabbit verdict, Linux verdict, evidence) and a legend of the three squares.

### Ears
The hero ears bleed from the hero; the seam ears mark chapters at a slab seam above 1100px, outline only, never over content.

## Do's and Don'ts

### Do:
- **Do** alternate `.on-paper` and `.on-ink` slabs and let variables do the theming.
- **Do** pair every marker with a label and every row with a link to its file.
- **Do** keep "QEMU TCG, an emulator" beside every benchmark figure, and call ENOSYS unavailable (hatch), never protection.
- **Do** keep text out of the corner where the seam ears reach.

### Don't:
- **Don't** add a hue, a shadow or a gradient to the chrome.
- **Don't** use a card grid under a slogan, or big-number metric blocks.
- **Don't** use mono for prose or headings.
- **Don't** use circles or colour for state.

---
name: Raymarcher
description: A field atlas of curved spacetime; every render is a numbered chart plate, measured from its margins.
colors:
  ink: "#0a0f1c"
  field-black: "#000000"
  plate: "#dbe3ec"
  plate-dim: "#93a1b6"
  rule: "#34465f"
  scale: "#5b7090"
  red: "#e4503f"
  red-ink: "#ff7a68"
typography:
  display:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "clamp(30px, 3.4vw, 46px)"
    fontWeight: 400
    lineHeight: 1
    letterSpacing: "-0.01em"
  headline:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "30px"
    fontWeight: 400
  title:
    fontFamily: "Spectral, Georgia, serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.1
  body:
    fontFamily: "Sofia Sans Condensed, Arial Narrow, sans-serif"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: 1.35
    fontFeature: "tnum"
  note:
    fontFamily: "Sofia Sans Condensed, Arial Narrow, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.3
  label:
    fontFamily: "Sofia Sans Condensed, Arial Narrow, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    letterSpacing: "0.2em"
  mark:
    fontFamily: "Sofia Sans Condensed, Arial Narrow, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    letterSpacing: "0.03em"
  scale:
    fontFamily: "Sofia Sans Condensed, Arial Narrow, sans-serif"
    fontSize: "10.5px"
    fontWeight: 500
    letterSpacing: "0.04em"
rounded:
  none: "0px"
spacing:
  margin-top: "34px"
  margin-left: "34px"
  margin-right: "14px"
  key-col-x: "16px"
  key-gap: "8px"
  cartouche-inset: "14px"
components:
  key-column:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.plate}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "10px 16px 14px"
  key-legend:
    textColor: "{colors.plate-dim}"
    typography: "{typography.label}"
  chart-entry:
    textColor: "{colors.plate-dim}"
    typography: "{typography.title}"
    padding: "3px 0"
  chart-entry-active:
    textColor: "{colors.plate}"
  choice:
    textColor: "{colors.plate-dim}"
    padding: "2px 0 4px"
  choice-active:
    textColor: "{colors.plate}"
  check-box:
    backgroundColor: "{colors.ink}"
    rounded: "{rounded.none}"
    size: "13px"
  slider-index:
    backgroundColor: "{colors.red}"
    rounded: "{rounded.none}"
    width: "3px"
    height: "18px"
  title-cartouche:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.plate}"
    typography: "{typography.display}"
    rounded: "{rounded.none}"
    padding: "10px 14px 12px"
---

# Design System: Raymarcher

## Overview

**Creative North Star: "Field Star Atlas"**

Every render is a numbered chart plate from a field star atlas. The lensed sky sits inside a double neatline on ink-blue paper and is measured from its margins: live azimuth and elevation scales on the top and left edges, a ruled title cartouche set into the chart's upper-left corner, red overprint marks with leader lines on the features the physics predicts, and one dense key band across the foot that holds every control. The render leads; the chrome is the apparatus of a printed chart, not an app panel.

Density lives in the margins and the key; the field itself stays bare except for measured marks. Everything is hairline, square and flat. Depth comes from the render alone, never from the interface. The earlier glassmorphic cyan dashboard is the confirmed anti-reference: no translucent floating panel, no glow, no glass.

The atlas also explains itself. Every mark printed on the chart is defined in a symbol key, the redshift colouring has a numeric scale beside it, and illustrative choices are labelled as such in the key's notes.

**Key Characteristics:**
- Double neatline around a black chart field; thin ink margins on every side.
- Live, measured tick scales in the top and left margins, tied to the camera.
- One red overprint, reserved for measured marks and the active state.
- Condensed sans with tabular numerals for catalogue labels; italic serif for object names.
- Square corners, hairline rules, no shadows, no glass.
- One key band at the foot, divided into ruled columns.

## Colors

A cold printed palette: ink-blue paper, plate-white type, a muted survey blue for rules and graduations, and one red overprint.

### Primary
- **Overprint Red** (red): the only accent. Measured marks on the chart (dashed shadow edge, leader lines, measured-point dots), the slider index notch, the checked checkbox fill and chart-entry dot, the active choice underline, and the render-refinement sweep on the top neatline.
- **Overprint Red, Type Weight** (red-ink): the lighter red used wherever red must be read as text or a 1px outline: mark labels on the chart, the active chart number, and every focus outline. Plain red is too dark for small text on ink; red-ink carries it.

### Neutral
- **Atlas Ink** (ink): the page ground, the margins, the key band, the cartouche fill (at 92% opacity) and the knockout gap around slider notches.
- **Field Black** (field-black): the chart field behind the canvas; the render owns this rectangle.
- **Plate White** (plate): primary text, values, active labels, the inner neatline (at 70% opacity), and secondary mark labels.
- **Plate Dim** (plate-dim): secondary text, notes, key legends, scale numerals, inactive catalogue names, the optical-axis cross.
- **Survey Rule** (rule): column dividers, key band rules, the outer neatline, the cartouche border.
- **Graticule Blue** (scale): tick lines on the margin scales, slider tracks and graduations, unchecked checkbox and chart-dot strokes.

### Named Rules
**The One Overprint Rule.** Red appears only on things that are measured or active: a predicted feature on the chart, the selected spacetime, the current slider value, the checked option, the refinement sweep, focus. Never on headings, decoration or structure.

**The Data Colour Exception.** The redshift scale bar (#ff4019 to #e6e6d9 to #3380ff) is data encoding matched to the shader's colouring, not palette. It always ships with numeric ticks and a plain-language note.

## Typography

**Display Font:** Spectral italic (with Georgia, serif)
**Body Font:** Sofia Sans Condensed (with Arial Narrow, sans-serif)

**Character:** A cartographer's pairing: a narrow, workmanlike sans with tabular numerals for every catalogue label and reading, and an italic serif reserved for proper names of objects and mathematical variables, as on an engraved atlas plate.

### Hierarchy
- **Display** (Spectral italic 400, clamp(30px, 3.4vw, 46px), line-height 1): the spacetime name in the title cartouche. One per plate.
- **Headline** (Spectral italic 400, 30px): the WebGPU-missing notice heading.
- **Title** (Spectral italic 400, 17px, line-height 1.1): catalogue names in the chart list.
- **Body** (Sofia Sans Condensed 500, 14px, line-height 1.35, tabular numerals): control labels, readouts, values (700 for values). Chart parameters in the cartouche use 600 at 15px with 0.04em tracking.
- **Note** (Sofia Sans Condensed 400, 12px, line-height 1.3): explanatory notes in the key and the symbol key; cartouche description at 13px, max 38ch.
- **Label** (Sofia Sans Condensed 700, 11px, 0.2em, uppercase): key column legends only. The imprint in the top margin uses 11px at 0.16em uppercase; axis names on the scales use 10.5px at 0.14em uppercase.
- **Mark** (Sofia Sans Condensed 600, 12px, 0.03em): labels printed on the chart, in red-ink with a 3px black stroke halo painted behind the fill; the second line drops to 500, 11.5px in plate white.
- **Scale** (Sofia Sans Condensed 500, 10.5px, 0.04em): degree numerals on the margin scales.

### Named Rules
**The Proper Name Rule.** The italic serif is for proper names and variables only: spacetime names, catalogue entries, math symbols such as a/M inside parameter labels. Everything else is the condensed sans.

**The Tabular Rule.** Numerals are always tabular (font-variant-numeric: tabular-nums on the root) so live readouts do not jitter.

## Layout

The page is a three-by-three plate grid: a left margin column (34px), the field column (flexible), and a right margin column (14px); rows are a top margin (34px), the field (flexible) and the key band (auto). On desktop the plate is exactly the viewport height (100dvh), so the field fills everything the key does not.

The top margin carries the azimuth scale and, at the right, the imprint (project name, subtitle, plate number), knocked out of the scale on an ink background. The left margin carries the elevation scale. Major ticks are 9px with degree labels, minor ticks 4px; major steps are chosen so labels sit at least about 80px apart and minor ticks at least 6px apart.

The key band spans the full width under the field, ruled top and bottom, divided into six columns weighted 1.05 / 1.25 / 0.9 / 1.3 / 1.15 / 1.25. Columns are separated by hairline left borders, padded 10px 16px 14px, with an 8px internal gap; the first column has no rule and no left padding.

Responsive: at 1180px and below the key becomes three columns, the plate grows past the viewport and the field is fixed at 70vh. At 720px and below the key becomes two columns, the field is 62vh, the top margin shrinks to 30px and the right margin to 10px, the imprint subtitle and the cartouche description are hidden, and the cartouche widens to 80% of the field.

## Elevation & Depth

The interface is flat. There are no elevation shadows, no blur and no translucent glass. Depth belongs to the render; the chrome is print. Layering is done by knockout instead: the imprint sits on an ink patch that interrupts the scale ticks, the cartouche is an ink panel at 92% opacity with a second ink outline 2px out that separates its rule from the image, slider notches carry a 3px ink ring (`box-shadow: 0 0 0 3px` ink) that cuts the track, and mark labels carry a 3px black stroke halo.

### Named Rules
**The Knockout Rule.** When chrome must separate from what lies under it, cut a gap in ink; never lift it with a shadow or glow.

## Shapes

Square everywhere (0px radius) except two things that are round by meaning: the chart-list dot (8px circle) and the measured-point dot on the chart (2.5px radius). The chart field carries the signature double neatline: an inner 1px rule in plate white at 70% and an outer 1px rule in survey blue, 3px apart. The title cartouche repeats the double rule at smaller scale. Dividers, tracks, ticks and borders are all 1px hairlines; the only heavier strokes are the 2px choice underline, the 2px sweep and the 3px slider notch.

## Components

### Key Column
Ruled, unboxed columns in one band. A legend in the label style on top, controls stacked with an 8px gap, notes at the foot. Columns are fieldsets so each group is labelled for assistive technology.

### Chart List (catalogue entries)
- **Style:** a three-part row: tabular number (700, plate-dim), an 8px hairline circle in graticule blue, and the italic serif name in plate-dim.
- **Hover:** the name brightens to plate white (160ms).
- **Active:** the circle fills red, the number turns red-ink, the name turns plate white.
- **Focus:** a 1px red-ink outline around the name, 2px offset.

### Sliders
- **Style:** a 1px graticule-blue hairline track with graduations (1px by 5px ticks) drawn at round values of each slider's own unit, ends included; the index is a square red notch 3px by 18px with a 3px ink knockout ring.
- **Label row:** label left, value right in 700 plate white; variables inside labels set in the italic serif.
- **Focus:** the global 1px red-ink outline, 3px offset.

### Choices (text radios)
- **Style:** bare text options (15px, 600, plate-dim) 14px apart.
- **Hover:** plate white.
- **Active:** plate white with a 2px red underline.
- **Focus:** 1px red-ink outline, 3px offset.

### Checkboxes
- **Style:** 13px square, 1px graticule-blue border, no fill.
- **Checked:** border turns red and a 7px red square scales in (140ms, cubic-bezier(0.16, 1, 0.3, 1)).

### Readout
A definition list of label (plate-dim, 500) and value (plate, 700, right-aligned) rows with 3px row gap, plus the resolution slider and the symbol key.

### Symbol Key
A hairline-topped list defining every mark printed on the chart: dashed red line for the CPU-traced shadow edge, dot plus leader for a measured point, plate-dim cross for the optical axis. Each sample is a 28 by 10px inline SVG drawn with the same stroke classes as the chart.

### Title Cartouche (signature)
A ruled box inside the neatline at 14px from the top-left corner, max width min(400px, 60%): display name, chart parameters (sans 600 with italic serif variables), and a short plate-dim description. Ink at 92% with a survey-blue rule and an ink outline 2px out. Not interactive.

### Measured Marks (signature)
An SVG overprint above the render: dashed red shadow edge (1.25px, 5 4 dash), red leaders with an elbow ending in a horizontal run, a 2.5px red dot at the measured point, a two-line label (red-ink title, plate-white sub) placed clear of the edge and flipped to stay on the field, and a plate-dim cross on the optical axis. Marks fade out in 90ms when stale and back in over 260ms (cubic-bezier(0.16, 1, 0.3, 1)) when the trace completes.

### Margin Scales (signature)
Live SVG scales in the top and left margins: tick positions are solved from the camera so the degrees are measured, not decorative. Graticule-blue ticks, plate-dim numerals, an uppercase axis name.

### Sweep
A 56px by 2px red bar sliding along the top neatline (1.1s, cubic-bezier(0.65, 0, 0.35, 1), alternating) while the image refines. Static under reduced motion.

## Do's and Don'ts

### Do:
- **Do** keep red for measured or active things only: marks, selected state, current value, focus, refinement sweep.
- **Do** use red-ink, not red, for any red text or 1px focus outline on ink.
- **Do** draw every new figure as a chart plate: black field inside the double neatline (inner plate white at 70%, outer survey blue, 3px apart), with measured scales where an axis exists.
- **Do** define every mark printed on a chart in a symbol key, and label illustrative visuals as illustrative.
- **Do** set proper names and variables in Spectral italic and everything else in Sofia Sans Condensed with tabular numerals.
- **Do** separate overlaid chrome with ink knockouts and 1px rules.
- **Do** graduate sliders at round values of their own unit.
- **Do** honour prefers-reduced-motion: stop the sweep and zero transitions.

### Don't:
- **Don't** use a translucent floating settings panel, glass blur or cyan glow; the glassmorphic dashboard is the anti-reference.
- **Don't** round corners on controls or panels; only the chart dot and measured-point dot are circles.
- **Don't** add elevation shadows; the slider notch's ink ring is a knockout, not a shadow.
- **Don't** print red on structure, headings or decoration.
- **Don't** put ticks or degree numbers in the margins unless they are measured from the camera.
- **Don't** colour-code data without numeric ticks and a plain-language note beside it.

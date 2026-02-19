# Debug Log: UI Overflow / Scrollable Version Strings

## Problem
Long model version strings (e.g. `vaudio_model_Qwen3-1.7B-Base_whisper_turbo_stack_dec0_conv2_12_5_svad_u_20251014_run2_lr5e-05_ga8_bs1_de0_lm0_e0_t0_au0_0_c2_ea0_ew1_cl1_tw1.0_0.95_1e-08_cos_wd0.1_2000_flash_attention_2_0_s80000_q3_svad_F`) overflow out of their card containers in several places.

## Affected Components
1. **ModelDetail.tsx** — `/models/:modelId` page header + version list cards
2. **ReleaseList.tsx** — `/releases` page cards
3. **ReleaseDetail.tsx** — `/releases/:releaseId` page header + info cards

## Goal
- Do NOT truncate — user wants to see full strings
- Make long strings horizontally scrollable within bounded containers
- Keep badges/buttons from being pushed out of view

---

## Attempt 1: `truncate` + `title` tooltips (REVERTED)
- Added `truncate max-w-[...]` + `title` attributes to all long fields
- **Result**: User rejected — wants full text visible, not hidden behind ellipsis

## Attempt 2: `overflow-x-auto whitespace-nowrap` on code blocks
- Replaced `truncate` with `overflow-x-auto whitespace-nowrap` on all `<code>` blocks
- Separated version into its own `<code>` block below badges row
- **Result**: Code blocks rendered but still overflowed cards

### Root cause analysis
`overflow-x-auto` on a block element only activates a scrollbar when the element's computed width < its content width. In a flex layout, if no ancestor constrains the width, the element expands to fit content instead of scrolling.

## Attempt 3: `overflow-hidden` on Card + `min-w-0` on CardContent
- Added `overflow-hidden` to `<Card>` wrapper
- Added `min-w-0` to `<CardContent>`
- **Result**: Card visually clips but inner `code` still doesn't scroll. The `Card` component is `flex flex-col` (see `card.tsx` line 10), so `overflow-hidden` clips the card boundary but doesn't create a width constraint for deeply nested flex children.

### Why `overflow-hidden` on Card alone doesn't work
The DOM structure is:
```
Card (flex-col, overflow-hidden)
  CardContent (px-6, p-4, min-w-0)
    div (flex, justify-between)
      div (flex-1, min-w-0)         ← outer content
        div (flex, gap-3)
          div (icon, shrink-0)
          div (min-w-0, flex-1)     ← inner content wrapper
            div (badges)
            code (overflow-x-auto, whitespace-nowrap)  ← SHOULD scroll
            div (metadata)
```

The `code` element is a **block child** of the inner content wrapper div. For `overflow-x-auto` to activate, the code's width must be bounded. The code's width = parent's width. The parent is a flex item (`min-w-0 flex-1`), which SHOULD be bounded by its flex container.

However, in practice the flex sizing algorithm propagates the `whitespace-nowrap` content's intrinsic width upward through the chain. Even with `min-w-0` at each level, the content size still influences the layout unless an ancestor explicitly clips with `overflow-hidden`.

## Attempt 4 (current): `overflow-hidden` on the inner content wrapper div
- Added `overflow-hidden` to the `div.min-w-0.flex-1` that directly wraps the badges + code + metadata
- This creates a **block formatting context** that forces the div to take its flex-allocated width, NOT its content's intrinsic width
- The `code` inside then has `overflow-x-auto` which creates the scrollbar

### Key insight
In a nested flex layout, `min-w-0` alone is necessary but not sufficient. You also need `overflow-hidden` (or `overflow: auto/scroll`) on a flex item to establish a proper width boundary. Without it, the flex item's "content size" still influences layout calculations.

The fix: **`overflow-hidden` on the flex item that wraps the scrollable content**.

### Files changed
- `ModelDetail.tsx` line 200: `div.min-w-0.flex-1` → `div.min-w-0.flex-1.overflow-hidden`
- `ReleaseList.tsx` line 234: `div.min-w-0.flex-1` → `div.min-w-0.flex-1.overflow-hidden`
- `ReleaseList.tsx` line 222: Card → `Card.overflow-hidden`, icon → `shrink-0`
- `ModelDetail.tsx` line 190: Card → `Card.overflow-hidden`

## Deployment note
The `frontend_build` Docker volume caches old builds. When deploying frontend changes, must delete the volume for new builds to be served:
```bash
docker stop registry-frontend-builder registry-nginx
docker rm registry-frontend-builder registry-nginx
docker volume rm catapult_frontend_build
bash deploy.sh
```
Without this, nginx serves stale JS bundles even after rebuilding the Docker image.

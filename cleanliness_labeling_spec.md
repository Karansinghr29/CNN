# Room Cleanliness — Labeling & Explanation Specification (v2)

Verified from 14 room folders / 141 images in this repository.
Used for (a) labeling training data and (b) producing an explained prediction for any NEW uploaded photo.

Never output a label without a reason. Never use a room number, folder name or a previous label as evidence.

---

## 1. Label definitions

The target is **overall housekeeping cleanliness + reasonable room neatness**, not floor cleanliness alone.

**CLEAN** — the area is reasonably maintained:
- floor/tiles reasonably clean
- no meaningful garbage, waste, food residue, dirty vessels or similar housekeeping dirt
- tenant belongings present but reasonably placed/organised
- kitchen/common area reasonably clean and maintained

**NOT_CLEAN** — visible evidence of any of:
1. garbage / waste / packing material discarded on the floor
2. dirty floor or tiles
3. food residue or dirty surfaces
4. unwashed / dirty vessels, kitchen or common-area dirt
5. excessive or scattered tenant belongings that make the room visibly untidy / not reasonably maintained

**UNCERTAIN** — the image genuinely lacks the evidence needed to judge (wall-only, ceiling-only, window-only or too tight a crop).
Do NOT use UNCERTAIN merely because: the room is dark but still visible, a tenant is present, normal belongings are present, or only one angle is shown.

## 2. Floor vs overall neatness — always judged separately

| floor_tile_condition | overall_neatness | label |
|---|---|---|
| CLEAN | NEAT | CLEAN |
| CLEAN | NOT_NEAT (scattered belongings, packaging, heaps) | NOT_CLEAN |
| DIRTY | any | NOT_CLEAN |
| NOT_VISIBLE / NOT_FULLY_VISIBLE | NOT_ASSESSABLE | UNCERTAIN |

Clean tiles alone never make a room CLEAN. Belongings placed around never by themselves make the tiles dirty — state which one was actually seen.

## 3. Normal tenant belongings (NOT dirt by themselves)

Clothes on the bed · bags/backpacks · shoes and slippers · laptop · bottles · bedding · toiletries ·
clothes hanging or drying · suitcases · storage racks and cartons kept in order · doormats · helmets · books.

These become a NOT_CLEAN reason only when the quantity and spread make the room visibly unmaintained
(e.g. open suitcase spilling onto the floor, clothes over every surface and the floor, cookware and cables piled on the floor).

## 4. Waste signals (strong NOT_CLEAN)

Discarded packing/kraft cover on the floor · torn packaging and plastic wrap left loose · food wrappers and snack waste ·
empty crushed bottles and used tissue left out · loose paper/tissue debris on tiles · dried spills and residue patches ·
unwashed vessels with food residue · grimy or greasy counters and sink surrounds.

Name the waste item explicitly in the reason.

## 5. Kitchen and common areas

Valid training data, judged by the same rules, and never excluded for being a kitchen.
A kitchen, drying rack, washing machine, shoe stand, dustbin or storage rack is NOT a problem in itself.
- Clean → "Kitchen/common area is clean with no visible housekeeping dirt or waste. The household items present are normal use."
- Dirty → name what is dirty: counter residue, unwashed vessels, stained sink, waste on the slab, cloth on the floor.

## 6. Maintenance vs housekeeping

Damp patches · peeling paint · wall holes · wall marks · loose wires · chipped panels · broken plaster · building damage
→ **maintenance observations**, recorded in `maintenance_note`, never the basis of NOT_CLEAN.
The label changes only if housekeeping dirt or waste is also visible.

## 7. Required output format (new uploaded photo)

```
Result: CLEAN | NOT_CLEAN | UNCERTAIN
Why:
- Floor/tiles: <what is actually visible>
- Belongings/neatness: <what is actually visible>
- Waste/dirt: <named items, or "none visible">
- (optional) Maintenance note: <damp/peeling/wires>, not a cleanliness issue
```

Rules: describe only what is visible in that image; never reference another room, photo or folder; plain language the owner can check against the photo; if UNCERTAIN, say exactly what is missing.

### Worked examples (from verified data)

**CLEAN**
```
Result: CLEAN
Why:
- Floor/tiles are clean and clear, with only a doormat and a dustbin in place.
- A bag and cushion are on the bed and shoes are kept under it - normal belongings, reasonably arranged.
- No garbage, food residue or housekeeping dirt is visible.
```

**NOT_CLEAN — waste on the floor**
```
Result: NOT_CLEAN
Why:
- A large crumpled packing cover is lying discarded on the floor beside the suitcase.
- Most of the floor is otherwise clean, but the visible waste material means the area is not clean.
```

**NOT_CLEAN — clean tiles, excessive mess**
```
Result: NOT_CLEAN
Why:
- Floor tiles are visibly clean, with no garbage or dirt.
- Clothes are strewn over both mattresses, an open suitcase is spilling onto the floor and shoes and bags are scattered.
- The overall room appears untidy and not properly maintained.
```

**NOT_CLEAN — kitchen / common area**
```
Result: NOT_CLEAN
Why:
- The kitchen counter has food residue and stains, with unwashed vessels left out.
- A cloth is lying on the floor beside the dustbin.
- The common area therefore appears not properly cleaned.
```

**CLEAN — kitchen / common area**
```
Result: CLEAN
Why:
- Kitchen area is clean with no visible residue or waste on the counter or floor.
- The vessels, dustbins and drying rack present are normal household use.
```

**UNCERTAIN**
```
Result: UNCERTAIN
Why:
- Only a window, wall and air-conditioner are visible in this image.
- No floor, bedding or belongings are shown, so cleanliness cannot be judged.
```

## 8. Manifest schema

`cleanliness_manifest.csv` / `.jsonl`, one row per image, 141 rows, identical records in both files.

| column | meaning |
|---|---|
| room_id | room folder name, unchanged |
| sub_area | `Room` or `Kitchen` (images inside a `Kitchen\` subfolder) |
| image_index | 1-based order within that room+sub_area, sorted by filename |
| image_file | original filename, unchanged |
| image_path | absolute path to the original image (never modified) |
| final_label | **image-level** label: CLEAN / NOT_CLEAN / UNCERTAIN |
| floor_tile_condition | CLEAN / DIRTY / NOT_FULLY_VISIBLE / NOT_VISIBLE |
| overall_neatness | NEAT / NOT_NEAT / NOT_ASSESSABLE |
| evidence_reason | visible evidence for that single image (user-readable) |
| maintenance_note | damp/peeling/wires etc., kept out of the label |
| room_final_label | verified room-level label |
| room_confidence | HIGH / MEDIUM-HIGH / MEDIUM / LOW-MEDIUM |
| room_evidence_reason | why the room as a whole got its label |

Train on `final_label` (image level) with `evidence_reason` as the explanation target.
`room_final_label` is for reporting to the owner and is NOT propagated to individual images.

## 9. Verified dataset counts

- **Images: 141** → CLEAN **105**, NOT_CLEAN **31**, UNCERTAIN **5**
- **Sub-areas:** Room **130**, Kitchen **11** (B41 C 5, C21 C 6)
- **Rooms: 14** → CLEAN **9**, NOT_CLEAN **5**, UNCERTAIN 0
  - CLEAN: A33 A, A34 A, B41 B, B41 D, C21 B, C21 C, C21 D, C31 C, C31 D
  - NOT_CLEAN: A33 C, A34 B, B41 C, C21 A, C31 B

Note on C21 C: the room label is CLEAN by owner decision. Two of its kitchen images
(`112605`, `112616`) remain NOT_CLEAN at image level; that is deliberately not propagated.

### The 5 UNCERTAIN images

| room_id | image_file | why |
|---|---|---|
| A34 B double attach | IMG_20260918_104008.jpg | Only AC unit, window and wall — no floor, bedding or belongings. |
| C21 A single common | C21 A single common.jpg | Mostly curtain, window and wall; too little floor or room shown. |
| C31 C double Common | C31 c double common.jpg | Blank wall, door edge and mirror frame; almost no floor or living area. |
| C31 D single attach | C31 D single attach.jpg | Wall, hook, coat and switchboard only — no floor or room area. |
| C31 D single attach | IMG_20260918_114101.jpg | Wall, hook and coat only; no floor or belongings visible. |

## 10. Known gaps before training

- Class imbalance: 74% of images are CLEAN. Use class weighting or collect more NOT_CLEAN examples.
- Only 5 UNCERTAIN images — too few to learn as a third class; a confidence threshold at inference is likely better.
- NOT_CLEAN examples come from 5 rooms in one property; new apartments may differ.
- No examples of food waste on the floor, overflowing bins, or wet/muddy floors.
- Each folder contains one photo named after the room (e.g. `A33 A Single attach. .jpg`) — confirm whether these are cover shots or duplicates.
- **Split by room, never by image** — images within a room are near-duplicate viewpoints and would leak.
  Stronger still: group by property prefix (A33, A34, B41, C21, C31) and use leave-one-property-out,
  since rooms in the same flat share flooring, furniture and the same photo session.

# External Validation Report — 4 new rooms

> **Prototype — not yet production validated.** 4 rooms from 2 properties cannot establish production performance.

- Generated: 2026-09-24T08:25:31+00:00
- Model artifact: `artifacts/model/demo_cleanliness_model.joblib` (not retrained, not refitted)
- Representation: tile2x2_concat = concat[whole, q1, q2, q3, q4] -> 10240-d
- Threshold: **0.07** — artifact default (median of the five LOPO fold thresholds); NOT tuned on this validation set
- Rooms: A11 B Double Attached, A11 C Double common, A12 B Double attached, A12 C Double common
- Properties: A11, A12

## Overall

| metric | value |
|---|---|
| Total images | 27 |
| Total rooms | 4 |
| Human CLEAN | 18 |
| Human NOT_CLEAN | 9 |
| Correct predictions | 19 |
| Incorrect predictions | 8 |
| False positives (CLEAN → NOT_CLEAN) | 1 |
| **False negatives (NOT_CLEAN → CLEAN)** | **7** |
| Accuracy | 0.7037 |
| NOT_CLEAN precision | 0.6667 |
| NOT_CLEAN recall | 0.2222 |
| CLEAN recall | 0.9444 |

### Confusion matrix

| | pred CLEAN | pred NOT_CLEAN |
|---|---|---|
| **actual CLEAN** | 17 | 1 |
| **actual NOT_CLEAN** | 7 | 2 |

## Per-room results

| room | property | images | human C / NC | correct | FP | FN | accuracy | NC recall |
|---|---|---|---|---|---|---|---|---|
| A11 B Double Attached | A11 | 6 | 6 / 0 | 6 | 0 | **0** | 1.0 | n/a |
| A11 C Double common | A11 | 7 | 2 / 5 | 4 | 0 | **3** | 0.5714 | 0.4 |
| A12 B Double attached | A12 | 8 | 4 / 4 | 4 | 0 | **4** | 0.5 | 0.0 |
| A12 C Double common | A12 | 6 | 6 / 0 | 5 | 1 | **0** | 0.8333 | n/a |

## False negatives — missed dirty rooms (key failure mode)

| room | image | P(NOT_CLEAN) | human reason |
|---|---|---|---|
| A11 C Double common | IMG-20260923-WA0053.jpg | 0.005178 | Towels and cloth are dumped on the floor, cables trail across the tiles and boxes and slippers are left around the desk, so the room looks visibly untidy although the tiles are clean. |
| A11 C Double common | IMG-20260923-WA0057.jpg | 0.0192 | A backpack and bags are left on the floor beside the wardrobe and clothing is draped over the wardrobe edge, so the belongings are not properly arranged even though the floor is clean. |
| A11 C Double common | IMG-20260923-WA0059.jpg | 0.000629 | Towels are dumped on the floor, cables trail across the tiles and a bucket and boxes are pushed under a crowded desk, making the room visibly untidy. |
| A12 B Double attached | IMG-20260923-WA0078.jpg | 0.000342 | Clothes lie loose across the bed and several bags and a backpack are dropped on the floor beside it, so the belongings are unorganized and the room looks untidy. |
| A12 B Double attached | IMG-20260923-WA0079.jpg | 0.050409 | Wide view shows clothes scattered over the bed with bags and luggage left around the floor; the belongings are not properly arranged. |
| A12 B Double attached | IMG-20260923-WA0081.jpg | 0.009187 | Clothes are strewn across the bed and a backpack, tote bags and a suitcase are dropped on the floor around it, making the room visibly untidy. |
| A12 B Double attached | IMG-20260923-WA0083.jpg | 0.00048 | A crumpled quilt and several loose garments are strewn across the bed with a jacket thrown over the bed frame; the bedding and clothes are unorganized. |

## False positives — clean rooms flagged dirty

| room | image | P(NOT_CLEAN) | human reason |
|---|---|---|---|
| A12 C Double common | IMG-20260923-WA0100.jpg | 0.165815 | Desk holds personal items in order with a bag on the chair; the floor is clean and no waste or residue is visible. |

## Per-image detail

| room | image | human | model | P(NOT_CLEAN) | match | human reason |
|---|---|---|---|---|---|---|
| A11 B Double Attached | IMG-20260923-WA0016.jpg | CLEAN | CLEAN | 0.023 | yes | Floor and surfaces are clean, bed made under the mosquito net, desk and dustbin in place; belongings are reasonably arranged and the room looks neat. |
| A11 B Double Attached | IMG-20260923-WA0020.jpg | CLEAN | CLEAN | 0.002 | yes | Wall, doorway and wardrobe are clean with nothing discarded; the room is reasonably organized. |
| A11 B Double Attached | IMG-20260923-WA0022.jpg | CLEAN | CLEAN | 0.004 | yes | Tiles are clean, both beds are made, the desk is tidy and the dustbin and mat are in place; the room looks neat. |
| A11 B Double Attached | IMG-20260923-WA0023.jpg | CLEAN | CLEAN | 0.001 | yes | Jackets are hung on wall hooks rather than left lying around, and the floor and doorway are clean. |
| A11 B Double Attached | IMG-20260923-WA0024.jpg | CLEAN | CLEAN | 0.001 | yes | Shoes are kept on a shelf, the bin is in use and the floor is clean; the wall-base stain is a maintenance issue, not housekeeping dirt. |
| A11 B Double Attached | IMG-20260923-WA0025.jpg | CLEAN | CLEAN | 0.000 | yes | Close floor view shows clean tiles with a mat and a suitcase stored under the bed; no waste or debris. |
| A11 C Double common | IMG-20260923-WA0053.jpg | NOT_CLEAN | CLEAN | 0.005 | no | Towels and cloth are dumped on the floor, cables trail across the tiles and boxes and slippers are left around the desk, so the room looks visibly untidy although the tiles are clean. |
| A11 C Double common | IMG-20260923-WA0054.jpg | NOT_CLEAN | NOT_CLEAN | 0.738 | yes | Cartons, shoes and bags are left on the floor between the beds and the desk is crowded with loose items, making the room visibly untidy despite the clean floor. |
| A11 C Double common | IMG-20260923-WA0055.jpg | CLEAN | CLEAN | 0.002 | yes | AC, curtain and doorway are clean with no waste or dirt, and no belongings are left scattered in view. |
| A11 C Double common | IMG-20260923-WA0056.jpg | NOT_CLEAN | NOT_CLEAN | 0.506 | yes | A laundry basket is overflowing with clothes spilling onto the floor, backpacks are dropped beside it and a plastic sheet lies on the tiles. |
| A11 C Double common | IMG-20260923-WA0057.jpg | NOT_CLEAN | CLEAN | 0.019 | no | A backpack and bags are left on the floor beside the wardrobe and clothing is draped over the wardrobe edge, so the belongings are not properly arranged even though the floor is clean. |
| A11 C Double common | IMG-20260923-WA0058.jpg | CLEAN | CLEAN | 0.001 | yes | Balcony doorway and curtains with a clean floor; no waste and no belongings left scattered. |
| A11 C Double common | IMG-20260923-WA0059.jpg | NOT_CLEAN | CLEAN | 0.001 | no | Towels are dumped on the floor, cables trail across the tiles and a bucket and boxes are pushed under a crowded desk, making the room visibly untidy. |
| A12 B Double attached | IMG-20260923-WA0073.jpg | CLEAN | CLEAN | 0.001 | yes | Bed is made with a few folded clothes, the chair is in place and bags are stored under the bed; tiles are clean and the room is reasonably organized. |
| A12 B Double attached | IMG-20260923-WA0075.jpg | CLEAN | CLEAN | 0.000 | yes | Only one jacket is draped over the bed frame, the rest of the view is clean and clear with nothing scattered. |
| A12 B Double attached | IMG-20260923-WA0078.jpg | NOT_CLEAN | CLEAN | 0.000 | no | Clothes lie loose across the bed and several bags and a backpack are dropped on the floor beside it, so the belongings are unorganized and the room looks untidy. |
| A12 B Double attached | IMG-20260923-WA0079.jpg | NOT_CLEAN | CLEAN | 0.050 | no | Wide view shows clothes scattered over the bed with bags and luggage left around the floor; the belongings are not properly arranged. |
| A12 B Double attached | IMG-20260923-WA0080.jpg | CLEAN | CLEAN | 0.001 | yes | Doorway and desk with a stationery organiser and charger cables kept on the table; floor clean and belongings reasonably contained. |
| A12 B Double attached | IMG-20260923-WA0081.jpg | NOT_CLEAN | CLEAN | 0.009 | no | Clothes are strewn across the bed and a backpack, tote bags and a suitcase are dropped on the floor around it, making the room visibly untidy. |
| A12 B Double attached | IMG-20260923-WA0082.jpg | CLEAN | CLEAN | 0.001 | yes | AC, curtain and wall are clean with no waste, dirt or scattered belongings in view. |
| A12 B Double attached | IMG-20260923-WA0083.jpg | NOT_CLEAN | CLEAN | 0.000 | no | A crumpled quilt and several loose garments are strewn across the bed with a jacket thrown over the bed frame; the bedding and clothes are unorganized. |
| A12 C Double common | IMG-20260923-WA0092.jpg | CLEAN | CLEAN | 0.001 | yes | Both beds are covered with bedding pulled back normally, the wardrobe and desk are in place and the floor is clean with no waste. |
| A12 C Double common | IMG-20260923-WA0093.jpg | CLEAN | CLEAN | 0.001 | yes | Doorway, a bag hung on the chair and a tidy desk; floor is clean and the room looks neat. |
| A12 C Double common | IMG-20260923-WA0098.jpg | CLEAN | CLEAN | 0.003 | yes | Bed headboard, pillows and a floor mat with clean tiles; nothing dirty or discarded. |
| A12 C Double common | IMG-20260923-WA0099.jpg | CLEAN | CLEAN | 0.006 | yes | Bottles and jars are grouped on the table with a storage basket and carton kept underneath - contained rather than scattered; floor clean. |
| A12 C Double common | IMG-20260923-WA0100.jpg | CLEAN | NOT_CLEAN | 0.166 | no | Desk holds personal items in order with a bag on the chair; the floor is clean and no waste or residue is visible. |
| A12 C Double common | IMG-20260923-WA0101.jpg | CLEAN | CLEAN | 0.037 | yes | A snack packet and bottle sit on the side table beside a made bed; belongings are reasonably arranged and the wall and floor are clean. |

## Caveats

- 4 rooms from 2 properties - far too small to establish production performance.
- Human reference labels are single-annotator and should be owner-reviewed.
- The threshold was not tuned on this set; doing so would invalidate it as held-out data.
- These images were NOT added to the training manifest and the model was NOT refitted.

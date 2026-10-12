# Viewer sprites

`sprite_sheet.jpg`, `spider_drones.jpg` (the drill drone used for the
Light unit) and `tank_directions.jpg` (the Ranged tank from each side) are
the source artwork (AI-generated, CC0 1.0, see
`THIRD_PARTY_NOTICES.md`). To change the sprites, edit or replace them, then:

```bash
python tools/sprites/cut_sheet.py   # sheet -> microrts_pz/render/sprites/*.png (needs scipy)
python tools/sprites/embed.py       # PNGs  -> WebP data URIs inside viewer.html
```

The art faces right; the viewer mirrors player 1's units so both armies
face each other. The tank instead has a sprite per side
(`Ranged_<player>_<N|E|S|W>`) and turns toward where it shoots or moves. Grid positions and the cell-to-sprite mapping live at the
top of `cut_sheet.py`.

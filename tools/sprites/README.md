# Viewer sprites

`sprite_sheet.jpg` is the source artwork (AI-generated, CC0 1.0, see
`THIRD_PARTY_NOTICES.md`). To change the sprites, edit or replace it, then:

```bash
python tools/sprites/cut_sheet.py   # sheet -> microrts_pz/render/sprites/*.png (needs scipy)
python tools/sprites/embed.py       # PNGs  -> WebP data URIs inside viewer.html
```

The art faces left; the viewer mirrors player 0's vehicles so both armies
face each other, and draws the glowing team rings itself.

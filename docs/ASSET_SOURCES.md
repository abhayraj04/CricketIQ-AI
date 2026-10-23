# Team and Player Assets

## Current status

- No team logos or player portraits are bundled. The dashboard uses short-name/initial badges instead of copying protected artwork.
- The official IPL team directory is the source for the 2026 participant names: <https://www.iplt20.com/teams> (checked 2026-10-21).
- The official fixtures page currently reports that fixtures are coming soon: <https://www.iplt20.com/matches/fixtures> (checked 2026-10-21). This does not verify participation or a schedule for 2027.
- The IPL/BCCI site terms state that site content and trademarks are owned or controlled by BCCI and that reproduction/use requires prior written authorization: <https://www.iplt20.com/terms-and-conditions> (checked 2026-10-21).
- Official colors were not copied from logos or guessed. `primary_color`, `secondary_color`, and `logo_path` remain `null` until a permitted, verified source is approved. Dashboard chart colors are an analytics palette, not claimed team branding.
- Player portraits are not bundled. `config/player_images.json` intentionally has an empty mapping; the UI uses initials from exact profile names. This also avoids misidentifying players where profile IDs collide.

## Verification before adding assets

For each new local asset, record its exact source URL, owner, license/permission, verification date, and the unambiguous team or player key in the mapping file. Do not add assets whose redistribution rights or identity are unclear. Keep historical franchise names distinct unless a documented identity mapping confirms they are the same franchise.
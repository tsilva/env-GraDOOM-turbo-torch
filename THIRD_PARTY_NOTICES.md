# Third-party source policy

GraDOOM's original code is MIT-licensed. Bundled third-party resources retain the separate licenses listed below. No ViZDoom, ZDoom, or Doom engine source code has been copied into this repository at this stage.

Reference source may be introduced when it advances semantic parity or training throughput. Every copied or adapted file must record its upstream repository, revision, original path, applicable license, local modifications, and redistribution obligations before it is committed. The project license must be changed before accepting source whose terms are incompatible with the current license.

WAD, PK3, and other game-data files are external test/runtime inputs. Doom II data must never be committed or redistributed by GraDOOM.

## ZDoom BulletChip resources

- Local component: `src/gradoom/assets/zdoom_bullet_chips.json`.
- Upstream repository: `https://github.com/ZDoom/gzdoom`.
- Upstream revision: `092b9c0515c2861270cde175cd8eaa30a253c8b1`.
- Original paths: `wadsrc/static/graphics/chip1.png` through `chip5.png`.
- Copyright: Copyright (c) 1998-2025 ZDoom and GZDoom teams and contributors. Git history records Randy Heit introducing these resources in ZDoom SVN r1082 on 2008-07-23.
- License: GPL-3.0-only; the complete license is provided in `LICENSES/GPL-3.0-only.txt`.
- Local modifications: the five 7-pixel-wide grayscale PNG pixel planes and their `grAb` offsets were decoded into JSON. PNG container metadata was removed, zero remains transparent, and no source pixel values were changed. Each source PNG SHA-256 is recorded beside its decoded data.
- Redistribution: preserve this notice and the GPL-3.0 license, provide the JSON resource as its preferred modifiable source form, and identify further modifications. GraDOOM's original MIT-licensed code remains a separately licensed work.

## Branding assets

The artwork under `image-assets/` and the root `logo.png` is AI-generated project branding inspired by DOOM's visual identity, including a stylized GraDOOM wordmark and a Doom Slayer/Praetor-suit likeness. It does not contain extracted game artwork, WAD data, or other official game asset files.

DOOM, Doom Slayer, and related names, characters, logos, and marks are owned by their respective rights holders. GraDOOM is an independent project and is not affiliated with or endorsed by id Software, Bethesda Softworks, or ZeniMax Media. Redistributors are responsible for evaluating trademark and character-licensing requirements for their intended use.

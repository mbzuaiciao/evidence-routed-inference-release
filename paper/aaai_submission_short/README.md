# AAAI-27 Main-Track Manuscript and Supplement

This directory contains an anonymous AAAI-27 main-track manuscript
(`main.tex`) and an independently compilable anonymous supplement
(`supplement.tex`) derived from the authoritative long-form manuscript at
commit
`dae85c33adae46fc41728209ac279db91351f2c5`.

Both roots use the official AAAI-27 author-kit files `aaai2027.sty` and
`aaai2027.bst` in anonymous `submission` mode. The copied style and
bibliography files must remain byte-identical to the author kit and must not be
modified.

Build with:

```sh
make main
make supplement
```

The main manuscript is also packaged as a single-file Prism source in
`eri_aaai27_main.tex`; that file is the root named by the main Prism archive.
The independently flattened supplement root is
`eri_aaai27_supplement.tex`.

Create deterministic Prism archives with:

```sh
make archives
```

The archive preparation script recursively inlines the modular development
sources. This follows the author kit's requirement that the final submission
use one root TeX file without `\input`; only the bibliography and official
custom style files remain separate.

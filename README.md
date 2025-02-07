# DLkitty

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://CellBH.github.io/DLkitty.jl/stable)
[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://CellBH.github.io/DLkitty.jl/dev)
[![Build Status](https://github.com/CellBH/DLkitty.jl/workflows/Test/badge.svg)](https://github.com/CellBH/DLkitty.jl/actions)
[![Test workflow status](https://github.com/CellBH/DLkitty.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/CellBH/DLkitty.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Lint workflow Status](https://github.com/CellBH/DLkitty.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/CellBH/DLkitty.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/CellBH/DLkitty.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/CellBH/DLkitty.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/CellBH/DLkitty.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/CellBH/DLkitty.jl)
[![DOI](https://zenodo.org/badge/DOI/FIXME)](https://doi.org/FIXME)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

## How to modify:

- Change training/preprocessing functions in [`src/execute.jl`](src/execute.jl)
- Change model structure in [`src/neural_net_model.jl`](src/neural_net_model.jl)

### Training and Use

```julia
using DLkitty

df = kcat_table_train_and_valid()
all_ngrams = load_all_sequence_ngrams(3)
trained_model = train(df, all_ngrams; n_samples=1000, n_epochs=100)

datum = (;
    SubstrateSMILES = ["C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O"],
    ProteinSequences = ["MAAVKASTSKATRPWYSHPVYARYWQHYHQAMAWMQSHHNAYRKAVESCFNLPWYLPSALLPQSSYDNEAAYPQSFYDHHVAWQDYPCSSSHFRRSGQHPRYSSRIQASTKEDQALSKEEEMETESDAEVECDLSNMEITEELRQYFAETERHREERRRQQQLDAERLDSYVNADHDLYCNTRRSVEAPTERPGERRQAEMKRLYGDSAAKIQAMEAAVQLSFDKHCDRKQPKYWPVIPLKF"],
    Temperature = 300.0,
    pH = 7.5
)
dist = predict_kcat_dist(trained_model, all_ngrams, datum)
```
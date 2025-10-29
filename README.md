# Dose optimization project

Tämä projekti optimoi PIX-annoksen ja alku-pH:n raakaveden laadun perusteella tavoitesameuden saavuttamiseksi.

# Rakenne

- `src/`: koodi (esikäsittely, koulutus, optimointi)
- `data/`: esimerkkidata (ei oikeaa tuotantodataa)
- `models/`: mallien tallennus
- `notebooks/`: valinnaiset visualisoinnit

# Asennus

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

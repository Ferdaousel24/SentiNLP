"""Prétraitement partagé par le notebook (entraînement) et l'API (prédiction).

Il est indispensable d'appliquer EXACTEMENT le même nettoyage à l'entraînement
et à la prédiction, sinon le modèle reçoit des textes qu'il n'a jamais vus.
"""
import re
import unicodedata

# Mots vides français + anglais. On GARDE volontairement les négations
# (pas, ne, jamais, rien, sans, non...) car elles changent le sens d'un avis :
# "pas bon" ne doit pas devenir "bon".
STOP_WORDS = set("""
le la les l un une des du de d et ou mais donc or ni car à a au aux en dans sur sous par pour avec chez
ce cet cette ces ca ça cela celui celle ceux celles je j tu il elle on nous vous ils elles me m te t se s
mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs qui que qu quoi dont où y
est es sont suis etes êtes sommes etait était ete été etre être ai as avons avez ont avait avoir
the a an and or but of to in on at for with is are was were be been this that these those it its
i you he she we they me my your his her our their
""".split()) - {"pas", "ne", "non", "jamais", "rien", "sans", "plus"}

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_NON_LETTRES = re.compile(r"[^a-zà-ÿœæ\s]")  # supprime ponctuation, chiffres, emojis...
_ESPACES = re.compile(r"\s+")


def nettoyer_texte(texte: str) -> str:
    """minuscules -> sans liens -> sans ponctuation/chiffres -> sans mots vides."""
    if not isinstance(texte, str):
        return ""
    t = texte.lower()
    t = _URL.sub(" ", t)
    t = t.replace("’", " ").replace("'", " ")  # "l'hôtel" -> "l hôtel"
    t = unicodedata.normalize("NFC", t)
    t = _NON_LETTRES.sub(" ", t)
    mots = [m for m in _ESPACES.split(t) if m and m not in STOP_WORDS and len(m) > 1]
    return " ".join(mots)

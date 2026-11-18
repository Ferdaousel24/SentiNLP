"""API Flask : POST /predict  {"texte": "..."}  ->  {"sentiment": "positif", "confiance": 0.87}

Lancer :  python app.py
Test   :  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
               -d '{"texte": "Produit excellent, je recommande"}'
"""
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from nettoyage import nettoyer_texte  # MÊME nettoyage qu'à l'entraînement

app = Flask(__name__)
CORS(app)  # autorise les appels depuis la page web (autre origine)

modele = joblib.load("modele_sentiment.joblib")
clf = modele.named_steps["clf"]


def confiance(texte_propre: str, indice: int):
    """Score de confiance en [0, 1] si le classifieur le permet."""
    if hasattr(modele, "predict_proba"):                 # Naive Bayes, Random Forest
        return float(modele.predict_proba([texte_propre])[0][indice])
    if hasattr(modele, "decision_function"):             # LinearSVC : softmax des marges
        d = modele.decision_function([texte_propre])[0]
        e = np.exp(d - d.max())
        return float((e / e.sum())[indice])
    return None


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    texte = str(data.get("texte", "")).strip()
    if not texte:
        return jsonify({"erreur": "Le champ 'texte' est vide."}), 400

    propre = nettoyer_texte(texte)
    if not propre:
        return jsonify({"erreur": "Le texte ne contient aucun mot exploitable."}), 400

    sentiment = modele.predict([propre])[0]
    indice = list(clf.classes_).index(sentiment)
    return jsonify({"sentiment": str(sentiment), "confiance": confiance(propre, indice)})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "classes": [str(c) for c in clf.classes_]})


if __name__ == "__main__":
    app.run(port=5000, debug=False)
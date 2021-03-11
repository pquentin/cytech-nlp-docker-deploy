# Déploiement d'un modèle Keras avec Docker et FastAPI

L'objet de ce TP est de déployer le modèle d'analyse d'opinion que
vous venez de développer. Le déployer, ça veut dire le rendre
utilisable facilement pour avoir accès à ses prédictions. On va donc
passer par un microservice, un petit serveur qui aura comme seule
tâche de fournir des prédictions étant donné des textes de critique de
film en entrée.

Nous commençons par un développement en local, puis nous utiliserons
Docker (toujours en local), ce qui vous permettra de facilement
reprendre cet exemple en dehors de ce cours si vous le souhaitez.

Nous allons procéder par étapes, afin de voir les problèmes au plus
vite. Vouloir commencer tout de suite avec Docker serait au final une
perte de temps, il est beaucoup plus facile de traiter les problèmes
en isolation puis d'assembler les composants par la suite. (Vous avez
du vous en rendre compte, depuis le temps que vous êtes à CY Tech !)

## Installation des dépendances

Avec Pytyon 3.8, créer un virtualenv classique :

```bash
python3.8 -m venv tp-nlp-deploy
source tp-nlp-deploy/bin/activate
pip install fastapi uvicorn requests
pip install tensorflow
```

(Pourquoi Python 3.8 ? Parce que TensorFlow ne fournit pas encore de
paquets précompilés pour Python 3.9)

Oui, TensorFlow prend beaucoup de place. On pourrait s'en passer, ce
qui réduirait beaucoup la taille de l'image Docker mais il aurait
fallu éviter Keras pour implémenter le tokenizer, et passer par
exemple par spaCy.

## API locale simple

Essayons de faire fonctionner FastAPI. Écrivez le code suivant dans un
fichier app/main.py:

```python3
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
```


Lancez l'application avec `uvicorn app.main:app --reload`, puis ouvrez
<http://127.0.0.1:8000/>. Vous devriez voir le message Hello World
dans votre navigateur.

## Récupérer les modèles

Dézippez dans `models/` le fichier `imdb_reviews_cn_model.zip` que vous
avez récupéré lors du TP précédent. La commande `tree models` devrait
vous afficher ceci :

```
models
└── imdb_reviews_cnn_model
    ├── 1
    │   ├── assets
    │   ├── saved_model.pb
    │   └── variables
    │       ├── variables.data-00000-of-00002
    │       ├── variables.data-00001-of-00002
    │       └── variables.index
    └── tokenizer.json
```

## TensorFlow Serving

Nous allons construire et lancer une image Docker TensorFlow Serving
avec ce modèle :

```
docker-compse up tf-serving
```

Notre code FastAPI va communiquer avec cet instance de TensorFlow
Serving sur le port 8501 pour obtenir les prédictions.

## Preprocessing

Cependant, tout le travail ne se fait pas dans TensorFlow Serving, il
faut aussi préparer le texte comme nous l'avons fait lors du TP. En
d'autres termes, il faut transformer le texte en séquence d'entier avec
padding.

 * Désérialiser le vectorizer via
   `tf.keras.preprocessing.text.tokenizer_from_json` (qui prend une
   chaîne de caractères, c'est à vous d'ouvrir le fichier et d'appeler
   `read()`)
 * Écrire une fonction `preprocess()` qui étant donné un texte comme
   "This was the biggest hit movie of 1971", renvoie une liste Python
   d'entiers à envoyer à TensorFlow (avec `texts_to_sequences` et
   `pad_sequences` comme lors du TP.

Pour tester que ces valeurs sont correctes, les utiliser pour appeler
TensorFlow Serving, sans passer par FastAPI pour le moment :

```bash
curl -d '{"instances": [[0, 0, ...]]}' -X POST http://localhost:8501/v1/models/imdb_reviews_cnn_model:predict
```

(Même si les prédictions sont fausses parce que le texte est très
différent des textes fournis en entrée sur lequel le modèle a été
entraîné, ce n'est pas grave parce que ce n'est pas l'objet de ce TP.)

## Intégration à l'app FastAPI

Nous voulons maintenant que FastAPI communique avec TensorFlow
Serving.

Ajoutez à `app/main.py` cette fonction dans qui va s'exécuter au
démarrage de votre application FastAPI pour instancier le tokenizer
Keras d'après le code écrit plus haut :

```python3
@app.on_event("startup")
def startup_event():
    app.state.tokenizer = ...
```

Ensuite, à la manière de `root` plus haut, implémenter un endpoint
`predict` qui accepte un texte, le transforme avec la fonction
`preprocess()`, appelle TensorFlow, puis renvoie `True` si la prédiction
est supérieure à 0.5, `False` sinon.

Lancez l'application comme indiqué plus haut, puis testez une requête
:

```bash
curl -X POST http://localhost:8000/v1/predict -d '{"text": "This was the biggest hit movie of 1971"}
```

## Préparatifs pour Docker

Il y a un changement à apporter pour que votre code fonctionne dans
Docker.

Premièrement, la communication avec FastAPI ne se fera plus par
localhost, mais avec le nom de l'image TensorFlow Serving au sein du
réseau Docker, qui vous sera passée dans la variable d'environnement
`TF_HOST` (cf. le fichier `docker-compose.yml`)

Il faut donc modifier votre code pour lire `TF_HOST`, en utilisant
`localhost` si la variable d'environnement n'est pas définie :

```python3
import os

TF_HOST: str = os.environ.get("TF_HOST", "localhost")
```

Et ensuite, au moment de la requête, définissez l'URL comme suit, en
utilisant par exemple une f-string Python :

```python3
f"http://{TF_HOST}:8501/v1/models/imdb_reviews_cnn_model:predict"
```

## FastAPI dans Docker

Fantastique ! Nous sommes prêts pour passer à Docker. Construisons
l'image de notre serveur et lançons la via-docker-compose :

```bash
docker-compose up fastapi
```

Vous pouvez alors relancer une requête qui cette fois va parler à notre
serveur dans Docker :

```bash
curl -X POST http://localhost:8000/v1/predict -d '{"text": "This was the biggest hit movie of 1971"}'
```

Et tester les performances de prédiction. Pour ceci, installez wrk :

 * Linux : https://github.com/wg/wrk/wiki/Installing-wrk-on-Linux
 * macOS: https://github.com/wg/wrk/wiki/Installing-wrk-on-OS-X
 * Windows 10 : https://github.com/wg/wrk/wiki/Installing-wrk-on-Windows-10

Puis testez les performances :

```bash
wrk -t 2 -c 10 http://127.0.0.1:8000/v1/predict -s post.lua
```

Combien de requêtes par seconde obtenez-vous ? J'en ai obtenu environ
250 sur mon laptop.

## Notes

Envoyez-moi votre fichier app.py par mail, un par groupe.

Pour des points bonus, faire la même chose avec BERT.

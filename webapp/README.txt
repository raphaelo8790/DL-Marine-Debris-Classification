Marine Debris Classification - web app to try the model on your own photos
============================================================================

1. Install once (in the same environment you train in):
       pip install gradio

2. Start it (from this folder):
       python app.py
   then open  http://127.0.0.1:7860  in your browser.

3. Pick a model (every Models\<name>_best.keras is listed, newest first), upload a photo,
   press Analyse.

What you get
  - Viewpoint the model sees (underwater / shore / drone / satellite). You can override it if a
    photo looks nothing like the training data.
  - Trash found / not found, with a probability. v3 models use their dedicated "trash present?"
    neuron; older models combine their class scores.
  - The debris classes that pass their threshold (thresholds from Models\Evaluations\eval_<name>; 0.5 if the
    model hasn't been evaluated yet).
  - Your photo cut into tiles like the training data (underwater 3x2, shore 3x3, drone ~640 px
    pieces, satellite whole), each tile tinted red by its trash probability.

Settings at the top of app.py
  RUNS / PREPARED  found automatically from where app.py sits (Models and Datasets\prepared)
  USE_GPU = False  runs on the CPU, which is fine for single photos and safe while training runs.

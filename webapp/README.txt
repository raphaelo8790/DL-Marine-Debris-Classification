Marine Debris Classification - web app to try the model on your own photos
============================================================================

Requirements
  Python 3.10 or newer, then install once:
       pip install torch keras numpy opencv-python gradio
  (Keras runs on the PyTorch backend. The app runs on the CPU, so no GPU or CUDA is needed.)
  The trained models must be in the Models folder (Models\<name>_best.keras). The datasets are
  not needed.

How to run
1. Start it (from this folder):
       python app.py
   then open  http://127.0.0.1:7860  in your browser.

2. Pick a model (every Models\<name>_best.keras is listed, newest first), upload a photo,
   press Analyse.

What you get
  - Viewpoint the model sees (underwater / shore / drone / satellite). You can override it if a
    photo looks nothing like the training data.
  - Trash found / not found, with a probability, from the model's dedicated "trash present?"
    neuron.
  - The debris classes that pass their threshold (thresholds from
    Models\Evaluations\eval_<name>\thresholds.json; 0.5 if the model hasn't been evaluated).
  - Your photo cut into tiles like the training data (underwater 3x2, shore 3x3, drone ~640 px
    pieces, satellite whole), each tile tinted red by its trash probability.

Settings at the top of app.py
  The Models folder is found automatically from where app.py sits.
  USE_GPU = False  runs on the CPU, which is fine for single photos.
